"""Find dish/plate/bowl pixels in photos under datasets/.

Two models are combined:

1. CLIPSeg (text-prompted segmentation) is asked "a plate" / "a bowl" to
   roughly locate each container, regardless of table/background style
   -- this solves the problem of background clutter (patterned mats,
   other objects) looking similar to a plate to plain color/texture
   heuristics. Its output is turned into a bounding box per detected
   region rather than used as a mask directly, since CLIPSeg's raw
   pixel output is low-resolution and blocky once upsampled.
2. SAM (Segment Anything) takes those boxes as prompts and produces a
   pixel-precise mask for each one, giving clean plate/bowl edges
   (including under any food covering part of the dish) instead of
   CLIPSeg's coarse, sometimes-broken outline.

Results (mask + overlay + per-image pixel stats) are written to output/.
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    SamModel,
    SamProcessor,
)

DATASET_DIR = "datasets"
OUTPUT_DIR = "output"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

CLIPSEG_MODEL_NAME = "CIDAS/clipseg-rd64-refined"
CLIPSEG_PROMPTS = ["a plate", "a bowl"]
CLIPSEG_PROB_THRESHOLD = 0.5
# CLIPSeg's transposed-convolution decoder hits an MPS bug on this model; CPU is fast enough (~1-2s/image).
CLIPSEG_DEVICE = "cpu"

SAM_MODEL_NAME = "facebook/sam-vit-base"
SAM_DEVICE = "cpu"

MIN_REGION_AREA_FRACTION = 0.01  # ignore CLIPSeg regions / final blobs smaller than 1% of the image
BOX_DEDUP_IOU_THRESHOLD = 0.85  # drop near-duplicate boxes so SAM isn't asked for the same object twice


def load_models():
    clipseg_processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL_NAME)
    clipseg_model = (
        CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL_NAME).to(CLIPSEG_DEVICE).eval()
    )
    sam_processor = SamProcessor.from_pretrained(SAM_MODEL_NAME)
    sam_model = SamModel.from_pretrained(SAM_MODEL_NAME).to(SAM_DEVICE).eval()
    return clipseg_processor, clipseg_model, sam_processor, sam_model


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_boxes(boxes):
    kept = []
    for box in boxes:
        if any(box_iou(box, other) >= BOX_DEDUP_IOU_THRESHOLD for other in kept):
            continue
        kept.append(box)
    return kept


def clipseg_boxes(image_bgr: np.ndarray, processor, model) -> list:
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    height, width = image_bgr.shape[:2]
    min_area = MIN_REGION_AREA_FRACTION * height * width

    boxes = []
    for prompt in CLIPSEG_PROMPTS:
        inputs = processor(text=[prompt], images=[pil_image], return_tensors="pt").to(CLIPSEG_DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # single prompt -> drop the batch dim
        prob_full = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = (prob_full >= CLIPSEG_PROB_THRESHOLD).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])

    return dedupe_boxes(boxes)


def sam_masks_for_boxes(image_bgr: np.ndarray, boxes: list, processor, model) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    combined = np.zeros((height, width), dtype=np.uint8)
    if not boxes:
        return combined

    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_image, input_boxes=[boxes], return_tensors="pt").to(SAM_DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]

    for i in range(masks.shape[0]):
        mask = (masks[i, 0].numpy().astype(np.uint8)) * 255
        combined = cv2.bitwise_or(combined, mask)

    return combined


def find_dish_hulls(image_bgr: np.ndarray, clipseg_processor, clipseg_model, sam_processor, sam_model) -> list:
    """Return one filled convex-hull mask (uint8 0/255) per detected dish."""
    boxes = clipseg_boxes(image_bgr, clipseg_processor, clipseg_model)
    mask = sam_masks_for_boxes(image_bgr, boxes, sam_processor, sam_model)

    # SAM can leave gaps where food breaks the "plate" boundary -- either
    # fully enclosed holes (e.g. a bright reflection) or, when the food
    # pile reaches the rim, a concave bite cut in from the edge. A plain
    # contour fill only closes the first kind, so each blob's outer edge
    # is replaced by its convex hull instead: plates/bowls are round or
    # rounded-rect, so the hull is a good stand-in for the true outline
    # and it closes both kinds of gap at once. SAM keeps distinct dishes
    # as separate blobs (unlike CLIPSeg's coarser, sometimes-bridged
    # regions), so hulling each one individually doesn't fuse them.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_REGION_AREA_FRACTION * image_bgr.shape[0] * image_bgr.shape[1]

    hulls = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        hull_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(contour), 255)
        hulls.append(hull_mask)

    return hulls


def find_plate_mask(image_bgr: np.ndarray, clipseg_processor, clipseg_model, sam_processor, sam_model) -> np.ndarray:
    hulls = find_dish_hulls(image_bgr, clipseg_processor, clipseg_model, sam_processor, sam_model)
    filled_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for hull in hulls:
        filled_mask = cv2.bitwise_or(filled_mask, hull)
    return filled_mask


def process_image(path: str, output_dir: str, clipseg_processor, clipseg_model, sam_processor, sam_model) -> None:
    image = cv2.imread(path)
    if image is None:
        print(f"skip (unreadable): {path}")
        return

    mask = find_plate_mask(image, clipseg_processor, clipseg_model, sam_processor, sam_model)

    plate_pixels = int(np.count_nonzero(mask))
    total_pixels = mask.shape[0] * mask.shape[1]
    percent = 100 * plate_pixels / total_pixels

    overlay = image.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)

    stem = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{stem}_mask.png"), mask)
    cv2.imwrite(os.path.join(output_dir, f"{stem}_overlay.png"), overlay)

    print(f"{os.path.basename(path)}: {plate_pixels} plate pixels ({percent:.1f}% of image)")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted(
        f for f in os.listdir(DATASET_DIR) if f.lower().endswith(IMAGE_EXTS)
    )
    if not images:
        print(f"no images found in {DATASET_DIR}/")
        return

    clipseg_processor, clipseg_model, sam_processor, sam_model = load_models()

    for filename in images:
        process_image(
            os.path.join(DATASET_DIR, filename), OUTPUT_DIR, clipseg_processor, clipseg_model, sam_processor, sam_model
        )


if __name__ == "__main__":
    main()
