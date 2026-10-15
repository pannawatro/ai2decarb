from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
IMAGE_PATH = Path("datasets/8.jpg")
OUTPUT_PATH = Path("result.jpg")
MASK_OUTPUT_PATH = Path("mask.jpg")
OVERLAY_OUTPUT_PATH = Path("overlay.jpg")
OVERLAY_COLOR = np.array([255, 0, 0], dtype=np.float32)  # yellow, RGB
OVERLAY_ALPHA = 0.5


def load_model(model_id: str):
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).eval()
    return processor, model


def run_inference(processor, model, img: Image.Image) -> np.ndarray:
    with torch.no_grad():
        logits = model(**processor(images=img, return_tensors="pt").to(model.device)).logits
    pred = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    return pred.argmax(dim=1)[0].cpu().numpy()


def get_food_mask(id2label: dict, pred: np.ndarray) -> np.ndarray:
    food_id = next(
        (class_id for class_id, label in id2label.items() if "food" in label.lower()),
        None,
    )
    if food_id is None:
        raise ValueError("No 'food' class found in this model.")
    return pred == food_id


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)
    print(f"Saved mask → {output_path}")


def save_overlay(img: Image.Image, mask: np.ndarray, output_path: Path) -> None:
    overlay = np.array(img).astype(np.float32)
    overlay[mask] = (1 - OVERLAY_ALPHA) * overlay[mask] + OVERLAY_ALPHA * OVERLAY_COLOR
    Image.fromarray(overlay.astype(np.uint8)).save(output_path)
    print(f"Saved overlay → {output_path}")


def visualize(img: Image.Image, mask: np.ndarray, output_path: Path) -> None:
    img_arr = np.array(img)
    food_only = img_arr.copy()
    food_only[~mask] = 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, image, title, kwargs in zip(
        axes,
        [img_arr, mask, food_only],
        ["Original", "Food Mask", "Food Only"],
        [{}, {"cmap": "gray"}, {}],
    ):

        ax.imshow(image, **kwargs)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    processor, model = load_model(MODEL_ID)
    img = Image.open(IMAGE_PATH).convert("RGB")

    pred = run_inference(processor, model, img)
    mask = get_food_mask(model.config.id2label, pred)
    save_mask(mask, MASK_OUTPUT_PATH)
    save_overlay(img, mask, OVERLAY_OUTPUT_PATH)
    visualize(img, mask, OUTPUT_PATH)
