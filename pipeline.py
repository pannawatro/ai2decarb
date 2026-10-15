"""Leftover estimation from a before/after pair of meal photos.

Dish masks (main.py: CLIPSeg + SAM) and the food mask (leftover.py: SegFormer)
are combined per dish. Each dish's occupancy is food-pixels-inside-dish divided
by dish pixels, so the dish acts as the scale reference and the two photos need
not share resolution or framing. Leftover = occupancy_after / occupancy_before
(before photo = 100%). Percentages are 2D area ratios, not volume.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

import dish as dish_seg
import leftover as food_seg


MAX_SIDE = 1024  # all outputs are area ratios, so full resolution is wasted compute (SAM resizes to 1024 anyway)


def downscale(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1:
        return image_bgr
    return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


MODEL_VERSION = "area-proxy-0.1"  # food-area / dish-area ratio used as a stand-in for weight ratio
MIN_BEFORE_OCCUPANCY = 0.02  # below this the before photo has no usable food baseline
MAX_PLAUSIBLE_RATIO = 1.5  # after/before food area above this -> probably a different plate


class UnusableImage(ValueError):
    """Photos cannot be compared (no plate, no food baseline, or not the same plate)."""


@dataclass
class Dish:
    mask: np.ndarray  # bool, full image size
    center: tuple  # (x, y) normalized to image size
    area_frac: float  # dish area / image area
    dish_px: int
    food_px: int

    @property
    def occupancy(self) -> float:
        return self.food_px / self.dish_px if self.dish_px else 0.0


class LeftoverPipeline:
    def __init__(self):
        self.clipseg_processor, self.clipseg_model, self.sam_processor, self.sam_model = dish_seg.load_models()
        self.segformer_processor, self.segformer_model = food_seg.load_model(food_seg.MODEL_ID)
        if torch.backends.mps.is_available():  # SegFormer runs fine on MPS; CLIPSeg/SAM must stay on CPU
            self.segformer_model = self.segformer_model.to("mps")

    def analyze(self, image_bgr: np.ndarray) -> tuple:
        """Return (dishes, food_mask) for one image."""
        image_bgr = downscale(image_bgr)
        h, w = image_bgr.shape[:2]
        hulls = dish_seg.find_dish_hulls(
            image_bgr, self.clipseg_processor, self.clipseg_model, self.sam_processor, self.sam_model
        )
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        pred = food_seg.run_inference(self.segformer_processor, self.segformer_model, pil)
        food = food_seg.get_food_mask(self.segformer_model.config.id2label, pred)

        dishes = []
        for hull in hulls:
            mask = hull > 0
            ys, xs = np.nonzero(mask)
            dishes.append(
                Dish(
                    mask=mask,
                    center=(xs.mean() / w, ys.mean() / h),
                    area_frac=mask.sum() / (h * w),
                    dish_px=int(mask.sum()),
                    food_px=int((mask & food).sum()),
                )
            )
        return dishes, food

    def compare(self, before_bgr: np.ndarray, after_bgr: np.ndarray) -> dict:
        before, _ = self.analyze(before_bgr)
        after, _ = self.analyze(after_bgr)
        if not before:
            raise UnusableImage("no plate detected in image (before-meal photo)")
        if not after:
            raise UnusableImage("no plate detected in leftover_image (after-meal photo)")

        pairs, unmatched_before, unmatched_after = match_dishes(before, after)
        total_before = sum(b.occupancy for b, _ in pairs)
        total_after = sum(a.occupancy for _, a in pairs)
        if total_before < MIN_BEFORE_OCCUPANCY:
            raise UnusableImage("no food detected on the plate in image (before-meal photo)")

        # Overall: food occupancy summed over matched dishes (each dish = 100% of its own footprint).
        raw_ratio = total_after / total_before
        if raw_ratio > MAX_PLAUSIBLE_RATIO:
            raise UnusableImage("leftover_image has much more food than image; it does not look like the same plate")
        overall = min(max(raw_ratio * 100, 0.0), 100.0)

        dish_results = [
            {
                "before_occupancy": round(b.occupancy, 4),
                "after_occupancy": round(a.occupancy, 4),
                "leftover_percent": round(min(a.occupancy / b.occupancy * 100, 100.0), 1) if b.occupancy > 0 else None,
            }
            for b, a in pairs
        ]
        warns = warnings(before, after, unmatched_before, unmatched_after)

        return {
            "leftover_percent": round(overall, 1),
            "confidence": round(confidence(before, after, total_before, raw_ratio), 2),
            "model_version": MODEL_VERSION,
            "dishes": dish_results,
            "warnings": warns,
        }


def confidence(before: list, after: list, total_before: float, raw_ratio: float) -> float:
    """Heuristic 0-1 score. Not calibrated against ground truth (none exists yet)."""
    c = 1.0
    if len(before) != len(after):
        c *= 0.5
    if total_before < 0.15:  # little food in the baseline photo -> ratio is noisy
        c *= 0.6
    if raw_ratio > 1.15:  # "more food after than before" beyond segmentation noise
        c *= 0.7
    return c


def match_dishes(before: list, after: list) -> tuple:
    """Pair dishes by size rank (largest with largest, ...).

    Rank within a photo survives changes in camera angle, distance and rotation,
    which position-based matching does not (photos are hand-held, taken minutes apart).
    """
    order_b = sorted(range(len(before)), key=lambda i: -before[i].area_frac)
    order_a = sorted(range(len(after)), key=lambda j: -after[j].area_frac)
    n = min(len(order_b), len(order_a))
    pairs = [(before[order_b[k]], after[order_a[k]]) for k in range(n)]
    return pairs, [before[i] for i in order_b[n:]], [after[j] for j in order_a[n:]]


def warnings(before, after, unmatched_before, unmatched_after) -> list:
    out = []
    if unmatched_before or unmatched_after:
        out.append(
            f"dish count differs (before={len(before)}, after={len(after)}); unmatched dishes were ignored"
        )
    if any(d.occupancy < 0.05 for d in before):
        out.append("a dish in the before image has almost no food; its leftover is unreliable")
    return out
