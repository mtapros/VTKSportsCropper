from __future__ import annotations

from models import BoundingBox, CropRecommendation


def parse_ratio(ratio_str: str) -> float:
    left, right = ratio_str.split(":")
    return float(left) / float(right)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def shift_box_to_fit(box: BoundingBox, img_w: int, img_h: int) -> BoundingBox:
    x1, y1, x2, y2 = box.as_tuple()
    w = x2 - x1
    h = y2 - y1

    if w > img_w:
        x1, x2 = 0, img_w
    else:
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > img_w:
            x1 -= (x2 - img_w)
            x2 = img_w

    if h > img_h:
        y1, y2 = 0, img_h
    else:
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 > img_h:
            y1 -= (y2 - img_h)
            y2 = img_h

    return BoundingBox(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def build_center_crop(img_w: int, img_h: int, ratio_str: str, margin_pct: float) -> BoundingBox:
    target_ratio = parse_ratio(ratio_str)
    margin = max(0.0, margin_pct / 100.0)

    usable_w = img_w * (1.0 - margin)
    usable_h = img_h * (1.0 - margin)

    crop_w = usable_w
    crop_h = crop_w / target_ratio

    if crop_h > usable_h:
        crop_h = usable_h
        crop_w = crop_h * target_ratio

    cx = img_w / 2.0
    cy = img_h / 2.0
    x1 = cx - crop_w / 2.0
    y1 = cy - crop_h / 2.0
    x2 = cx + crop_w / 2.0
    y2 = cy + crop_h / 2.0

    return shift_box_to_fit(BoundingBox(int(x1), int(y1), int(x2), int(y2)), img_w, img_h)


def build_demo_crops(img_w: int, img_h: int, ratio_str: str, margin_pct: float) -> list[CropRecommendation]:
    crop = build_center_crop(img_w, img_h, ratio_str, margin_pct)
    return [
        CropRecommendation(name="Primary_Full", bbox=crop, color="#00FF00"),
    ]