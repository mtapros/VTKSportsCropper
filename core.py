from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageOps
import cv2
import json
import numpy as np
import re
import subprocess
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from models import BoundingBox, Detection


SUPPORTED_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

_FLORENCE_PROCESSOR = None
_FLORENCE_MODEL = None
_FLORENCE_DEVICE = None


class ImageRepository:
    def list_images(self, folder: Path) -> list[Path]:
        found: list[Path] = []
        for pattern in SUPPORTED_PATTERNS:
            found.extend(folder.glob(pattern))
        return sorted(set(found))

    def load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB").copy()

    def save_crop(self, image: Image.Image, bbox: BoundingBox, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped = image.crop(bbox.as_tuple())
        cropped.save(output_path, quality=95)


def get_florence_components():
    global _FLORENCE_PROCESSOR, _FLORENCE_MODEL, _FLORENCE_DEVICE

    if _FLORENCE_PROCESSOR is not None and _FLORENCE_MODEL is not None:
        return _FLORENCE_PROCESSOR, _FLORENCE_MODEL, _FLORENCE_DEVICE

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "microsoft/Florence-2-base"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

    _FLORENCE_PROCESSOR = processor
    _FLORENCE_MODEL = model
    _FLORENCE_DEVICE = device
    return _FLORENCE_PROCESSOR, _FLORENCE_MODEL, _FLORENCE_DEVICE


def run_florence_phrase_detection(image, phrase: str) -> list[Detection]:
    processor, model, device = get_florence_components()
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"

    inputs = processor(
        text=f"{task_prompt} {phrase}",
        images=image,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        num_beams=3,
    )
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed = processor.post_process_generation(
        decoded,
        task=task_prompt,
        image_size=(image.width, image.height),
    )

    raw_bboxes = []
    if isinstance(parsed, dict):
        result = parsed.get(task_prompt)
        if not isinstance(result, dict) and len(parsed) == 1:
            only_value = next(iter(parsed.values()))
            if isinstance(only_value, dict):
                result = only_value
        if isinstance(result, dict):
            raw_bboxes = result.get("bboxes", []) or []

    detections: list[Detection] = []
    for i, bbox in enumerate(raw_bboxes, start=1):
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                id=i,
                label=phrase,
                bbox=BoundingBox(x1, y1, x2, y2),
                color="#00BFFF",
                source=f"phrase:{phrase}",
            )
        )
    return detections


def run_florence_od_detection(image) -> list[Detection]:
    processor, model, device = get_florence_components()
    task_prompt = "<OD>"

    inputs = processor(
        text=task_prompt,
        images=image,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        num_beams=3,
    )
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed = processor.post_process_generation(
        decoded,
        task=task_prompt,
        image_size=(image.width, image.height),
    )

    raw_bboxes = []
    raw_labels = []

    if isinstance(parsed, dict):
        result = parsed.get(task_prompt)
        if not isinstance(result, dict) and len(parsed) == 1:
            only_value = next(iter(parsed.values()))
            if isinstance(only_value, dict):
                result = only_value
        if isinstance(result, dict):
            raw_bboxes = result.get("bboxes", []) or []
            raw_labels = result.get("labels", []) or []

    while len(raw_labels) < len(raw_bboxes):
        raw_labels.append("object")

    detections: list[Detection] = []
    for i, (bbox, label) in enumerate(zip(raw_bboxes, raw_labels), start=1):
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                id=i,
                label=str(label).strip() or "object",
                bbox=BoundingBox(x1, y1, x2, y2),
                color="#7FD1FF",
                source="od",
            )
        )
    return detections


def parse_to_float_list(val):
    if isinstance(val, (int, float)):
        return [float(val)]
    if isinstance(val, str):
        nums = re.findall(r"-?\d+\.?\d*", val)
        return [float(n) for n in nums]
    if isinstance(val, list):
        out = []
        for v in val:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out
    return []


def get_af_points_and_boxes(file_path: Path):
    try:
        cmd = [
            "exiftool",
            "-j",
            "-AF*",
            "-AFArea*",
            "-AFPointsInFocus",
            "-AFAreaXPositions",
            "-AFAreaYPositions",
            "-AFAreaWidths",
            "-AFAreaHeights",
            "-ImageWidth",
            "-ImageHeight",
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)[0] if result.stdout.strip() else {}
    except FileNotFoundError:
        return [], []
    except Exception:
        return [], []

    af_in_focus = parse_to_float_list(payload.get("AFPointsInFocus"))
    af_x = parse_to_float_list(payload.get("AFAreaXPositions"))
    af_y = parse_to_float_list(payload.get("AFAreaYPositions"))
    af_w = parse_to_float_list(payload.get("AFAreaWidths"))
    af_h = parse_to_float_list(payload.get("AFAreaHeights"))

    try:
        img_w = int(payload.get("ImageWidth") or 0)
        img_h = int(payload.get("ImageHeight") or 0)
    except Exception:
        img_w, img_h = 0, 0

    if not img_w or not img_h:
        try:
            with Image.open(file_path) as dim_img:
                img_w, img_h = dim_img.size
        except Exception:
            return [], []

    points = []
    boxes = []

    for point_idx_float in af_in_focus:
        point_idx = int(point_idx_float)
        if point_idx < len(af_x) and point_idx < len(af_y):
            x = af_x[point_idx]
            y = af_y[point_idx]
            w = af_w[point_idx] if point_idx < len(af_w) else 56.0
            h = af_h[point_idx] if point_idx < len(af_h) else 56.0
            cx = (img_w / 2.0) + x
            cy = (img_h / 2.0) + y
            box = BoundingBox(
                int(round(cx - (w / 2.0))),
                int(round(cy - (h / 2.0))),
                int(round(cx + (w / 2.0))),
                int(round(cy + (h / 2.0))),
            )
            points.append((cx, cy))
            boxes.append(box)

    return points, boxes


def get_focus_score(image, box: BoundingBox) -> float:
    rgb = np.array(image)
    cv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(cv_img.shape[1], int(box.x2))
    y2 = min(cv_img.shape[0], int(box.y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    face_region_y2 = y1 + int((y2 - y1) * 0.35)
    face_region_y2 = max(y1 + 1, min(face_region_y2, y2))

    crop = cv_img[y1:face_region_y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def parse_ratio(ratio_str: str) -> float:
    left, right = ratio_str.split(":")
    return float(left) / float(right)


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
    ratio = parse_ratio(ratio_str)
    margin_pct = max(0.0, float(margin_pct))
    usable_w = img_w * (1.0 - (margin_pct / 100.0))
    usable_h = img_h * (1.0 - (margin_pct / 100.0))

    crop_w = usable_w
    crop_h = crop_w / ratio

    if crop_h > usable_h:
        crop_h = usable_h
        crop_w = crop_h * ratio

    cx = img_w / 2.0
    cy = img_h / 2.0
    box = BoundingBox(
        int(cx - crop_w / 2.0),
        int(cy - crop_h / 2.0),
        int(cx + crop_w / 2.0),
        int(cy + crop_h / 2.0),
    )
    return shift_box_to_fit(box, img_w, img_h)


def union_boxes(boxes: list[BoundingBox]) -> BoundingBox | None:
    if not boxes:
        return None
    return BoundingBox(
        min(b.x1 for b in boxes),
        min(b.y1 for b in boxes),
        max(b.x2 for b in boxes),
        max(b.y2 for b in boxes),
    )


def expand_box(box: BoundingBox, img_w: int, img_h: int, margin_pct: float) -> BoundingBox:
    margin = max(0.0, margin_pct) / 100.0
    x_pad = box.width * margin
    y_pad = box.height * margin

    expanded = BoundingBox(
        int(round(box.x1 - x_pad)),
        int(round(box.y1 - y_pad)),
        int(round(box.x2 + x_pad)),
        int(round(box.y2 + y_pad)),
    )
    return shift_box_to_fit(expanded, img_w, img_h)


def fit_box_to_ratio(box: BoundingBox, img_w: int, img_h: int, ratio_str: str) -> BoundingBox:
    ratio = parse_ratio(ratio_str)

    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    w = max(1.0, float(box.width))
    h = max(1.0, float(box.height))

    current_ratio = w / h

    if current_ratio < ratio:
        w = h * ratio
    else:
        h = w / ratio

    fitted = BoundingBox(
        int(round(cx - w / 2.0)),
        int(round(cy - h / 2.0)),
        int(round(cx + w / 2.0)),
        int(round(cy + h / 2.0)),
    )
    return shift_box_to_fit(fitted, img_w, img_h)


def build_crop_around_subject(subject_box: BoundingBox, img_w: int, img_h: int, ratio_str: str, margin_pct: float) -> BoundingBox:
    expanded = expand_box(subject_box, img_w, img_h, margin_pct)
    return fit_box_to_ratio(expanded, img_w, img_h, ratio_str)


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    return inter_area / float(area1 + area2 - inter_area)