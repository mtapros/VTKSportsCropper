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
    next_id = 1
    for bbox in raw_bboxes:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                id=next_id,
                label=phrase,
                bbox=BoundingBox(x1, y1, x2, y2),
                color="#00BFFF",
                source="florence_phrase",
            )
        )
        next_id += 1

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

    detections: list[Detection] = []
    next_id = 1
    for bbox, label in zip(raw_bboxes, raw_labels):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                id=next_id,
                label=str(label),
                bbox=BoundingBox(x1, y1, x2, y2),
                color="#00BFFF",
                source="florence_od",
            )
        )
        next_id += 1

    return detections


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


def clamp_margin_pct_to_image_edges(
    subject_box: BoundingBox,
    img_w: int,
    img_h: int,
    margin_pct: float,
) -> float:
    margin_pct = max(0.0, float(margin_pct))

    box_w = max(1.0, float(subject_box.width))
    box_h = max(1.0, float(subject_box.height))

    left_room = max(0.0, float(subject_box.x1))
    right_room = max(0.0, float(img_w - subject_box.x2))
    top_room = max(0.0, float(subject_box.y1))
    bottom_room = max(0.0, float(img_h - subject_box.y2))

    max_x_margin_pct = min(left_room, right_room) / box_w * 100.0
    max_y_margin_pct = min(top_room, bottom_room) / box_h * 100.0

    safe_margin_pct = min(margin_pct, max_x_margin_pct, max_y_margin_pct)
    return max(0.0, safe_margin_pct)


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
    effective_margin_pct = clamp_margin_pct_to_image_edges(
        subject_box=subject_box,
        img_w=img_w,
        img_h=img_h,
        margin_pct=margin_pct,
    )

    expanded = expand_box(subject_box, img_w, img_h, effective_margin_pct)

    ratio = parse_ratio(ratio_str)
    subject_cx = (subject_box.x1 + subject_box.x2) / 2.0
    subject_cy = (subject_box.y1 + subject_box.y2) / 2.0

    max_left = subject_cx
    max_right = img_w - subject_cx
    max_up = subject_cy
    max_down = img_h - subject_cy

    max_half_w = min(max_left, max_right)
    max_half_h = min(max_up, max_down)

    req_half_w = max(1.0, expanded.width / 2.0)
    req_half_h = max(1.0, expanded.height / 2.0)

    half_w = req_half_w
    half_h = half_w / ratio

    if half_h < req_half_h:
        half_h = req_half_h
        half_w = half_h * ratio

    max_half_w_by_h = max_half_h * ratio
    max_half_h_by_w = max_half_w / ratio

    if half_w > max_half_w:
        half_w = max_half_w
        half_h = half_w / ratio

    if half_h > max_half_h:
        half_h = max_half_h
        half_w = half_h * ratio

    if half_h < req_half_h:
        half_h = min(max_half_h, req_half_h)
        half_w = min(max_half_w, half_h * ratio)

    if half_w < req_half_w:
        half_w = min(max_half_w, req_half_w)
        half_h = min(max_half_h, half_w / ratio)

    x1 = int(round(subject_cx - half_w))
    y1 = int(round(subject_cy - half_h))
    x2 = int(round(subject_cx + half_w))
    y2 = int(round(subject_cy + half_h))

    centered = BoundingBox(x1, y1, x2, y2)
    return shift_box_to_fit(centered, img_w, img_h)


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def get_focus_score(image, bbox: BoundingBox) -> float:
    try:
        arr = np.array(image)
        if arr.size == 0:
            return 0.0

        h, w = arr.shape[:2]
        x1 = max(0, min(w, bbox.x1))
        y1 = max(0, min(h, bbox.y1))
        x2 = max(0, min(w, bbox.x2))
        y2 = max(0, min(h, bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = arr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def get_af_points_and_boxes(image_path: Path):
    image_path = Path(image_path)
    sidecar = image_path.with_suffix(".json")

    af_points = []
    af_boxes = []

    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))

            for pt in data.get("af_points", []):
                if isinstance(pt, dict):
                    x = int(pt.get("x", 0))
                    y = int(pt.get("y", 0))
                    af_points.append((x, y))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    af_points.append((int(pt[0]), int(pt[1])))

            for box in data.get("af_boxes", []):
                if isinstance(box, dict):
                    af_boxes.append(
                        BoundingBox(
                            int(box.get("x1", 0)),
                            int(box.get("y1", 0)),
                            int(box.get("x2", 0)),
                            int(box.get("y2", 0)),
                        )
                    )
                elif isinstance(box, (list, tuple)) and len(box) >= 4:
                    af_boxes.append(
                        BoundingBox(
                            int(box[0]),
                            int(box[1]),
                            int(box[2]),
                            int(box[3]),
                        )
                    )
        except Exception:
            pass

    if af_points or af_boxes:
        return af_points, af_boxes

    try:
        exiftool = "exiftool"
        result = subprocess.run(
            [exiftool, "-j", str(image_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return [], []

        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or not payload:
            return [], []

        data = payload[0]

        for key, value in data.items():
            key_lower = key.lower()

            if "af" in key_lower and "point" in key_lower and isinstance(value, str):
                matches = re.findall(r"(-?\d+)\s*,\s*(-?\d+)", value)
                for x, y in matches:
                    af_points.append((int(x), int(y)))

            if "af" in key_lower and "box" in key_lower and isinstance(value, str):
                matches = re.findall(r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)", value)
                for x1, y1, x2, y2 in matches:
                    af_boxes.append(BoundingBox(int(x1), int(y1), int(x2), int(y2)))

    except Exception:
        pass

    return af_points, af_boxes