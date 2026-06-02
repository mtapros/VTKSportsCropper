from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageOps, ExifTags, ImageDraw, ImageFont

from core import (
    build_crop_around_subject,
    compute_iou,
    get_focus_score,
    run_florence_od_detection,
    run_florence_phrase_detection,
)
from lmstudio_client import LMStudioClient
from models import CropBox, Detection, SportProfile, BoundingBox


DANCE_CULL_SCHEMA_VERSION = "dance_v2"


class DanceCullCache:
    def __init__(self, folder: Path) -> None:
        self._path = Path(folder) / "VL_Debug" / "dance_cull_cache.json"
        self._entries: dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _entry_key(self, image_path: Path, rules_hash: str) -> str:
        try:
            mtime_ms = int(image_path.stat().st_mtime * 1000)
        except Exception:
            mtime_ms = 0
        return f"{image_path.name}|{mtime_ms}|{rules_hash}"

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("schema_version") == DANCE_CULL_SCHEMA_VERSION:
                    self._entries = data.get("entries", {})
                    return
        except Exception:
            pass
        self._entries = {}

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(
                    {"schema_version": DANCE_CULL_SCHEMA_VERSION, "entries": self._entries},
                    f,
                    indent=2,
                    default=str,
                )
            self._dirty = False
        except Exception:
            pass

    def get(self, image_path: Path, rules_hash: str) -> dict | None:
        return self._entries.get(self._entry_key(image_path, rules_hash))

    def put(self, image_path: Path, rules_hash: str, entry: dict) -> None:
        self._entries[self._entry_key(image_path, rules_hash)] = entry
        self._dirty = True
        self.save()


class AICullTool:
    tool_id = "ai_cull"
    display_name = "AI Cull Tool"
    BURST_VL_SCORE_GAP = 8.0
    MAX_VL_BURST_CANDIDATES = 6

    DANCE_PICK_COLORS = [
        ("red", "#FF4D4D"),
        ("yellow", "#FFD84D"),
        ("cyan", "#33D6FF"),
        ("lime", "#7CFF4D"),
        ("magenta", "#FF5CFF"),
        ("orange", "#FF9A3D"),
    ]

    VL_TARGET_LONG_EDGE = 1024

    def __init__(self, app):
        self.app = app
        self.panel = None

        self.prompt_vars = [tk.StringVar() for _ in range(4)]
        self.detection_mode_var = tk.StringVar(value="Phrase Only")

        self.keep_threshold_var = tk.StringVar(value="80")
        self.maybe_threshold_var = tk.StringVar(value="55")
        self.blur_penalty_threshold_var = tk.StringVar(value="12")
        self.blur_reject_threshold_var = tk.StringVar(value="6")
        self.blur_penalty_points_var = tk.StringVar(value="20")

        self.enable_burst_var = tk.BooleanVar(value=True)
        self.burst_fps_var = tk.StringVar(value="8")
        self.keep_per_burst_var = tk.StringVar(value="1")
        self.use_vl_burst_tiebreaker_var = tk.BooleanVar(value=False)
        self.prefer_face_var = tk.BooleanVar(value=True)

        self.use_dance_vl_var = tk.BooleanVar(value=True)
        self.use_dance_vl_subject_picker_var = tk.BooleanVar(value=True)
        self.save_vl_debug_images_var = tk.BooleanVar(value=True)
        self.show_dance_debug_preview_var = tk.BooleanVar(value=True)

        self.current_score = 0.0
        self.current_decision = "Reject"
        self.current_hero_id: int | None = None
        self.current_ball_id: int | None = None
        self.current_vl_rubric: dict | None = None
        self.current_vl_subject_reason: str = ""
        self.current_vl_debug_image_path: Path | None = None
        self.current_vl_mismatch: bool = False
        self.current_vl_mismatch_context: dict | None = None

        self.auto_running = False
        self.auto_cancel_requested = False
        self.auto_images: list[Path] = []
        self.auto_results: list[dict] = []
        self.auto_index = 0
        self.auto_button = None
        self.stop_button = None

        self.dance_frame = None

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")
        pad = {"padx": 10, "pady": 4}

        tk.Label(
            self.panel,
            text="AI Cull Settings",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", **pad)

        tk.Label(self.panel, text="Detection Mode", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        ttk.Combobox(
            self.panel,
            textvariable=self.detection_mode_var,
            values=["Hybrid", "Phrase Only"],
            state="readonly",
        ).pack(fill="x", **pad)

        tk.Label(
            self.panel,
            text="Prompts",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(10, 4), padx=10)

        for i, var in enumerate(self.prompt_vars, start=1):
            tk.Label(self.panel, text=f"Prompt {i}", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
            tk.Entry(self.panel, textvariable=var).pack(fill="x", **pad)

        tk.Label(self.panel, text="Keep Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.keep_threshold_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="Maybe Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.maybe_threshold_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="Blur Penalty Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.blur_penalty_threshold_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="Blur Reject Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.blur_reject_threshold_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="Blur Penalty Points", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.blur_penalty_points_var).pack(fill="x", **pad)

        tk.Label(
            self.panel,
            text="Burst Suppression",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(12, 4), padx=10)

        tk.Checkbutton(
            self.panel,
            text="Enable Burst Suppression",
            variable=self.enable_burst_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        tk.Label(self.panel, text="Burst FPS Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.burst_fps_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="Keep Per Burst", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.keep_per_burst_var).pack(fill="x", **pad)

        tk.Checkbutton(
            self.panel,
            text="Use VL Burst Tie-Breaker",
            variable=self.use_vl_burst_tiebreaker_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        tk.Checkbutton(
            self.panel,
            text="Prefer Visible Face",
            variable=self.prefer_face_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        self.dance_frame = tk.Frame(self.panel, bg="#2a2a2a")
        self.dance_frame.pack(fill="x", padx=10, pady=(12, 4))

        tk.Label(
            self.dance_frame,
            text="Vision Model Culling",
            bg="#2a2a2a",
            fg="#ffd27f",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        tk.Checkbutton(
            self.dance_frame,
            text="Use LM Studio VL Rubric",
            variable=self.use_dance_vl_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w")

        tk.Checkbutton(
            self.dance_frame,
            text="Use VL Subject Picker",
            variable=self.use_dance_vl_subject_picker_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w")

        tk.Checkbutton(
            self.dance_frame,
            text="Save VL Debug Images",
            variable=self.save_vl_debug_images_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w")

        tk.Checkbutton(
            self.dance_frame,
            text="Show 4-Panel Debug Preview",
            variable=self.show_dance_debug_preview_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w")

        tk.Button(self.panel, text="Rerun AI Cull", command=self.rerun).pack(fill="x", padx=10, pady=(10, 4))
        tk.Button(self.panel, text="Approve Keep/Maybe", command=self.approve).pack(fill="x", padx=10, pady=(0, 4))

        self.auto_button = tk.Button(self.panel, text="Auto Cull Input Folder", command=self.auto_cull_input_folder)
        self.auto_button.pack(fill="x", padx=10, pady=(0, 4))

        self.stop_button = tk.Button(
            self.panel,
            text="Stop Auto Cull",
            command=self.stop_auto_cull,
            state="disabled",
            bg="#8b1e1e",
            fg="white",
        )
        self.stop_button.pack(fill="x", padx=10, pady=(0, 4))

        self._refresh_dynamic_sections()
        return self.panel

    def _current_profile_is_dance(self) -> bool:
        profile = self.app.get_current_profile()
        sport_type = getattr(profile, "sport_type", "").strip().lower()
        if sport_type == "dance":
            return True
        return "dance" in profile.name.lower()

    def _refresh_dynamic_sections(self):
        if self.dance_frame is None:
            return
        if not self.dance_frame.winfo_ismapped():
            self.dance_frame.pack(fill="x", padx=10, pady=(12, 4))

    def apply_profile(self, profile: SportProfile):
        prompts = list(profile.prompts[:4])
        while len(prompts) < 4:
            prompts.append("")
        for i, var in enumerate(self.prompt_vars):
            var.set(prompts[i])
        self._refresh_dynamic_sections()

    def get_profile_data(self) -> SportProfile:
        base_profile = self.app.get_current_profile()
        profile_name = self.app.get_selected_profile_name() or "AI Cull"
        return SportProfile(
            name=profile_name,
            prompts=[v.get().strip() for v in self.prompt_vars],
            focus_min=0.0,
            focus_relative=0.0,
            edge_margin=0,
            margin_buffer=0.0,
            main_ratio="4:5",
            auto_rotate=False,
            join_descriptors=False,
            safe_ratios={},
            sport_type=getattr(base_profile, "sport_type", "generic"),
            vl_rubric_name=getattr(base_profile, "vl_rubric_name", "generic"),
            prefer_full_body=getattr(base_profile, "prefer_full_body", True),
            penalize_cropped_feet=getattr(base_profile, "penalize_cropped_feet", True),
            favor_symmetry=getattr(base_profile, "favor_symmetry", False),
            favor_peak_action=getattr(base_profile, "favor_peak_action", True),
            prefer_clean_pose=getattr(base_profile, "prefer_clean_pose", True),
            prefer_single_subject=getattr(base_profile, "prefer_single_subject", False),
        )

    def get_runtime_config(self) -> dict:
        profile = self.get_profile_data()
        prompts = [p.strip() for p in profile.prompts if p.strip()]
        return {
            "prompts": prompts,
            "detection_mode": self.detection_mode_var.get().strip() or "Phrase Only",
            "keep_threshold": float(self.keep_threshold_var.get().strip() or "80"),
            "maybe_threshold": float(self.maybe_threshold_var.get().strip() or "55"),
            "blur_penalty_threshold": float(self.blur_penalty_threshold_var.get().strip() or "12"),
            "blur_reject_threshold": float(self.blur_reject_threshold_var.get().strip() or "6"),
            "blur_penalty_points": float(self.blur_penalty_points_var.get().strip() or "20"),
            "enable_burst": bool(self.enable_burst_var.get()),
            "burst_fps": float(self.burst_fps_var.get().strip() or "8"),
            "keep_per_burst": int(self.keep_per_burst_var.get().strip() or "1"),
            "use_vl_burst_tiebreaker": bool(self.use_vl_burst_tiebreaker_var.get()),
            "prefer_face": bool(self.prefer_face_var.get()),
            "sport_type": getattr(profile, "sport_type", "generic"),
            "use_dance_vl": bool(self.use_dance_vl_var.get()),
            "use_dance_vl_subject_picker": bool(self.use_dance_vl_subject_picker_var.get()),
            "save_vl_debug_images": bool(self.save_vl_debug_images_var.get()),
            "show_dance_debug_preview": bool(self.show_dance_debug_preview_var.get()),
        }

    def _is_ball_label(self, label: str) -> bool:
        return "ball" in label.lower()

    def _is_face_label(self, label: str) -> bool:
        return "face" in label.lower()

    def _is_person_label(self, label: str) -> bool:
        label = label.lower()
        person_terms = ["person", "player", "athlete", "goalkeeper", "man", "woman", "boy", "girl", "dancer", "face"]
        return any(term in label for term in person_terms)

    def _prompts_include_ball(self, prompts: list[str]) -> bool:
        for prompt in prompts:
            if "ball" in prompt.strip().lower():
                return True
        return False

    def _is_full_person_label(self, label: str) -> bool:
        if self._is_face_label(label):
            return False
        label = label.lower()
        person_terms = ["person", "player", "athlete", "goalkeeper", "man", "woman", "boy", "girl", "dancer"]
        return any(term in label for term in person_terms)

    def _filter_nested_detections(self, detections: list[Detection]) -> list[Detection]:
        filtered: list[Detection] = []
        for det in detections:
            det_area = det.bbox.width * det.bbox.height
            if det_area <= 0:
                filtered.append(det)
                continue
            is_nested = False
            for other in detections:
                if other.id == det.id:
                    continue
                other_area = other.bbox.width * other.bbox.height
                if other_area <= det_area:
                    continue
                ix1 = max(det.bbox.x1, other.bbox.x1)
                iy1 = max(det.bbox.y1, other.bbox.y1)
                ix2 = min(det.bbox.x2, other.bbox.x2)
                iy2 = min(det.bbox.y2, other.bbox.y2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter_area / det_area >= 0.70:
                    is_nested = True
                    break
            if not is_nested:
                filtered.append(det)
        return filtered

    def _vl_candidate_people(self, all_people: list[Detection]) -> list[Detection]:
        full_person = [d for d in all_people if self._is_full_person_label(d.label)]
        return self._filter_nested_detections(full_person)

    def _dance_rules_hash(self) -> str:
        try:
            tool = self.app.tools_by_id.get("lmstudio")
            model = tool.model_var.get().strip() if tool else ""
        except Exception:
            model = ""
        raw = f"{DANCE_CULL_SCHEMA_VERSION}|{model}|vl1024|bigbadges|fullbatchflow"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _get_dance_cull_cache(self, image_path: Path) -> DanceCullCache:
        folder = self.app.state.input_folder or image_path.parent
        return DanceCullCache(Path(folder))

    @staticmethod
    def _det_to_dict(det: Detection) -> dict:
        return {
            "id": det.id,
            "label": det.label,
            "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
            "color": det.color,
            "source": det.source,
        }

    @staticmethod
    def _det_from_dict(d: dict) -> Detection:
        b = d["bbox"]
        return Detection(
            id=d["id"],
            label=d["label"],
            bbox=BoundingBox(b[0], b[1], b[2], b[3]),
            color=d.get("color", "#00BFFF"),
            source=d.get("source", "cache"),
        )

    def _compute_crop_proposal(
        self,
        subject: Detection | None,
        img_w: int,
        img_h: int,
        ratio_str: str = "4:5",
        margin_pct: float = 12.0,
    ) -> tuple[BoundingBox | None, float]:
        if subject is None or img_w <= 0 or img_h <= 0:
            return None, 0.0

        crop = build_crop_around_subject(
            subject_box=subject.bbox,
            img_w=img_w,
            img_h=img_h,
            ratio_str=ratio_str,
            margin_pct=margin_pct,
        )

        subject_cx = (subject.bbox.x1 + subject.bbox.x2) / 2.0
        subject_cy = (subject.bbox.y1 + subject.bbox.y2) / 2.0
        dx_norm = abs(subject_cx - img_w / 2.0) / img_w
        dy_norm = abs(subject_cy - img_h / 2.0) / img_h
        max_offset = max(dx_norm, dy_norm)

        if max_offset < 0.10:
            penalty = 0.0
        elif max_offset < 0.25:
            penalty = -5.0 * ((max_offset - 0.10) / 0.15)
        else:
            penalty = -5.0 - 10.0 * min(1.0, (max_offset - 0.25) / 0.25)

        return crop, round(penalty, 2)

    def _log_manual_review_entry(self, image_path: Path, context: dict) -> None:
        try:
            debug_dir = self._dance_debug_dir(image_path)
            log_path = debug_dir / "manual_review.jsonl"
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image_path": str(image_path),
                **context,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def _dedupe_detections(self, detections: list[Detection]) -> list[Detection]:
        detections = sorted(detections, key=lambda d: d.bbox.width * d.bbox.height, reverse=True)
        deduped: list[Detection] = []
        next_id = 1

        for det in detections:
            duplicate = False
            for existing in deduped:
                if compute_iou(det.bbox, existing.bbox) > 0.65:
                    duplicate = True
                    break
            if not duplicate:
                det.id = next_id
                next_id += 1
                deduped.append(det)
        return deduped

    def _distance_between_boxes(self, a: BoundingBox, b: BoundingBox) -> float:
        acx = (a.x1 + a.x2) / 2.0
        acy = (a.y1 + a.y2) / 2.0
        bcx = (b.x1 + b.x2) / 2.0
        bcy = (b.y1 + b.y2) / 2.0
        return math.hypot(acx - bcx, acy - bcy)

    def _contains_box_center(self, outer: BoundingBox, inner: BoundingBox) -> bool:
        cx = (inner.x1 + inner.x2) / 2.0
        cy = (inner.y1 + inner.y2) / 2.0
        return outer.x1 <= cx <= outer.x2 and outer.y1 <= cy <= outer.y2

    def _overlaps_any_af(self, box: BoundingBox) -> bool:
        for af_box in self.app.current_af_boxes:
            if compute_iou(box, af_box) > 0 or self._contains_box_center(af_box, box):
                return True
        return False

    def _distance_to_nearest_af(self, box: BoundingBox) -> float:
        if not self.app.current_af_boxes:
            return float("inf")
        return min(self._distance_between_boxes(box, af_box) for af_box in self.app.current_af_boxes)

    def _pick_hero_person(self, person_detections: list[Detection]) -> tuple[Detection | None, bool, float]:
        if not person_detections:
            return None, False, 0.0

        af_matches = [d for d in person_detections if self._overlaps_any_af(d.bbox)]
        if af_matches:
            scored = []
            for det in af_matches:
                focus = get_focus_score(self.app.current_image, det.bbox)
                scored.append((focus, det))
                self.app.log(f'Cull AF candidate "{det.label}" focus={focus:.1f}')
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1], True, scored[0][0]

        ranked = []
        for det in person_detections:
            dist = self._distance_to_nearest_af(det.bbox)
            focus = get_focus_score(self.app.current_image, det.bbox)
            ranked.append((dist, -focus, det, focus))
            self.app.log(f'Cull near-AF candidate "{det.label}" dist={dist:.1f} focus={focus:.1f}')
        ranked.sort(key=lambda item: (item[0], item[1]))
        best = ranked[0]
        return best[2], False, best[3]

    def _pick_support_ball(self, hero: Detection | None, balls: list[Detection]) -> tuple[Detection | None, bool]:
        if hero is None or not balls:
            return None, False

        nearest_ball = min(balls, key=lambda d: self._distance_between_boxes(hero.bbox, d.bbox))
        hero_diag = math.hypot(hero.bbox.width, hero.bbox.height)
        ball_dist = self._distance_between_boxes(hero.bbox, nearest_ball.bbox)

        is_support = ball_dist <= max(hero_diag * 1.5, 180)
        return nearest_ball, is_support

    def _pick_face_for_hero(self, hero: Detection | None, faces: list[Detection]) -> tuple[Detection | None, bool, float]:
        if hero is None or not faces:
            return None, False, 0.0
        candidates = []
        for face in faces:
            if compute_iou(hero.bbox, face.bbox) > 0 or self._contains_box_center(hero.bbox, face.bbox):
                focus = get_focus_score(self.app.current_image, face.bbox)
                candidates.append((focus, face))
        if not candidates:
            return None, False, 0.0
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_focus, best_face = candidates[0]
        return best_face, True, best_focus

    def _unique_preserve_order(self, items: list[str]) -> list[str]:
        seen = set()
        out = []
        for item in items:
            key = item.strip().lower()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(item.strip())
        return out

    def _cache_key(self, image_path, prompts: list[str], mode: str) -> tuple:
        return ("ai_cull", str(image_path), tuple(prompts), mode)

    def _get_hybrid_detections_for_current_image(self, prompts: list[str]) -> list[Detection]:
        image_path = self.app.state.current_image_path
        if image_path is None or self.app.current_image is None:
            return []

        cache_key = self._cache_key(image_path, prompts, "hybrid_v1")
        if cache_key in self.app.ai_detection_cache:
            self.app.log("AI Cull: using cached hybrid Florence detections.")
            return self.app.ai_detection_cache[cache_key]

        self.app.log(f"AI Cull hybrid prompts: {prompts}")
        self.app.log("AI Cull: running Florence OD...")
        od_detections = run_florence_od_detection(self.app.current_image)
        self.app.log(f"AI Cull: OD returned {len(od_detections)} detections.")

        phrase_detections: list[Detection] = []
        for phrase in prompts:
            self.app.log(f'AI Cull phrase prompt: "{phrase}"')
            phrase_detections.extend(run_florence_phrase_detection(self.app.current_image, phrase))

        self.app.log(f"AI Cull: phrase grounding returned {len(phrase_detections)} raw detections.")

        merged = self._dedupe_detections(od_detections + phrase_detections)
        self.app.ai_detection_cache[cache_key] = merged
        self.app.log(f"AI Cull: merged to {len(merged)} unique detections.")
        return merged

    def _get_phrase_only_detections_for_current_image(self, prompts: list[str]) -> list[Detection]:
        image_path = self.app.state.current_image_path
        if image_path is None or self.app.current_image is None:
            return []

        phrase_only_prompts = self._unique_preserve_order(prompts)
        cache_key = self._cache_key(image_path, phrase_only_prompts, "phrase_only_v4_blur_gate")
        if cache_key in self.app.ai_detection_cache:
            self.app.log("AI Cull: using cached phrase-only Florence detections.")
            return self.app.ai_detection_cache[cache_key]

        self.app.log(f"AI Cull phrase-only prompts: {phrase_only_prompts}")
        self.app.log("AI Cull: running phrase-only Florence detections...")
        phrase_detections: list[Detection] = []
        for phrase in phrase_only_prompts:
            self.app.log(f'AI Cull phrase prompt: "{phrase}"')
            phrase_detections.extend(run_florence_phrase_detection(self.app.current_image, phrase))

        self.app.log(f"AI Cull: phrase-only returned {len(phrase_detections)} raw detections.")

        merged = self._dedupe_detections(phrase_detections)
        self.app.ai_detection_cache[cache_key] = merged
        self.app.log(f"AI Cull: phrase-only merged to {len(merged)} unique detections.")
        return merged

    def _normalize_focus_score(self, focus: float) -> float:
        if focus <= 0:
            return 0.0
        if focus >= 120:
            return 30.0
        return (focus / 120.0) * 30.0

    def _subject_size_score(self, hero: Detection | None) -> float:
        if hero is None or self.app.current_image is None:
            return 0.0
        img_area = self.app.current_image.width * self.app.current_image.height
        hero_area = hero.bbox.width * hero.bbox.height
        ratio = hero_area / float(img_area) if img_area else 0.0

        if ratio >= 0.12:
            return 10.0
        return max(0.0, ratio / 0.12) * 10.0

    def _compute_keeper_score(
        self,
        hero: Detection | None,
        has_af_match: bool,
        focus_score: float,
        has_support_ball: bool,
        use_ball_scoring: bool,
        has_face: bool,
        face_focus: float,
        prefer_face: bool,
        crop_center_penalty: float = 0.0,
    ) -> tuple[float, dict]:
        breakdown = {
            "person": 0.0,
            "af": 0.0,
            "focus": 0.0,
            "size": 0.0,
            "blur_penalty": 0.0,
        }

        if use_ball_scoring:
            breakdown["ball"] = 0.0

        if prefer_face:
            breakdown["face"] = 0.0
            breakdown["face_focus"] = 0.0

        if hero is not None:
            breakdown["person"] = 25.0

        if has_af_match:
            breakdown["af"] = 25.0

        breakdown["focus"] = self._normalize_focus_score(focus_score)
        breakdown["size"] = self._subject_size_score(hero)

        if use_ball_scoring and has_support_ball:
            breakdown["ball"] = 10.0

        if prefer_face and has_face:
            breakdown["face"] = 8.0
            breakdown["face_focus"] = min(7.0, self._normalize_focus_score(face_focus) * 0.25)

        blur_penalty_threshold = float(self.blur_penalty_threshold_var.get().strip() or "12")
        blur_penalty_points = float(self.blur_penalty_points_var.get().strip() or "20")

        if hero is not None and focus_score < blur_penalty_threshold:
            breakdown["blur_penalty"] = -blur_penalty_points

        if crop_center_penalty < 0.0:
            breakdown["crop_center"] = crop_center_penalty

        total = sum(breakdown.values())
        return total, breakdown

    def _decision_from_score(self, score: float, hero_focus_score: float, hero_exists: bool) -> tuple[str, str | None]:
        if not hero_exists:
            return "Reject", "no_hero"

        blur_reject_threshold = float(self.blur_reject_threshold_var.get().strip() or "6")
        keep_threshold = float(self.keep_threshold_var.get().strip() or "80")
        maybe_threshold = float(self.maybe_threshold_var.get().strip() or "55")

        if hero_focus_score < blur_reject_threshold:
            return "Reject", "blur_reject"

        if score >= keep_threshold:
            return "Keep", None
        if score >= maybe_threshold:
            return "Maybe", None
        return "Reject", None

    def _decision_rank(self, decision: str) -> int:
        return {"Keep": 2, "Maybe": 1, "Reject": 0}.get(decision, 0)

    def _build_bursts(self, ordered_paths: list[Path], burst_fps: float) -> list[list[Path]]:
        if not ordered_paths:
            return []

        fps = max(0.1, float(burst_fps or 0.0))
        max_gap_seconds = max(0.05, (1.0 / fps) * 1.5)

        bursts: list[list[Path]] = []
        current_burst: list[Path] = []
        previous_mtime: float | None = None

        for path in ordered_paths:
            image_path = Path(path)
            try:
                current_mtime = float(image_path.stat().st_mtime)
            except Exception:
                current_mtime = None

            if not current_burst:
                current_burst = [image_path]
            elif (
                previous_mtime is not None
                and current_mtime is not None
                and (current_mtime - previous_mtime) <= max_gap_seconds
            ):
                current_burst.append(image_path)
            else:
                bursts.append(current_burst)
                current_burst = [image_path]

            previous_mtime = current_mtime

        if current_burst:
            bursts.append(current_burst)

        return bursts

    def _rank_burst_candidates(self, burst_results: list[dict]) -> list[dict]:
        return sorted(
            burst_results,
            key=lambda item: (
                self._decision_rank(str(item.get("decision", "Reject"))),
                float(item.get("score", 0.0)),
                float(item.get("hero_focus", 0.0)),
                bool(item.get("has_face", False)),
                float(item.get("face_focus", 0.0)),
            ),
            reverse=True,
        )

    def _get_vl_burst_candidates(self, ranked: list[dict], keep_per_burst: int) -> list[dict]:
        if len(ranked) <= 1:
            return ranked

        top_score = float(ranked[0].get("score", 0.0))
        top_rank = self._decision_rank(str(ranked[0].get("decision", "Reject")))
        minimum_candidates = max(keep_per_burst + 1, 2)
        candidates: list[dict] = []

        for item in ranked:
            decision_rank = self._decision_rank(str(item.get("decision", "Reject")))
            score_gap = top_score - float(item.get("score", 0.0))

            if len(candidates) < minimum_candidates:
                candidates.append(item)
            elif decision_rank >= 1 and score_gap <= self.BURST_VL_SCORE_GAP:
                candidates.append(item)
            elif decision_rank == top_rank and score_gap <= self.BURST_VL_SCORE_GAP:
                candidates.append(item)
            else:
                break

            if len(candidates) >= self.MAX_VL_BURST_CANDIDATES:
                break

        return candidates

    def _should_use_vl_burst_selector(self, ranked: list[dict], keep_per_burst: int, config: dict) -> bool:
        if not bool(config.get("use_vl_burst_tiebreaker", False)):
            return False
        if len(ranked) <= keep_per_burst:
            return False

        candidates = self._get_vl_burst_candidates(ranked, keep_per_burst)
        if len(candidates) <= keep_per_burst:
            return False

        keep_or_maybe = [item for item in candidates if self._decision_rank(str(item.get("decision", "Reject"))) >= 1]
        if len(keep_or_maybe) > keep_per_burst:
            return True

        top_score = float(candidates[0].get("score", 0.0))
        next_score = float(candidates[min(len(candidates) - 1, keep_per_burst)].get("score", 0.0))
        return (top_score - next_score) <= self.BURST_VL_SCORE_GAP

    def _select_burst_winners_with_vl(self, ranked: list[dict], keep_per_burst: int) -> tuple[list[dict], dict]:
        base_url, model, timeout, temperature, max_tokens = self._dance_lmstudio_settings()
        client = LMStudioClient(base_url=base_url, timeout=timeout)
        profile = self.get_profile_data()
        rubric_name = getattr(profile, "vl_rubric_name", "generic")
        if str(getattr(profile, "sport_type", "")).strip().lower() == "dance" and str(rubric_name).strip().lower() == "generic":
            rubric_name = "dance"
        candidates = self._get_vl_burst_candidates(ranked, keep_per_burst)
        frame_map: dict[str, dict] = {}
        frames: list[dict] = []

        for index, item in enumerate(candidates, start=1):
            frame_id = f"frame_{index}"
            frame_map[frame_id] = item
            image_path = Path(item["path"])
            frames.append(
                {
                    "frame_id": frame_id,
                    "image_path": image_path,
                    "filename": image_path.name,
                    "heuristic_score": float(item.get("score", 0.0)),
                    "decision": str(item.get("decision", "Reject")),
                    "focus_score": float(item.get("hero_focus", 0.0)),
                    "face_visible": bool(item.get("has_face", False)),
                }
            )

        selection = client.select_burst_best_frame(
            model=model,
            frames=frames,
            rubric_name=rubric_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        best_frame = str(selection.get("best_frame", "")).strip()
        if best_frame not in frame_map:
            raise ValueError(f"VL burst selector returned invalid best_frame: {best_frame!r}")

        ordered: list[dict] = [frame_map[best_frame]]
        seen_paths = {Path(frame_map[best_frame]["path"])}

        for key in selection.get("alternates", []):
            frame_id = str(key).strip()
            candidate = frame_map.get(frame_id)
            if not candidate:
                continue
            candidate_path = Path(candidate["path"])
            if candidate_path in seen_paths:
                continue
            ordered.append(candidate)
            seen_paths.add(candidate_path)

        for item in ranked:
            item_path = Path(item["path"])
            if item_path in seen_paths:
                continue
            ordered.append(item)
            seen_paths.add(item_path)

        return ordered, selection

    def _dance_lmstudio_settings(self) -> tuple[str, str, float, float, int]:
        tool = self.app.tools_by_id.get("lmstudio")
        if tool is None:
            raise RuntimeError("LM Studio tool is not loaded.")

        base_url = tool.base_url_var.get().strip()
        model = tool.model_var.get().strip()
        if not base_url:
            raise RuntimeError("LM Studio base URL is empty.")
        if not model:
            raise RuntimeError("No LM Studio model selected.")

        try:
            timeout = float(tool.timeout_var.get().strip() or "60")
        except Exception:
            timeout = 60.0

        try:
            temperature = float(tool.temperature_var.get().strip() or "0.1")
        except Exception:
            temperature = 0.1

        try:
            max_tokens = int(tool.max_tokens_var.get().strip() or "700")
        except Exception:
            max_tokens = 700

        return base_url, model, timeout, temperature, max_tokens

    def _score_dance_vl_rubric(self, rubric: dict) -> tuple[float, str]:
        score = 0.0

        score += {"strong": 20, "acceptable": 12, "soft": -5, "blurry": -25}.get(str(rubric.get("sharpness", "")).strip().lower(), 0)
        score += {"strong": 15, "good": 10, "partial": 0, "weak": -15}.get(str(rubric.get("subject_visibility", "")).strip().lower(), 0)
        score += {"clear": 10, "partial": 3, "not_visible": -6}.get(str(rubric.get("face_visibility", "")).strip().lower(), 0)
        score += {"yes": 8, "partial": 2, "no": -20, "unknown": 0}.get(str(rubric.get("face_facing_camera", "")).strip().lower(), 0)
        score += {"full": 12, "mostly_full": 6, "partial": -10}.get(str(rubric.get("full_body_visibility", "")).strip().lower(), 0)
        score += {"fully_visible": 10, "partially_cropped": -4, "cropped_out": -15}.get(str(rubric.get("feet_visibility", "")).strip().lower(), 0)
        score += {"fully_visible": 6, "partially_cropped": -2, "cropped_out": -8}.get(str(rubric.get("hands_visibility", "")).strip().lower(), 0)
        score += {"strong": 15, "good": 8, "awkward": -10, "unclear": -12}.get(str(rubric.get("pose_quality", "")).strip().lower(), 0)
        score += {"strong": 15, "good": 8, "average": 0, "weak": -10}.get(str(rubric.get("moment_quality", "")).strip().lower(), 0)
        score += {"strong": 10, "good": 6, "average": 0, "weak": -8}.get(str(rubric.get("composition_quality", "")).strip().lower(), 0)
        score += {"low": 4, "moderate": 0, "high": -8}.get(str(rubric.get("background_distraction", "")).strip().lower(), 0)
        score += {"good": 8, "somewhat_clear": 3, "poor": -8}.get(str(rubric.get("subject_separation", "")).strip().lower(), 0)

        sharpness = str(rubric.get("sharpness", "")).strip().lower()
        subject_visibility = str(rubric.get("subject_visibility", "")).strip().lower()
        pose_quality = str(rubric.get("pose_quality", "")).strip().lower()
        moment_quality = str(rubric.get("moment_quality", "")).strip().lower()
        full_body_visibility = str(rubric.get("full_body_visibility", "")).strip().lower()
        feet_visibility = str(rubric.get("feet_visibility", "")).strip().lower()
        face_facing_camera = str(rubric.get("face_facing_camera", "")).strip().lower()
        overall_keeper = str(
            rubric.get("overall_keeper", rubric.get("overall_dance_keeper", ""))
        ).strip().lower()

        if sharpness == "blurry" or subject_visibility == "weak":
            return score, "Reject"
        if face_facing_camera == "no":
            return score, "Reject"
        if pose_quality == "unclear" and moment_quality == "weak":
            return score, "Reject"
        if feet_visibility == "cropped_out" or full_body_visibility == "partial":
            if score >= 55:
                return score, "Maybe"
        if overall_keeper in {"keep", "maybe", "reject"}:
            return score, overall_keeper.capitalize()

        if score >= 55:
            return score, "Keep"
        if score >= 30:
            return score, "Maybe"
        return score, "Reject"

    def _load_rgb_image(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")

    def _scale_image_to_long_edge(self, image: Image.Image, long_edge: int) -> Image.Image:
        w, h = image.size
        longest = max(w, h)
        if longest <= long_edge:
            return image
        scale = long_edge / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.LANCZOS)

    def _hex_to_rgba(self, hex_color: str, alpha: int) -> tuple[int, int, int, int]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return (255, 255, 0, alpha)
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)

    def _dance_debug_dir(self, image_path: Path) -> Path:
        base = self.app.state.input_folder or image_path.parent
        out = Path(base) / "VL_Debug"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _font(self, size: int = 18):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            try:
                return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
            except Exception:
                return ImageFont.load_default()

    def _text_bbox(self, draw: ImageDraw.ImageDraw, text: str, font):
        try:
            return draw.textbbox((0, 0), text, font=font)
        except Exception:
            width = max(20, int(len(text) * 9))
            height = 18
            return (0, 0, width, height)

    def _draw_badge(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        text: str,
        fill=(0, 0, 0, 220),
        outline=(255, 255, 255, 255),
        text_fill=(255, 255, 255, 255),
        font=None,
        pad_x: int = 8,
        pad_y: int = 5,
    ):
        font = font or self._font(18)
        left, top, right, bottom = self._text_bbox(draw, text, font)
        text_w = right - left
        text_h = bottom - top
        box = [x, y, x + text_w + pad_x * 2, y + text_h + pad_y * 2]
        draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=3)
        draw.text((x + pad_x, y + pad_y), text, fill=text_fill, font=font)
        return box

    def _clamp_label_position(self, img_w: int, img_h: int, x: int, y: int, w: int, h: int) -> tuple[int, int]:
        x = max(4, min(x, img_w - w - 4))
        y = max(4, min(y, img_h - h - 4))
        return x, y

    def _scaled_detection(self, det: Detection, sx: float, sy: float) -> Detection:
        return Detection(
            id=det.id,
            label=det.label,
            bbox=BoundingBox(
                int(round(det.bbox.x1 * sx)),
                int(round(det.bbox.y1 * sy)),
                int(round(det.bbox.x2 * sx)),
                int(round(det.bbox.y2 * sy)),
            ),
            color=det.color,
            source=det.source,
        )

    def _prepare_scaled_debug_image(self, image_path: Path, detections: list[Detection]) -> tuple[Image.Image, list[Detection]]:
        img = self._load_rgb_image(image_path)
        orig_w, orig_h = img.size
        scaled_img = self._scale_image_to_long_edge(img, self.VL_TARGET_LONG_EDGE)
        new_w, new_h = scaled_img.size

        sx = new_w / float(orig_w) if orig_w else 1.0
        sy = new_h / float(orig_h) if orig_h else 1.0
        scaled_dets = [self._scaled_detection(d, sx, sy) for d in detections]
        return scaled_img, scaled_dets

    def _draw_florence_candidates_preview(self, image_path: Path, detections: list[Detection]) -> Image.Image:
        img, detections = self._prepare_scaled_debug_image(image_path, detections)
        draw = ImageDraw.Draw(img, "RGBA")

        base = max(img.size)
        font_id = self._font(max(52, int(base * 0.055)))
        font_label = self._font(max(34, int(base * 0.032)))
        line_width = max(8, int(base * 0.010))

        for det in detections:
            bbox = det.bbox
            color = "#33D6FF"
            color_rgba = self._hex_to_rgba(color, 255)
            fill_rgba = self._hex_to_rgba(color, 55)

            draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline=color_rgba, width=line_width)
            draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], fill=fill_rgba)

            id_text = f"ID {det.id}"
            id_box = self._text_bbox(draw, id_text, font_id)
            id_w = (id_box[2] - id_box[0]) + 28
            id_h = (id_box[3] - id_box[1]) + 18
            cx = (bbox.x1 + bbox.x2) // 2
            cy = (bbox.y1 + bbox.y2) // 2
            id_x, id_y = self._clamp_label_position(
                img.width,
                img.height,
                cx - (id_w // 2),
                cy - id_h - 10,
                id_w,
                id_h,
            )
            self._draw_badge(
                draw,
                id_x,
                id_y,
                id_text,
                fill=(0, 0, 0, 235),
                outline=color_rgba,
                text_fill=(255, 255, 255, 255),
                font=font_id,
                pad_x=14,
                pad_y=8,
            )

            label_text = str(det.label).strip() or "candidate"
            label_box = self._text_bbox(draw, label_text, font_label)
            label_w = (label_box[2] - label_box[0]) + 24
            label_h = (label_box[3] - label_box[1]) + 14
            label_x, label_y = self._clamp_label_position(
                img.width,
                img.height,
                cx - (label_w // 2),
                cy + 8,
                label_w,
                label_h,
            )
            self._draw_badge(
                draw,
                label_x,
                label_y,
                label_text,
                fill=(0, 0, 0, 220),
                outline=color_rgba,
                text_fill=color_rgba,
                font=font_label,
                pad_x=12,
                pad_y=7,
            )

        return img

    def _draw_final_subject_preview(self, image_path: Path, chosen: Detection | None) -> Image.Image:
        scaled = [chosen] if chosen is not None else []
        img, scaled_chosen = self._prepare_scaled_debug_image(image_path, scaled)
        draw = ImageDraw.Draw(img, "RGBA")
        base = max(img.size)

        if chosen is None:
            font = self._font(max(42, int(base * 0.045)))
            text = "NO FINAL PICK"
            left, top, right, bottom = self._text_bbox(draw, text, font)
            text_w = right - left
            text_h = bottom - top
            x = max(12, (img.width - text_w - 36) // 2)
            y = 12
            self._draw_badge(
                draw,
                x,
                y,
                text,
                fill=(0, 0, 0, 225),
                outline=(255, 80, 80, 255),
                text_fill=(255, 220, 220, 255),
                font=font,
                pad_x=18,
                pad_y=10,
            )
            return img

        chosen = scaled_chosen[0]
        bbox = chosen.bbox
        color = "#00FF66"
        color_rgba = self._hex_to_rgba(color, 255)
        fill_rgba = self._hex_to_rgba(color, 40)

        thick = max(8, int(base * 0.010))
        draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], fill=fill_rgba)
        draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline=color_rgba, width=thick)

        font = self._font(max(56, int(base * 0.050)))
        text = f"FINAL PICK • ID {chosen.id}"
        box = self._text_bbox(draw, text, font)
        box_w = (box[2] - box[0]) + 30
        box_h = (box[3] - box[1]) + 20
        cx = (bbox.x1 + bbox.x2) // 2
        id_x, id_y = self._clamp_label_position(
            img.width,
            img.height,
            cx - (box_w // 2),
            bbox.y1 + 12,
            box_w,
            box_h,
        )
        self._draw_badge(
            draw,
            id_x,
            id_y,
            text,
            fill=(0, 0, 0, 235),
            outline=color_rgba,
            text_fill=(255, 255, 255, 255),
            font=font,
            pad_x=15,
            pad_y=10,
        )

        return img

    def _draw_shaded_candidate_image(self, image_path: Path, detections: list[Detection]) -> tuple[Path, dict[int, str], dict[str, int]]:
        base_rgb, detections = self._prepare_scaled_debug_image(image_path, detections)
        base = base_rgb.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        longest = max(base.size)
        font_main = self._font(max(64, int(longest * 0.070)))
        font_sub = self._font(max(34, int(longest * 0.035)))

        det_to_color: dict[int, str] = {}
        color_to_det: dict[str, int] = {}

        line_width = max(10, int(longest * 0.012))

        for idx, det in enumerate(detections):
            color_name, color_hex = self.DANCE_PICK_COLORS[idx % len(self.DANCE_PICK_COLORS)]
            det_to_color[det.id] = color_name
            color_to_det[color_name] = det.id

            bbox = det.bbox
            fill_rgba = self._hex_to_rgba(color_hex, 92)
            outline_rgba = self._hex_to_rgba(color_hex, 255)

            draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], fill=fill_rgba)
            draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline=outline_rgba, width=line_width)

            main_text = f"ID {det.id}"
            sub_text = color_name.upper()

            main_box = self._text_bbox(draw, main_text, font_main)
            main_w = (main_box[2] - main_box[0]) + 34
            main_h = (main_box[3] - main_box[1]) + 24

            sub_box = self._text_bbox(draw, sub_text, font_sub)
            sub_w = (sub_box[2] - sub_box[0]) + 26
            sub_h = (sub_box[3] - sub_box[1]) + 18

            cx = (bbox.x1 + bbox.x2) // 2
            cy = (bbox.y1 + bbox.y2) // 2

            main_x, main_y = self._clamp_label_position(
                base.width,
                base.height,
                cx - (main_w // 2),
                cy - main_h - 8,
                main_w,
                main_h,
            )
            self._draw_badge(
                draw,
                main_x,
                main_y,
                main_text,
                fill=(0, 0, 0, 240),
                outline=outline_rgba,
                text_fill=(255, 255, 255, 255),
                font=font_main,
                pad_x=17,
                pad_y=12,
            )

            sub_x, sub_y = self._clamp_label_position(
                base.width,
                base.height,
                cx - (sub_w // 2),
                cy + 10,
                sub_w,
                sub_h,
            )
            self._draw_badge(
                draw,
                sub_x,
                sub_y,
                sub_text,
                fill=(0, 0, 0, 228),
                outline=outline_rgba,
                text_fill=outline_rgba,
                font=font_sub,
                pad_x=13,
                pad_y=9,
            )

        composed = Image.alpha_composite(base, overlay).convert("RGB")

        debug_dir = self._dance_debug_dir(image_path)
        debug_path = debug_dir / f"{image_path.stem}_dance_vl_candidates.jpg"
        composed.save(debug_path, format="JPEG", quality=92)

        return debug_path, det_to_color, color_to_det

    def _extract_json_object(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty response")

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("No valid JSON object found")

    def _map_position_to_detection(self, detections: list[Detection], position: str) -> Detection | None:
        if not detections:
            return None

        ordered = sorted(detections, key=lambda d: ((d.bbox.x1 + d.bbox.x2) / 2.0))

        position = position.strip().lower()
        if position == "left":
            return ordered[0]
        if position == "right":
            return ordered[-1]
        if position == "center":
            img_cx = self.app.current_image.width / 2.0 if self.app.current_image is not None else 0.0
            return min(ordered, key=lambda d: abs(((d.bbox.x1 + d.bbox.x2) / 2.0) - img_cx))
        return None

    def _select_main_dance_detection_from_candidates(self, image_path: Path, detections: list[Detection]) -> tuple[Detection | None, str]:
        if not detections:
            return None, "no detections"

        base_url, model, timeout, temperature, max_tokens = self._dance_lmstudio_settings()
        client = LMStudioClient(base_url=base_url, timeout=timeout)

        debug_path, det_to_color, color_to_det = self._draw_shaded_candidate_image(image_path, detections)
        self.current_vl_debug_image_path = debug_path

        valid_ids = {d.id for d in detections}
        valid_colors = set(color_to_det.keys())
        candidate_lines = ", ".join(
            f"ID {d.id} ({det_to_color[d.id]} box)"
            for d in detections
            if d.id in det_to_color
        )

        system_prompt = (
            "You are a dance recital photo subject selection assistant.\n\n"
            "The image shows shaded bounding box overlays on candidate dancer regions. "
            "Each box has a LARGE centered numeric ID badge (e.g. 'ID 1', 'ID 2') AND a color badge "
            "(e.g. 'RED', 'YELLOW').\n\n"
            "CRITICAL DEFINITIONS:\n"
            "- 'main_subject_detection_id' is the PRIMARY identifier. "
            "  Read the large numeric ID label directly from the image.\n"
            "- 'main_subject_color' is SECONDARY. "
            "  It refers ONLY to the color of the SHADED BOUNDING BOX OVERLAY — "
            "  NOT the dancer's costume, outfit, skin tone, hair color, or stage lighting. "
            f"  Valid overlay color names: {sorted(valid_colors)}.\n"
            "- If the numeric ID and the box color appear inconsistent, "
            "  set 'conflict_detected' to true.\n\n"
            f"Candidates present: {candidate_lines}\n\n"
            "Return ONLY valid JSON with exactly these keys (no markdown, no fences):\n"
            "  main_subject_detection_id  — integer, PRIMARY: the numeric ID from the image label\n"
            "  main_subject_color         — string, SECONDARY: the bounding box overlay color name\n"
            "  main_subject_position      — string: 'left', 'center', 'right', or ''\n"
            "  conflict_detected          — boolean: true if ID and color appear inconsistent\n"
            "  reason                     — string: one short sentence\n\n"
            "Rules:\n"
            "- Choose the most visually important, clear, and cull-worthy dancer.\n"
            "- IGNORE dancer costume color, outfit, skin, and hair when choosing main_subject_color.\n"
            "- Use the large centered ID badge as your primary signal.\n"
            "- Do NOT output anything outside the JSON object.\n"
        )

        user_prompt = (
            "Select the single highlighted candidate that is the clearest main dance subject.\n"
            "Use the large centered numeric ID label as your primary signal.\n"
            "Return only valid JSON."
        )

        self.app.log(f"AI Cull Dance VL subject picker sending: {debug_path}")

        raw_text = client.vision_chat_text(
            model=model,
            image_path=debug_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=min(temperature, 0.1),
            max_tokens=min(max_tokens, 300),
        )

        parsed = self._extract_json_object(raw_text)

        chosen_color = str(parsed.get("main_subject_color", "")).strip().lower()
        chosen_position = str(parsed.get("main_subject_position", "")).strip().lower()
        chosen_id_raw = parsed.get("main_subject_detection_id", 0)
        reason = str(parsed.get("reason", "")).strip()
        conflict_from_vl = bool(parsed.get("conflict_detected", False))

        chosen_by_color: Detection | None = None
        chosen_by_position: Detection | None = None
        chosen_by_id: Detection | None = None

        if chosen_color in color_to_det:
            det_id = color_to_det[chosen_color]
            chosen_by_color = next((d for d in detections if d.id == det_id), None)

        if chosen_position:
            chosen_by_position = self._map_position_to_detection(detections, chosen_position)

        try:
            chosen_id = int(chosen_id_raw)
        except Exception:
            chosen_id = 0
        if chosen_id > 0:
            chosen_by_id = next((d for d in detections if d.id == chosen_id), None)

        mismatches: list[str] = []

        invalid_id = chosen_id > 0 and chosen_id not in valid_ids
        invalid_color = bool(chosen_color) and chosen_color not in valid_colors

        if invalid_id:
            mismatches.append(f"invalid_id={chosen_id} (valid ids: {sorted(valid_ids)})")
        if invalid_color:
            mismatches.append(f"invalid_color={chosen_color!r} (valid colors: {sorted(valid_colors)})")
        if chosen_by_id and chosen_by_color and chosen_by_id.id != chosen_by_color.id:
            mismatches.append(
                f"id_color_mismatch: id={chosen_id}→det{chosen_by_id.id} "
                f"vs color={chosen_color}→det{chosen_by_color.id}"
            )
        if (
            chosen_by_color is None
            and chosen_by_id
            and chosen_by_position
            and chosen_by_id.id != chosen_by_position.id
        ):
            mismatches.append(
                f"id_position_mismatch: id={chosen_id}→det{chosen_by_id.id} "
                f"vs position={chosen_position}→det{chosen_by_position.id}"
            )

        needs_manual_review = bool(mismatches) or conflict_from_vl

        if needs_manual_review:
            self.current_vl_mismatch = True
            mismatch_context: dict = {
                "mismatches": mismatches,
                "conflict_from_vl": conflict_from_vl,
                "parsed": parsed,
                "raw_vl_response": raw_text,
                "det_to_color": {str(k): v for k, v in det_to_color.items()},
                "color_to_det": color_to_det,
                "candidate_ids": sorted(valid_ids),
                "valid_colors": sorted(valid_colors),
                "vl_debug_image_path": str(debug_path),
            }
            self.current_vl_mismatch_context = mismatch_context
            self._log_manual_review_entry(image_path, mismatch_context)
            mismatch_summary = "; ".join(mismatches)
            if conflict_from_vl:
                mismatch_summary += "; VL self-reported conflict"
            self.app.log(f"AI Cull Dance VL MISMATCH — flagged for manual review: {mismatch_summary}")

        chosen = chosen_by_id or chosen_by_color or chosen_by_position
        if chosen is None:
            raise ValueError(f"Could not map VL subject picker output to any candidate: {parsed}")

        return chosen, (reason or f"Selected detection {chosen.id}")

    def _evaluate_dance_with_vl(self, image_path: Path) -> dict:
        base_url, model, timeout, temperature, max_tokens = self._dance_lmstudio_settings()
        client = LMStudioClient(base_url=base_url, timeout=timeout)
        profile = self.get_profile_data()
        rubric_name = getattr(profile, "vl_rubric_name", "generic")
        if hasattr(client, "generic_cull_rubric"):
            generic_cull = client.generic_cull_rubric
            generic_kwargs = {
                "model": model,
                "image_path": image_path,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            try:
                params = inspect.signature(generic_cull).parameters
                if "rubric_name" in params:
                    generic_kwargs["rubric_name"] = rubric_name
            except (TypeError, ValueError):
                pass
            rubric = generic_cull(**generic_kwargs)
        else:
            rubric = client.dance_cull_rubric(
                model=model,
                image_path=image_path,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        score, decision = self._score_dance_vl_rubric(rubric)
        return {
            "path": Path(image_path),
            "score": score,
            "decision": decision,
            "rubric": rubric,
            "mode": "dance_vl",
            "prefer_face": self.prefer_face_var.get(),
            "has_face": str(rubric.get("face_visibility", "")).strip().lower() in {"clear", "partial"},
            "face_focus": 0.0,
            "hero_focus": 0.0,
            "burst_suppressed": False,
            "burst_winner_paths": [],
        }

    def _evaluate_dance_full_pipeline(self, image_path: Path, config: dict) -> dict:
        previous_path = self.app.state.current_image_path
        previous_image = self.app.current_image
        previous_af = list(self.app.current_af_boxes)

        self.current_vl_debug_image_path = None
        self.current_vl_mismatch = False
        self.current_vl_mismatch_context = None
        self.current_vl_subject_reason = ""

        try:
            self.app.load_image(Path(image_path))

            profile = self.get_profile_data()
            prompts = [p.strip() for p in profile.prompts if p.strip()]
            detection_mode = config.get("detection_mode", "Phrase Only")

            if detection_mode == "Phrase Only":
                detections = self._get_phrase_only_detections_for_current_image(prompts)
            else:
                detections = self._get_hybrid_detections_for_current_image(prompts)

            people = [d for d in detections if self._is_person_label(d.label)]
            vl_candidates = self._vl_candidate_people(people)
            vl_candidates = vl_candidates[: len(self.DANCE_PICK_COLORS)]

            chosen: Detection | None = None
            chosen_reason = ""

            if bool(config.get("use_dance_vl_subject_picker", False)) and vl_candidates:
                chosen, chosen_reason = self._select_main_dance_detection_from_candidates(
                    Path(image_path),
                    vl_candidates,
                )

            img_w = self.app.current_image.width if self.app.current_image else 0
            img_h = self.app.current_image.height if self.app.current_image else 0
            crop_proposal, crop_center_penalty = self._compute_crop_proposal(
                chosen, img_w, img_h, profile.main_ratio or "4:5"
            )

            result = self._evaluate_dance_with_vl(Path(image_path))
            final_score = float(result["score"]) + crop_center_penalty
            final_decision = str(result["decision"])

            rules_hash = self._dance_rules_hash()
            cache = self._get_dance_cull_cache(Path(image_path))
            cache_entry: dict = {
                "image_path": str(image_path),
                "florence_detections": [self._det_to_dict(d) for d in detections],
                "dance_candidates": [self._det_to_dict(d) for d in vl_candidates],
                "vl_debug_image_path": (
                    str(self.current_vl_debug_image_path)
                    if self.current_vl_debug_image_path else None
                ),
                "chosen_id": chosen.id if chosen else None,
                "chosen_reason": chosen_reason,
                "vl_mismatch": self.current_vl_mismatch,
                "vl_mismatch_context": self.current_vl_mismatch_context,
                "crop_proposal": (
                    [crop_proposal.x1, crop_proposal.y1, crop_proposal.x2, crop_proposal.y2]
                    if crop_proposal else None
                ),
                "crop_center_penalty": crop_center_penalty,
                "vl_rubric": result.get("rubric", {}),
                "cull_score": final_score,
                "cull_decision": final_decision,
            }
            cache.put(Path(image_path), rules_hash, cache_entry)

            return {
                "path": Path(image_path),
                "score": final_score,
                "decision": final_decision,
                "rubric": result.get("rubric", {}),
                "mode": "dance_vl_full_pipeline",
                "prefer_face": config.get("prefer_face", True),
                "has_face": result.get("has_face", False),
                "face_focus": 0.0,
                "hero_focus": 0.0,
                "burst_suppressed": False,
                "burst_winner_paths": [],
                "crop_center_penalty": crop_center_penalty,
                "vl_mismatch": self.current_vl_mismatch,
                "vl_debug_image_path": self.current_vl_debug_image_path,
                "chosen_id": chosen.id if chosen else None,
                "chosen_reason": chosen_reason,
            }
        finally:
            self.app.state.current_image_path = previous_path
            self.app.current_image = previous_image
            self.app.current_af_boxes = previous_af

    def evaluate_image_for_pipeline(self, image_path, config: dict) -> dict:
        if bool(config.get("use_dance_vl", False)):
            return self._evaluate_dance_full_pipeline(Path(image_path), config)

        previous_path = self.app.state.current_image_path
        previous_image = self.app.current_image
        previous_af = list(self.app.current_af_boxes)

        try:
            self.app.load_image(Path(image_path))

            prompts = config["prompts"]
            detection_mode = config["detection_mode"]
            use_ball_scoring = self._prompts_include_ball(prompts)
            prefer_face = config["prefer_face"]

            if detection_mode == "Phrase Only":
                detections = self._get_phrase_only_detections_for_current_image(prompts)
            else:
                detections = self._get_hybrid_detections_for_current_image(prompts)

            people = [d for d in detections if self._is_person_label(d.label)]
            faces = [d for d in detections if self._is_face_label(d.label)]
            balls = [d for d in detections if self._is_ball_label(d.label)] if use_ball_scoring else []

            hero, has_af_match, focus_score = self._pick_hero_person(people)
            support_ball, has_support_ball = self._pick_support_ball(hero, balls) if use_ball_scoring else (None, False)
            face_det, has_face, face_focus = self._pick_face_for_hero(hero, faces) if prefer_face else (None, False, 0.0)

            img_w = self.app.current_image.width if self.app.current_image else 0
            img_h = self.app.current_image.height if self.app.current_image else 0
            _, crop_center_penalty = self._compute_crop_proposal(hero, img_w, img_h)

            old_keep = self.keep_threshold_var.get()
            old_maybe = self.maybe_threshold_var.get()
            old_blur_penalty = self.blur_penalty_threshold_var.get()
            old_blur_reject = self.blur_reject_threshold_var.get()
            old_blur_points = self.blur_penalty_points_var.get()

            self.keep_threshold_var.set(str(config["keep_threshold"]))
            self.maybe_threshold_var.set(str(config["maybe_threshold"]))
            self.blur_penalty_threshold_var.set(str(config["blur_penalty_threshold"]))
            self.blur_reject_threshold_var.set(str(config["blur_reject_threshold"]))
            self.blur_penalty_points_var.set(str(config["blur_penalty_points"]))

            try:
                score, breakdown = self._compute_keeper_score(
                    hero=hero,
                    has_af_match=has_af_match,
                    focus_score=focus_score,
                    has_support_ball=has_support_ball,
                    use_ball_scoring=use_ball_scoring,
                    has_face=has_face,
                    face_focus=face_focus,
                    prefer_face=prefer_face,
                    crop_center_penalty=crop_center_penalty,
                )
                decision, override_reason = self._decision_from_score(score, focus_score, hero is not None)
            finally:
                self.keep_threshold_var.set(old_keep)
                self.maybe_threshold_var.set(old_maybe)
                self.blur_penalty_threshold_var.set(old_blur_penalty)
                self.blur_reject_threshold_var.set(old_blur_reject)
                self.blur_penalty_points_var.set(old_blur_points)

            return {
                "path": Path(image_path),
                "detections": detections,
                "hero": hero,
                "hero_focus": focus_score,
                "has_af_match": has_af_match,
                "support_ball": support_ball,
                "has_support_ball": has_support_ball,
                "face_det": face_det,
                "has_face": has_face,
                "face_focus": face_focus,
                "score": score,
                "decision": decision,
                "override_reason": override_reason,
                "breakdown": breakdown,
                "use_ball_scoring": use_ball_scoring,
                "prefer_face": prefer_face,
                "mode": detection_mode,
                "crop_center_penalty": crop_center_penalty,
            }
        finally:
            self.app.state.current_image_path = previous_path
            self.app.current_image = previous_image
            self.app.current_af_boxes = previous_af

    def apply_burst_suppression_for_pipeline(self, results: list[dict], config: dict) -> list[dict]:
        if not results or not config.get("enable_burst", False):
            return results

        path_to_result = {Path(r["path"]): r for r in results}
        ordered_paths = [Path(r["path"]) for r in results]
        bursts = self._build_bursts(ordered_paths, float(config.get("burst_fps", 8.0)))
        keep_per_burst = max(1, int(config.get("keep_per_burst", 1)))

        for burst_paths in bursts:
            burst_results = [path_to_result[p] for p in burst_paths if p in path_to_result]
            if not burst_results:
                continue
            ranked = self._rank_burst_candidates(burst_results)
            burst_selection: dict | None = None
            burst_selection_source = "heuristic"
            if self._should_use_vl_burst_selector(ranked, keep_per_burst, config):
                try:
                    ranked, burst_selection = self._select_burst_winners_with_vl(ranked, keep_per_burst)
                    burst_selection_source = "vl"
                    best_frame = str(burst_selection.get("best_frame", "")).strip()
                    self.app.log(
                        f"AI Cull burst VL selector chose {best_frame or 'top frame'} "
                        f"for burst starting at {Path(burst_results[0]['path']).name}"
                    )
                except Exception as exc:
                    self.app.log(
                        f"AI Cull burst VL selector failed for {Path(burst_results[0]['path']).name}: {exc}"
                    )
            winners = ranked[:keep_per_burst]
            winner_paths = [Path(item["path"]) for item in winners]
            winner_path_set = set(winner_paths)

            for item in burst_results:
                item["burst_size"] = len(burst_results)
                item["burst_winner_paths"] = [str(p) for p in winner_paths]
                item["burst_selection_source"] = burst_selection_source
                item["burst_vl_reason"] = str((burst_selection or {}).get("reason", "")).strip()
                item["burst_vl_confidence"] = (burst_selection or {}).get("confidence")
                if Path(item["path"]) not in winner_path_set:
                    item["decision"] = "Reject"
                    item["burst_suppressed"] = True
                else:
                    item["burst_suppressed"] = False

        return results

    def on_image_changed(self):
        self.current_score = 0.0
        self.current_decision = "Reject"
        self.current_hero_id = None
        self.current_ball_id = None
        self.current_vl_rubric = None
        self.current_vl_subject_reason = ""
        self.current_vl_debug_image_path = None
        self.current_vl_mismatch = False
        self.current_vl_mismatch_context = None
        self.app.clear_debug_views()

        self._refresh_dynamic_sections()

        if self.app.current_image is None:
            self.app.set_manual_boxes([])
            self.app.set_manual_selected_ids(set())
            self.app.set_overlays([])
            return

        runtime_config = self.get_runtime_config()

        if str(runtime_config.get("sport_type", "")).lower() == "dance" or bool(runtime_config.get("use_dance_vl", False)):
            profile = self.get_profile_data()
            prompts = [p.strip() for p in profile.prompts if p.strip()]
            detection_mode = self.detection_mode_var.get().strip() or "Phrase Only"

            if detection_mode == "Phrase Only":
                detections = self._get_phrase_only_detections_for_current_image(prompts)
            else:
                detections = self._get_hybrid_detections_for_current_image(prompts)

            people = [d for d in detections if self._is_person_label(d.label)]
            vl_candidates = self._vl_candidate_people(people)
            vl_candidates = vl_candidates[: len(self.DANCE_PICK_COLORS)]

            self.app.set_manual_boxes(people)

            chosen: Detection | None = None
            chosen_reason = ""
            original_preview = self._scale_image_to_long_edge(
                self._load_rgb_image(self.app.state.current_image_path),
                self.VL_TARGET_LONG_EDGE,
            )
            florence_preview = self._draw_florence_candidates_preview(
                self.app.state.current_image_path, vl_candidates
            )
            vl_input_preview = None
            final_preview = self._draw_final_subject_preview(self.app.state.current_image_path, None)

            if bool(runtime_config.get("use_dance_vl_subject_picker", False)) and vl_candidates:
                try:
                    chosen, chosen_reason = self._select_main_dance_detection_from_candidates(
                        self.app.state.current_image_path,
                        vl_candidates,
                    )
                    if self.current_vl_debug_image_path is not None:
                        vl_input_preview = self.current_vl_debug_image_path
                    final_preview = self._draw_final_subject_preview(
                        self.app.state.current_image_path, chosen
                    )
                    self.current_vl_subject_reason = chosen_reason
                    self.app.log(
                        f"AI Cull Dance VL subject picker: "
                        f'selected id={chosen.id if chosen else "none"} '
                        f'reason="{chosen_reason}"'
                    )
                    if self.current_vl_debug_image_path is not None:
                        self.app.log(f"AI Cull Dance VL debug image: {self.current_vl_debug_image_path}")
                except Exception as exc:
                    self.app.log(f"AI Cull Dance VL subject picker failed: {exc}")

            img_w = self.app.current_image.width if self.app.current_image else 0
            img_h = self.app.current_image.height if self.app.current_image else 0
            crop_proposal, crop_center_penalty = self._compute_crop_proposal(
                chosen, img_w, img_h, profile.main_ratio or "4:5"
            )
            if crop_center_penalty < 0:
                self.app.log(
                    f"AI Cull Dance: crop centre penalty={crop_center_penalty:.1f} "
                    "(subject off-centre)"
                )

            selected_ids: set[int] = set()
            overlays: list[CropBox] = []

            if chosen is not None:
                self.current_hero_id = chosen.id
                selected_ids.add(chosen.id)
                overlays.append(CropBox(name="Dance_VL_Subject", bbox=chosen.bbox, color="#FFD400"))

            self.app.set_manual_selected_ids(selected_ids)
            self.app.set_overlays(overlays)

            if bool(runtime_config.get("show_dance_debug_preview", False)):
                if vl_input_preview is None and self.current_vl_debug_image_path is not None:
                    vl_input_preview = self.current_vl_debug_image_path

                self.app.set_debug_views([
                    ("Florence Candidates", florence_preview),
                    ("VL Input", vl_input_preview),
                    ("Final Subject", final_preview),
                    ("Original", original_preview),
                ])

            rules_hash = self._dance_rules_hash()
            cache = self._get_dance_cull_cache(self.app.state.current_image_path)
            cache_entry: dict = {
                "image_path": str(self.app.state.current_image_path),
                "florence_detections": [self._det_to_dict(d) for d in detections],
                "dance_candidates": [self._det_to_dict(d) for d in vl_candidates],
                "vl_debug_image_path": (
                    str(self.current_vl_debug_image_path)
                    if self.current_vl_debug_image_path else None
                ),
                "chosen_id": chosen.id if chosen else None,
                "chosen_reason": chosen_reason,
                "vl_mismatch": self.current_vl_mismatch,
                "vl_mismatch_context": self.current_vl_mismatch_context,
                "crop_proposal": (
                    [crop_proposal.x1, crop_proposal.y1, crop_proposal.x2, crop_proposal.y2]
                    if crop_proposal else None
                ),
                "crop_center_penalty": crop_center_penalty,
            }

            if bool(runtime_config.get("use_dance_vl", False)):
                try:
                    result = self._evaluate_dance_with_vl(self.app.state.current_image_path)
                    self.current_score = float(result["score"]) + crop_center_penalty
                    self.current_decision = str(result["decision"])
                    self.current_vl_rubric = dict(result.get("rubric", {}))
                    self.app.log(
                        "AI Cull Dance VL: "
                        f"decision={self.current_decision} "
                        f"score={self.current_score:.1f} "
                        f"(crop_penalty={crop_center_penalty:.1f}) "
                        f"summary={self.current_vl_rubric.get('summary', '')}"
                    )
                    cache_entry["cull_score"] = self.current_score
                    cache_entry["cull_decision"] = self.current_decision
                    cache_entry["vl_rubric"] = self.current_vl_rubric
                    cache.put(self.app.state.current_image_path, rules_hash, cache_entry)
                    return
                except Exception as exc:
                    self.app.log(f"AI Cull Dance VL failed, falling back to heuristic cull: {exc}")

            cache.put(self.app.state.current_image_path, rules_hash, cache_entry)

        profile = self.get_profile_data()
        prompts = [p.strip() for p in profile.prompts if p.strip()]
        detection_mode = self.detection_mode_var.get().strip() or "Phrase Only"
        use_ball_scoring = self._prompts_include_ball(prompts)
        prefer_face = self.prefer_face_var.get()

        if detection_mode == "Phrase Only":
            detections = self._get_phrase_only_detections_for_current_image(prompts)
        else:
            detections = self._get_hybrid_detections_for_current_image(prompts)

        self.app.set_manual_boxes(detections)

        if not detections:
            self.app.set_manual_selected_ids(set())
            self.app.set_overlays([])
            self.current_score = 0.0
            self.current_decision = "Reject"
            self.app.log("AI Cull: no detections -> Reject")
            return

        people = [d for d in detections if self._is_person_label(d.label)]
        faces = [d for d in detections if self._is_face_label(d.label)]

        support_ball = None
        has_support_ball = False

        if use_ball_scoring:
            balls = [d for d in detections if self._is_ball_label(d.label)]
        else:
            balls = []

        hero, has_af_match, focus_score = self._pick_hero_person(people)
        face_det, has_face, face_focus = self._pick_face_for_hero(hero, faces) if prefer_face else (None, False, 0.0)

        if use_ball_scoring:
            support_ball, has_support_ball = self._pick_support_ball(hero, balls)

        score, breakdown = self._compute_keeper_score(
            hero=hero,
            has_af_match=has_af_match,
            focus_score=focus_score,
            has_support_ball=has_support_ball,
            use_ball_scoring=use_ball_scoring,
            has_face=has_face,
            face_focus=face_focus,
            prefer_face=prefer_face,
        )
        decision, override_reason = self._decision_from_score(
            score=score,
            hero_focus_score=focus_score,
            hero_exists=(hero is not None),
        )

        self.current_score = score
        self.current_decision = decision
        self.current_hero_id = hero.id if hero is not None else None
        self.current_ball_id = support_ball.id if (support_ball is not None and has_support_ball) else None

        selected_ids = set()
        overlays: list[CropBox] = []

        if hero is not None:
            selected_ids.add(hero.id)
            hero_color = "#FF3333" if override_reason == "blur_reject" else "#FFD400"
            overlays.append(CropBox(name="Cull_Hero", bbox=hero.bbox, color=hero_color))

        if face_det is not None and prefer_face:
            selected_ids.add(face_det.id)
            overlays.append(CropBox(name="Cull_Face", bbox=face_det.bbox, color="#00FFAA"))

        if use_ball_scoring and support_ball is not None and has_support_ball:
            selected_ids.add(support_ball.id)
            overlays.append(CropBox(name="Cull_Ball", bbox=support_ball.bbox, color="#00BFFF"))

        self.app.set_manual_selected_ids(selected_ids)
        self.app.set_overlays(overlays)

        hero_label = hero.label if hero is not None else "none"

        breakdown_parts = [
            f'person:{breakdown["person"]:.1f}',
            f'af:{breakdown["af"]:.1f}',
            f'focus:{breakdown["focus"]:.1f}',
            f'size:{breakdown["size"]:.1f}',
            f'blur_penalty:{breakdown["blur_penalty"]:.1f}',
        ]
        if prefer_face:
            breakdown_parts.append(f'face:{breakdown.get("face", 0.0):.1f}')
            breakdown_parts.append(f'face_focus:{breakdown.get("face_focus", 0.0):.1f}')
        if use_ball_scoring:
            breakdown_parts.append(f'ball:{breakdown["ball"]:.1f}')
        if "crop_center" in breakdown:
            breakdown_parts.append(f'crop_center:{breakdown["crop_center"]:.1f}')

        log_message = (
            "AI Cull: "
            f'mode="{detection_mode}" '
            f'decision={decision} '
            f"score={score:.1f} "
            f'hero="{hero_label}" '
            f"af_match={has_af_match} "
            f"focus={focus_score:.1f} "
        )

        if prefer_face:
            log_message += f"has_face={has_face} face_focus={face_focus:.1f} "

        if use_ball_scoring:
            log_message += f"ball={has_support_ball} "

        if override_reason == "blur_reject":
            log_message += "override=blur_reject "
        elif breakdown["blur_penalty"] < 0:
            log_message += "override=blur_penalty "

        log_message += "breakdown=" + ",".join(breakdown_parts)
        self.app.log(log_message)

    def rerun(self):
        self.on_image_changed()

    def _get_cull_output_dir(self, decision: str) -> Path | None:
        if self.app.state.input_folder is None:
            return None
        out_dir = self.app.state.input_folder / "Cull" / decision
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _copy_image_to_decision_folder(self, source: Path, decision: str, score: float):
        output_dir = self._get_cull_output_dir(decision)
        if output_dir is None:
            self.app.log("AI Cull: no input folder selected.")
            return

        destination = output_dir / source.name
        shutil.copy2(source, destination)
        self.app.log(f"AI Cull: copied {source.name} -> {decision} ({score:.1f})")

    def stop_auto_cull(self):
        if not self.auto_running:
            return
        self.auto_cancel_requested = True
        self.app.log("AI Cull: stop requested...")

    def auto_cull_input_folder(self):
        if self.auto_running:
            self.app.log("AI Cull: auto cull already running.")
            return

        if self.app.state.input_folder is None:
            self.app.log("AI Cull: please select an input folder first.")
            return

        if not self.app.state.image_paths:
            self.app.log("AI Cull: loading images from input folder...")
            self.app.start_batch()

        self.auto_images = [Path(p) for p in self.app.state.image_paths]
        if not self.auto_images:
            self.app.log("AI Cull: no images found in input folder.")
            return

        self.auto_results = []
        self.auto_index = 0
        self.auto_running = True
        self.auto_cancel_requested = False

        if self.auto_button is not None:
            self.auto_button.config(state="disabled")
        if self.stop_button is not None:
            self.stop_button.config(state="normal")

        self.app.log(
            f"AI Cull: starting auto cull on {len(self.auto_images)} image(s) "
            "(full per-image Florence + VL workflow, immediate save, burst suppression skipped)..."
        )
        self.app.root.after(10, self._auto_cull_step)

    def _auto_cull_step(self):
        if self.auto_cancel_requested:
            self._finish_auto_cull(cancelled=True)
            return

        if self.auto_index >= len(self.auto_images):
            keep_count = sum(1 for r in self.auto_results if r.get("decision") == "Keep")
            maybe_count = sum(1 for r in self.auto_results if r.get("decision") == "Maybe")
            reject_count = sum(1 for r in self.auto_results if r.get("decision") == "Reject")
            self.app.log(
                f"AI Cull: complete. Keep={keep_count}, Maybe={maybe_count}, Reject={reject_count}"
            )
            self._finish_auto_cull(cancelled=False)
            return

        image_path = self.auto_images[self.auto_index]
        self.app.state.current_index = self.auto_index
        self.app.load_current_image()

        try:
            config = self.get_runtime_config()
            config["enable_burst"] = False

            result = self.evaluate_image_for_pipeline(image_path, config)
            self.auto_results.append(result)

            decision = result.get("decision", "Reject")
            if decision not in ("Keep", "Maybe", "Reject"):
                decision = "Reject"
            score = float(result.get("score", 0.0))

            self._copy_image_to_decision_folder(Path(image_path), decision, score)

            self.app.log(
                f"AI Cull Auto {self.auto_index + 1}/{len(self.auto_images)}: "
                f"{image_path.name} -> {decision} score={score:.1f}"
            )
        except Exception as exc:
            self.app.log(f"AI Cull: failed on {image_path.name}: {exc}")

        self.auto_index += 1
        self.app.root.after(1, self._auto_cull_step)

    def _finish_auto_cull(self, cancelled: bool):
        self.auto_running = False
        self.auto_cancel_requested = False

        if self.auto_button is not None:
            self.auto_button.config(state="normal")
        if self.stop_button is not None:
            self.stop_button.config(state="disabled")

        if cancelled:
            self.app.log("AI Cull: auto cull cancelled.")

    def approve(self):
        if self.app.state.current_image_path is None:
            return

        source = self.app.state.current_image_path
        decision = self.current_decision
        score = self.current_score

        if decision not in ("Keep", "Maybe", "Reject"):
            decision = "Reject"

        self._copy_image_to_decision_folder(source, decision, score)
        self.app.next_image()