from __future__ import annotations

import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ExifTags, ImageDraw, ImageFont

from core import (
    compute_iou,
    get_focus_score,
    run_florence_od_detection,
    run_florence_phrase_detection,
)
from lmstudio_client import LMStudioClient
from models import CropBox, Detection, SportProfile, BoundingBox


class AICullTool:
    tool_id = "ai_cull"
    display_name = "AI Cull Tool"

    DANCE_PICK_COLORS = [
        ("red", "#FF4D4D"),
        ("yellow", "#FFD84D"),
        ("cyan", "#33D6FF"),
        ("lime", "#7CFF4D"),
        ("magenta", "#FF5CFF"),
        ("orange", "#FF9A3D"),
    ]

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
        self.prefer_face_var = tk.BooleanVar(value=True)

        self.use_dance_vl_var = tk.BooleanVar(value=True)
        self.use_dance_vl_subject_picker_var = tk.BooleanVar(value=True)
        self.save_vl_debug_images_var = tk.BooleanVar(value=True)

        self.current_score = 0.0
        self.current_decision = "Reject"
        self.current_hero_id: int | None = None
        self.current_ball_id: int | None = None
        self.current_vl_rubric: dict | None = None
        self.current_vl_subject_reason: str = ""
        self.current_vl_debug_image_path: Path | None = None

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
            text="Dance VL Culling",
            bg="#2a2a2a",
            fg="#ffd27f",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        tk.Checkbutton(
            self.dance_frame,
            text="Use LM Studio VL Dance Rubric",
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
        if self._current_profile_is_dance():
            self.dance_frame.pack(fill="x", padx=10, pady=(12, 4))
        else:
            self.dance_frame.pack_forget()

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
            dance_prefer_full_body=getattr(base_profile, "dance_prefer_full_body", True),
            dance_penalize_cropped_feet=getattr(base_profile, "dance_penalize_cropped_feet", True),
            dance_favor_symmetry=getattr(base_profile, "dance_favor_symmetry", False),
            dance_favor_peak_action=getattr(base_profile, "dance_favor_peak_action", True),
            dance_prefer_clean_pose=getattr(base_profile, "dance_prefer_clean_pose", True),
            dance_prefer_single_subject=getattr(base_profile, "dance_prefer_single_subject", False),
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
            "prefer_face": bool(self.prefer_face_var.get()),
            "sport_type": getattr(profile, "sport_type", "generic"),
            "use_dance_vl": bool(self.use_dance_vl_var.get()),
            "use_dance_vl_subject_picker": bool(self.use_dance_vl_subject_picker_var.get()),
            "save_vl_debug_images": bool(self.save_vl_debug_images_var.get()),
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

        if sharpness == "blurry" or subject_visibility == "weak":
            return score, "Reject"
        if pose_quality == "unclear" and moment_quality == "weak":
            return score, "Reject"
        if feet_visibility == "cropped_out" or full_body_visibility == "partial":
            if score >= 55:
                return score, "Maybe"

        if score >= 55:
            return score, "Keep"
        if score >= 30:
            return score, "Maybe"
        return score, "Reject"

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

    def _draw_shaded_candidate_image(self, image_path: Path, detections: list[Detection]) -> tuple[Path, dict[int, str], dict[str, int]]:
        with Image.open(image_path) as img:
            base = img.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            font = ImageFont.load_default()

            det_to_color: dict[int, str] = {}
            color_to_det: dict[str, int] = {}

            line_width = max(3, int(max(base.size) * 0.004))

            for idx, det in enumerate(detections):
                color_name, color_hex = self.DANCE_PICK_COLORS[idx % len(self.DANCE_PICK_COLORS)]
                det_to_color[det.id] = color_name
                color_to_det[color_name] = det.id

                bbox = det.bbox
                fill_rgba = self._hex_to_rgba(color_hex, 70)
                outline_rgba = self._hex_to_rgba(color_hex, 255)

                draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], fill=fill_rgba, outline=outline_rgba, width=line_width)

                label = f"{det.id} / {color_name}"
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                    text_w = right - left
                    text_h = bottom - top
                except Exception:
                    text_w = max(40, len(label) * 7)
                    text_h = 14

                pad = 5
                label_x = max(0, bbox.x1)
                label_y = max(0, bbox.y1 - text_h - pad * 2)

                draw.rectangle(
                    [label_x, label_y, label_x + text_w + pad * 2, label_y + text_h + pad * 2],
                    fill=(0, 0, 0, 210),
                    outline=outline_rgba,
                    width=1,
                )
                draw.text((label_x + pad, label_y + pad), label, fill=(255, 255, 255, 255), font=font)

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

        system_prompt = (
            "You are a dance recital photo subject selection assistant.\n\n"
            "The image contains shaded colored candidate dancer regions.\n"
            "Choose the single candidate that best corresponds to the main dancer to evaluate for culling.\n\n"
            "Return only valid JSON.\n"
            "Do not use markdown.\n"
            "Do not include code fences.\n"
            "Do not include any text before or after the JSON.\n\n"
            "Use exactly these keys:\n"
            "- main_subject_color\n"
            "- main_subject_position\n"
            "- main_subject_detection_id\n"
            "- reason\n\n"
            "Rules:\n"
            "- main_subject_color must be one of the visible color names\n"
            '- main_subject_position must be one of: "left", "center", "right", or ""\n'
            "- main_subject_detection_id must be one of the visible numeric labels or 0 if uncertain\n"
            "- reason must be one short sentence\n"
            "- Prefer the most visually important, readable, and cull-worthy dancer.\n"
        )

        user_prompt = (
            "Select the single highlighted candidate that is the clearest main dance subject for this frame.\n"
            "Prioritize the colored region over the numeric label if they seem inconsistent.\n"
            "Return only valid JSON."
        )

        self.app.log(f"AI Cull Dance VL subject picker sending: {debug_path}")

        text = client.vision_chat_text(
            model=model,
            image_path=debug_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=min(temperature, 0.1),
            max_tokens=min(max_tokens, 250),
        )

        parsed = self._extract_json_object(text)

        chosen_color = str(parsed.get("main_subject_color", "")).strip().lower()
        chosen_position = str(parsed.get("main_subject_position", "")).strip().lower()
        chosen_id_raw = parsed.get("main_subject_detection_id", 0)
        reason = str(parsed.get("reason", "")).strip()

        chosen_by_color = None
        chosen_by_position = None
        chosen_by_id = None

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

        chosen = chosen_by_color or chosen_by_position or chosen_by_id

        if chosen is None:
            raise ValueError(f"Could not map VL subject picker output: {parsed}")

        if chosen_by_color and chosen_by_id and chosen_by_color.id != chosen_by_id.id:
            self.app.log(
                f"AI Cull Dance VL mismatch: color={chosen_color}->{chosen_by_color.id}, "
                f"id={chosen_id}->{chosen_by_id.id}; trusting color."
            )

        if chosen_by_color is None and chosen_by_position and chosen_by_id and chosen_by_position.id != chosen_by_id.id:
            self.app.log(
                f"AI Cull Dance VL mismatch: position={chosen_position}->{chosen_by_position.id}, "
                f"id={chosen_id}->{chosen_by_id.id}; trusting position."
            )

        return chosen, (reason or f"Selected detection {chosen.id}")

    def _evaluate_dance_with_vl(self, image_path: Path) -> dict:
        base_url, model, timeout, temperature, max_tokens = self._dance_lmstudio_settings()
        client = LMStudioClient(base_url=base_url, timeout=timeout)
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

    def _get_capture_timestamp(self, image_path: Path) -> float:
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if exif:
                    exif_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                        if key in exif_map:
                            dt = datetime.strptime(str(exif_map[key]), "%Y:%m:%d %H:%M:%S")
                            return dt.timestamp()
        except Exception:
            pass
        return image_path.stat().st_mtime

    def _build_bursts(self, image_paths: list[Path], fps: float) -> list[list[Path]]:
        if not image_paths:
            return []
        max_gap = 1.0 / max(fps, 0.001)

        stamped = [(Path(p), self._get_capture_timestamp(Path(p))) for p in image_paths]
        stamped.sort(key=lambda x: x[1])

        bursts: list[list[Path]] = []
        current = [stamped[0][0]]

        for i in range(1, len(stamped)):
            prev_t = stamped[i - 1][1]
            cur_p, cur_t = stamped[i]
            if (cur_t - prev_t) <= max_gap:
                current.append(cur_p)
            else:
                bursts.append(current)
                current = [cur_p]
        bursts.append(current)
        return bursts

    def _rank_burst_candidates(self, items: list[dict]) -> list[dict]:
        def key(item):
            return (
                self._decision_rank(item["decision"]),
                1 if (item.get("prefer_face") and item.get("has_face")) else 0,
                item["score"],
                item.get("face_focus", 0.0),
                item.get("hero_focus", 0.0),
            )
        return sorted(items, key=key, reverse=True)

    def evaluate_image_for_pipeline(self, image_path, config: dict) -> dict:
        if str(config.get("sport_type", "")).lower() == "dance" and bool(config.get("use_dance_vl", False)):
            return self._evaluate_dance_with_vl(Path(image_path))

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
            ranked = self._rank_burst_candidates(burst_results)
            winners = ranked[:keep_per_burst]
            winner_paths = {Path(item["path"]) for item in winners}

            for item in burst_results:
                item["burst_size"] = len(burst_results)
                item["burst_winner_paths"] = [str(p) for p in winner_paths]
                if Path(item["path"]) not in winner_paths:
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

        self._refresh_dynamic_sections()

        if self.app.current_image is None:
            self.app.set_manual_boxes([])
            self.app.set_manual_selected_ids(set())
            self.app.set_overlays([])
            return

        runtime_config = self.get_runtime_config()

        if str(runtime_config.get("sport_type", "")).lower() == "dance":
            prompts = [p.strip() for p in self.get_profile_data().prompts if p.strip()]
            detection_mode = self.detection_mode_var.get().strip() or "Phrase Only"

            if detection_mode == "Phrase Only":
                detections = self._get_phrase_only_detections_for_current_image(prompts)
            else:
                detections = self._get_hybrid_detections_for_current_image(prompts)

            people = [d for d in detections if self._is_person_label(d.label)]
            people = people[: len(self.DANCE_PICK_COLORS)]

            self.app.set_manual_boxes(people)

            chosen = None
            chosen_reason = ""

            if bool(runtime_config.get("use_dance_vl_subject_picker", False)) and people:
                try:
                    chosen, chosen_reason = self._select_main_dance_detection_from_candidates(
                        self.app.state.current_image_path,
                        people,
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

            selected_ids = set()
            overlays: list[CropBox] = []

            if chosen is not None:
                self.current_hero_id = chosen.id
                selected_ids.add(chosen.id)
                overlays.append(CropBox(name="Dance_VL_Subject", bbox=chosen.bbox, color="#FFD400"))

            self.app.set_manual_selected_ids(selected_ids)
            self.app.set_overlays(overlays)

            if bool(runtime_config.get("use_dance_vl", False)):
                try:
                    result = self._evaluate_dance_with_vl(self.app.state.current_image_path)
                    self.current_score = float(result["score"])
                    self.current_decision = str(result["decision"])
                    self.current_vl_rubric = dict(result.get("rubric", {}))
                    self.app.log(
                        "AI Cull Dance VL: "
                        f"decision={self.current_decision} "
                        f"score={self.current_score:.1f} "
                        f"summary={self.current_vl_rubric.get('summary', '')}"
                    )
                    return
                except Exception as exc:
                    self.app.log(f"AI Cull Dance VL failed, falling back to heuristic cull: {exc}")

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

        self.app.log(f"AI Cull: starting auto cull on {len(self.auto_images)} image(s)...")
        self.app.root.after(10, self._auto_cull_step)

    def _auto_cull_step(self):
        if self.auto_cancel_requested:
            self._finish_auto_cull(cancelled=True)
            return

        if self.auto_index >= len(self.auto_images):
            config = self.get_runtime_config()
            self.auto_results = self.apply_burst_suppression_for_pipeline(self.auto_results, config)

            keep_count = 0
            maybe_count = 0
            reject_count = 0

            for result in self.auto_results:
                decision = result["decision"]
                path = Path(result["path"])
                score = float(result.get("score", 0.0))

                if decision not in ("Keep", "Maybe", "Reject"):
                    decision = "Reject"

                self._copy_image_to_decision_folder(path, decision, score)

                if decision == "Keep":
                    keep_count += 1
                elif decision == "Maybe":
                    maybe_count += 1
                else:
                    reject_count += 1

            self.app.log(
                f"AI Cull: complete. Keep={keep_count}, Maybe={maybe_count}, Reject={reject_count}"
            )
            self._finish_auto_cull(cancelled=False)
            return

        image_path = self.auto_images[self.auto_index]
        self.app.state.current_index = self.auto_index
        self.app.load_current_image()

        try:
            result = self.evaluate_image_for_pipeline(image_path, self.get_runtime_config())
            self.auto_results.append(result)
            self.app.log(
                f"AI Cull Auto {self.auto_index + 1}/{len(self.auto_images)}: "
                f"{image_path.name} -> {result['decision']} score={result['score']:.1f}"
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