from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from core import (
    build_center_crop,
    build_crop_around_subject,
    compute_iou,
    get_focus_score,
    parse_ratio,
    run_florence_od_detection,
    run_florence_phrase_detection,
    shift_box_to_fit,
    union_boxes,
)
from lmstudio_client import LMStudioClient
from models import CropBox, Detection, SportProfile, BoundingBox


class AICropTool:
    tool_id = "ai_crop"
    display_name = "AI Crop Tool"

    def __init__(self, app):
        self.app = app
        self.panel = None

        self.prompt_vars = [tk.StringVar() for _ in range(4)]
        self.margin_var = tk.StringVar(value="12")
        self.main_ratio_var = tk.StringVar(value="4:5")
        self.auto_rotate_var = tk.BooleanVar(value=True)
        self.safe_23_var = tk.BooleanVar(value=True)
        self.safe_57_var = tk.BooleanVar(value=True)
        self.safe_11_var = tk.BooleanVar(value=False)
        self.detection_mode_var = tk.StringVar(value="Hybrid")

        self.auto_running = False
        self.auto_cancel_requested = False
        self.auto_images: list[Path] = []
        self.auto_index = 0
        self.auto_saved_total = 0
        self.auto_button = None
        self.stop_button = None

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")

        pad = {"padx": 10, "pady": 4}
        tk.Label(
            self.panel,
            text="AI Crop Settings",
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

        tk.Label(self.panel, text="Main Ratio", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        ttk.Combobox(
            self.panel,
            textvariable=self.main_ratio_var,
            values=["4:5", "5:7", "2:3", "1:1", "16:9"],
            state="readonly",
        ).pack(fill="x", **pad)

        tk.Label(self.panel, text="Margin Buffer %", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.margin_var).pack(fill="x", **pad)

        tk.Checkbutton(
            self.panel,
            text="Auto rotate ratio",
            variable=self.auto_rotate_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        safe_frame = tk.Frame(self.panel, bg="#2a2a2a")
        safe_frame.pack(fill="x", padx=10, pady=4)
        tk.Label(safe_frame, text="Safe Ratios:", bg="#2a2a2a", fg="white").pack(anchor="w")
        tk.Checkbutton(safe_frame, text="2:3", variable=self.safe_23_var, bg="#2a2a2a", fg="white", selectcolor="#444").pack(side=tk.LEFT)
        tk.Checkbutton(safe_frame, text="5:7", variable=self.safe_57_var, bg="#2a2a2a", fg="white", selectcolor="#444").pack(side=tk.LEFT)
        tk.Checkbutton(safe_frame, text="1:1", variable=self.safe_11_var, bg="#2a2a2a", fg="white", selectcolor="#444").pack(side=tk.LEFT)

        tk.Button(self.panel, text="Rerun AI Crop", command=self.rerun).pack(fill="x", padx=10, pady=(10, 4))
        tk.Button(self.panel, text="Approve Crop", command=self.approve).pack(fill="x", padx=10, pady=(0, 4))

        self.auto_button = tk.Button(self.panel, text="Auto Crop Input Folder", command=self.auto_crop_input_folder)
        self.auto_button.pack(fill="x", padx=10, pady=(0, 4))

        self.stop_button = tk.Button(
            self.panel,
            text="Stop Auto Crop",
            command=self.stop_auto_crop,
            state="disabled",
            bg="#8b1e1e",
            fg="white",
        )
        self.stop_button.pack(fill="x", padx=10, pady=(0, 4))

        return self.panel

    def apply_profile(self, profile: SportProfile):
        prompts = list(profile.prompts[:4])
        while len(prompts) < 4:
            prompts.append("")
        for i, var in enumerate(self.prompt_vars):
            var.set(prompts[i])

        self.margin_var.set(str(profile.margin_buffer))
        self.main_ratio_var.set(str(profile.main_ratio))
        self.auto_rotate_var.set(bool(profile.auto_rotate))
        self.safe_23_var.set(bool(profile.safe_ratios.get("2:3", True)))
        self.safe_57_var.set(bool(profile.safe_ratios.get("5:7", True)))
        self.safe_11_var.set(bool(profile.safe_ratios.get("1:1", False)))

    def get_profile_data(self) -> SportProfile:
        profile_name = self.app.get_selected_profile_name() or "Generic Sport"
        return SportProfile(
            name=profile_name,
            prompts=[v.get().strip() for v in self.prompt_vars],
            focus_min=0.0,
            focus_relative=0.0,
            edge_margin=0,
            margin_buffer=float(self.margin_var.get().strip() or "12"),
            main_ratio=self.main_ratio_var.get().strip() or "4:5",
            auto_rotate=self.auto_rotate_var.get(),
            join_descriptors=False,
            safe_ratios={
                "2:3": self.safe_23_var.get(),
                "5:7": self.safe_57_var.get(),
                "1:1": self.safe_11_var.get(),
            },
        )

    def get_runtime_config(self) -> dict:
        profile = self.get_profile_data()
        prompts = [p.strip() for p in profile.prompts if p.strip()]
        return {
            "prompts": prompts,
            "detection_mode": self.detection_mode_var.get().strip() or "Hybrid",
            "margin_buffer": float(self.margin_var.get().strip() or "12"),
            "main_ratio": self.main_ratio_var.get().strip() or "4:5",
            "auto_rotate": bool(self.auto_rotate_var.get()),
            "safe_ratios": {
                "2:3": bool(self.safe_23_var.get()),
                "5:7": bool(self.safe_57_var.get()),
                "1:1": bool(self.safe_11_var.get()),
            },
        }

    def _is_ball_label(self, label: str) -> bool:
        label = label.lower()
        return "ball" in label

    def _is_person_label(self, label: str) -> bool:
        label = label.lower()
        person_terms = ["person", "player", "athlete", "goalkeeper", "man", "woman", "boy", "girl"]
        return any(term in label for term in person_terms)

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

    def _pick_hero_person(self, person_detections: list[Detection]) -> tuple[Detection | None, str]:
        if not person_detections:
            return None, "no person candidates"

        af_matches = [d for d in person_detections if self._overlaps_any_af(d.bbox)]
        if af_matches:
            scored = []
            for det in af_matches:
                score = get_focus_score(self.app.current_image, det.bbox)
                scored.append((score, det))
                self.app.log(f'AF candidate "{det.label}" focus={score:.1f}')
            scored.sort(key=lambda item: item[0], reverse=True)
            hero = scored[0][1]
            return hero, f'AF+focus hero="{hero.label}" score={scored[0][0]:.1f}'

        ranked = []
        for det in person_detections:
            dist = self._distance_to_nearest_af(det.bbox)
            focus = get_focus_score(self.app.current_image, det.bbox)
            ranked.append((dist, -focus, det))
            self.app.log(f'Near-AF candidate "{det.label}" dist={dist:.1f} focus={focus:.1f}')
        ranked.sort(key=lambda item: (item[0], item[1]))
        hero = ranked[0][2]
        return hero, f'nearest-AF hero="{hero.label}"'

    def _pick_support_ball(self, hero: Detection | None, balls: list[Detection]) -> Detection | None:
        if hero is None or not balls:
            return None

        nearest_ball = min(balls, key=lambda d: self._distance_between_boxes(hero.bbox, d.bbox))
        hero_diag = math.hypot(hero.bbox.width, hero.bbox.height)
        ball_dist = self._distance_between_boxes(hero.bbox, nearest_ball.bbox)

        if ball_dist <= max(hero_diag * 1.5, 180):
            return nearest_ball
        return None

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
        return (str(image_path), tuple(prompts), mode)

    def _find_cached_cull_entry(self, image_path: Path, config: dict) -> dict | None:
        cache_entries = config.get("cached_cull_entries")
        if not isinstance(cache_entries, dict):
            return None
        return cache_entries.get(image_path.name.lower())

    def _bbox_from_cached_entry(self, image_path: Path, cache_entry: dict) -> BoundingBox | None:
        chosen_id = cache_entry.get("chosen_id")
        if chosen_id is None:
            self.app.log(f"AI Crop cache: missing chosen_id for {image_path.name}; falling back to detection.")
            return None

        try:
            chosen_id = int(chosen_id)
        except Exception:
            self.app.log(f"AI Crop cache: invalid chosen_id for {image_path.name}; falling back to detection.")
            return None

        candidates = cache_entry.get("dance_candidates", [])
        if not isinstance(candidates, list):
            self.app.log(f"AI Crop cache: invalid dance_candidates for {image_path.name}; falling back to detection.")
            return None

        chosen = None
        for d in candidates:
            try:
                if int(d.get("id", -1)) == chosen_id:
                    chosen = d
                    break
            except Exception:
                continue
        if chosen is None:
            self.app.log(
                f"AI Crop cache: chosen_id={chosen_id} not found in dance_candidates for {image_path.name}; "
                "falling back to detection."
            )
            return None

        bbox = chosen.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            self.app.log(f"AI Crop cache: invalid bbox for {image_path.name}; falling back to detection.")
            return None

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            self.app.log(f"AI Crop cache: non-numeric bbox for {image_path.name}; falling back to detection.")
            return None

        if x2 <= x1 or y2 <= y1:
            self.app.log(f"AI Crop cache: degenerate bbox for {image_path.name}; falling back to detection.")
            return None

        return BoundingBox(x1, y1, x2, y2)

    def _is_bbox_near_edge(self, bbox: BoundingBox, img_w: int, img_h: int) -> bool:
        edge_x = max(8, int(round(img_w * 0.04)))
        edge_y = max(8, int(round(img_h * 0.04)))
        return (
            bbox.x1 <= edge_x
            or bbox.y1 <= edge_y
            or bbox.x2 >= (img_w - edge_x)
            or bbox.y2 >= (img_h - edge_y)
        )

    def _tight_edge_crop(self, bbox: BoundingBox, img_w: int, img_h: int, config: dict) -> BoundingBox:
        width = max(1, bbox.width)
        height = max(1, bbox.height)

        tight_w_margin = int(round(width * 0.06))
        tight_bottom = int(round(height * 0.80))
        tight_box = BoundingBox(
            x1=max(0, bbox.x1 + tight_w_margin),
            y1=max(0, bbox.y1),
            x2=min(img_w, bbox.x2 - tight_w_margin),
            y2=min(img_h, bbox.y1 + tight_bottom),
        )
        if tight_box.x2 <= tight_box.x1 or tight_box.y2 <= tight_box.y1:
            tight_box = bbox

        ratio_str = config["main_ratio"]
        if parse_ratio(ratio_str) >= 1.0:
            ratio_str = "4:5"

        crop = build_crop_around_subject(
            subject_box=tight_box,
            img_w=img_w,
            img_h=img_h,
            ratio_str=ratio_str,
            margin_pct=max(2.0, float(config["margin_buffer"]) * 0.4),
        )

        crop_w = crop.width
        crop_h = crop.height
        head_x = (bbox.x1 + bbox.x2) / 2.0
        head_y = bbox.y1 + (height * 0.12)

        target_x1 = int(round(head_x - (crop_w / 2.0)))
        target_y1 = int(round(head_y - (crop_h / 3.0)))
        target = BoundingBox(
            x1=target_x1,
            y1=target_y1,
            x2=target_x1 + crop_w,
            y2=target_y1 + crop_h,
        )
        return shift_box_to_fit(target, img_w, img_h)

    def _cached_crop_for_pipeline(self, image_path: Path, config: dict) -> tuple[BoundingBox | None, str]:
        cache_entries = config.get("cached_cull_entries")
        if not isinstance(cache_entries, dict) or not cache_entries:
            return None, "cache unavailable"

        entry = self._find_cached_cull_entry(image_path, config)
        if entry is None:
            self.app.log(f"AI Crop cache: no cache entry for {image_path.name}; falling back to detection.")
            return None, "cache miss"

        bbox = self._bbox_from_cached_entry(image_path, entry)
        if bbox is None:
            return None, "cache invalid"

        img_w = self.app.current_image.width
        img_h = self.app.current_image.height

        scene_entry = entry.get("scene_classification", {})
        if not isinstance(scene_entry, dict):
            scene_entry = {}

        use_scene_classifier = bool(config.get("use_dance_scene_classifier", False))
        if use_scene_classifier:
            scene_type = str(scene_entry.get("scene_type", entry.get("scene_type", "unknown"))).strip().lower()

            keep_full_frame = LMStudioClient._to_bool(
                scene_entry.get("should_keep_full_frame", entry.get("should_keep_full_frame", False))
            )
            avoid_subject_crop = LMStudioClient._to_bool(
                scene_entry.get("should_avoid_subject_crop", entry.get("should_avoid_subject_crop", False))
            )
            if scene_type in LMStudioClient.COMPOSITION_PRESERVE_SCENE_TYPES or keep_full_frame or avoid_subject_crop:
                crop = BoundingBox(0, 0, img_w, img_h)
                self.app.log(f"AI Crop cache: {image_path.name} using full-frame composition-preserving mode.")
                return crop, "cache full-frame"
        else:
            cached_scene_type = str(scene_entry.get("scene_type", entry.get("scene_type", "unknown"))).strip().lower()
            if cached_scene_type in LMStudioClient.COMPOSITION_PRESERVE_SCENE_TYPES or LMStudioClient._to_bool(
                scene_entry.get("should_keep_full_frame", entry.get("should_keep_full_frame", False))
            ):
                self.app.log(
                    f"AI Crop cache: {image_path.name} has cached scene data "
                    f"(type={cached_scene_type}) but scene classification is disabled; "
                    "using subject-based crop instead."
                )

        near_edge = self._is_bbox_near_edge(bbox, img_w, img_h)

        if near_edge:
            crop = self._tight_edge_crop(bbox, img_w, img_h, config)
            self.app.log(f"AI Crop cache: {image_path.name} using edge-tight 3/4 crop mode.")
            return crop, "cache edge-tight"

        crop = build_crop_around_subject(
            subject_box=bbox,
            img_w=img_w,
            img_h=img_h,
            ratio_str=config["main_ratio"],
            margin_pct=config["margin_buffer"],
        )
        self.app.log(f"AI Crop cache: {image_path.name} using cached full-body crop mode.")
        return crop, "cache full-body"

    def _get_hybrid_detections_for_current_image(self, prompts: list[str]) -> list[Detection]:
        image_path = self.app.state.current_image_path
        if image_path is None or self.app.current_image is None:
            return []

        cache_key = self._cache_key(image_path, prompts, "hybrid_v2_focus")
        if cache_key in self.app.ai_detection_cache:
            self.app.log("AI Crop: using cached hybrid Florence detections.")
            return self.app.ai_detection_cache[cache_key]

        self.app.log("AI Crop: running Florence OD...")
        od_detections = run_florence_od_detection(self.app.current_image)
        self.app.log(f"AI Crop: OD returned {len(od_detections)} detections.")

        phrase_detections: list[Detection] = []
        for phrase in prompts:
            self.app.log(f'AI Crop phrase prompt: "{phrase}"')
            phrase_detections.extend(run_florence_phrase_detection(self.app.current_image, phrase))

        self.app.log(f"AI Crop: phrase grounding returned {len(phrase_detections)} raw detections.")

        merged = self._dedupe_detections(od_detections + phrase_detections)
        self.app.ai_detection_cache[cache_key] = merged
        self.app.log(f"AI Crop: merged to {len(merged)} unique detections.")
        return merged

    def _get_phrase_only_detections_for_current_image(self, prompts: list[str]) -> list[Detection]:
        image_path = self.app.state.current_image_path
        if image_path is None or self.app.current_image is None:
            return []

        phrase_only_prompts = self._unique_preserve_order(prompts)
        cache_key = self._cache_key(image_path, phrase_only_prompts, "phrase_only_v2_exact_prompts")

        if cache_key in self.app.ai_detection_cache:
            self.app.log("AI Crop: using cached phrase-only Florence detections.")
            return self.app.ai_detection_cache[cache_key]

        self.app.log("AI Crop: running phrase-only Florence detections...")
        phrase_detections: list[Detection] = []
        for phrase in phrase_only_prompts:
            self.app.log(f'AI Crop phrase prompt: "{phrase}"')
            phrase_detections.extend(run_florence_phrase_detection(self.app.current_image, phrase))

        self.app.log(f"AI Crop: phrase-only returned {len(phrase_detections)} raw detections.")

        merged = self._dedupe_detections(phrase_detections)
        self.app.ai_detection_cache[cache_key] = merged
        self.app.log(f"AI Crop: phrase-only merged to {len(merged)} unique detections.")
        return merged

    def evaluate_image_for_pipeline(self, image_path, config: dict) -> dict:
        previous_path = self.app.state.current_image_path
        previous_image = self.app.current_image
        previous_af = list(self.app.current_af_boxes)

        try:
            self.app.load_image(Path(image_path))

            cached_crop, cached_reason = self._cached_crop_for_pipeline(Path(image_path), config)
            if cached_crop is not None:
                return {
                    "path": Path(image_path),
                    "detections": [],
                    "hero": None,
                    "support_ball": None,
                    "crop": cached_crop,
                    "hero_reason": cached_reason,
                }

            prompts = config["prompts"]
            detection_mode = config["detection_mode"]

            if detection_mode == "Phrase Only":
                detections = self._get_phrase_only_detections_for_current_image(prompts)
            else:
                detections = self._get_hybrid_detections_for_current_image(prompts)

            people = [d for d in detections if self._is_person_label(d.label)]
            balls = [d for d in detections if self._is_ball_label(d.label)]

            hero, hero_reason = self._pick_hero_person(people)
            support_ball = self._pick_support_ball(hero, balls)

            chosen = []
            if hero is not None:
                chosen.append(hero)
            if support_ball is not None and support_ball not in chosen:
                chosen.append(support_ball)

            subject_union = union_boxes([d.bbox for d in chosen]) if chosen else None

            if subject_union is not None:
                crop = build_crop_around_subject(
                    subject_box=subject_union,
                    img_w=self.app.current_image.width,
                    img_h=self.app.current_image.height,
                    ratio_str=config["main_ratio"],
                    margin_pct=config["margin_buffer"],
                )
            else:
                crop = build_center_crop(
                    self.app.current_image.width,
                    self.app.current_image.height,
                    config["main_ratio"],
                    config["margin_buffer"],
                )

            return {
                "path": Path(image_path),
                "detections": detections,
                "hero": hero,
                "support_ball": support_ball,
                "crop": crop,
                "hero_reason": hero_reason,
            }
        finally:
            self.app.state.current_image_path = previous_path
            self.app.current_image = previous_image
            self.app.current_af_boxes = previous_af

    def on_image_changed(self):
        if self.app.current_image is None:
            self.app.set_manual_boxes([])
            self.app.set_overlays([])
            return

        profile = self.get_profile_data()
        prompts = [p.strip() for p in profile.prompts if p.strip()]
        detection_mode = self.detection_mode_var.get().strip() or "Hybrid"

        if detection_mode == "Phrase Only":
            detections = self._get_phrase_only_detections_for_current_image(prompts)
        else:
            detections = self._get_hybrid_detections_for_current_image(prompts)

        if not detections:
            crop = build_center_crop(
                self.app.current_image.width,
                self.app.current_image.height,
                profile.main_ratio,
                profile.margin_buffer,
            )
            self.app.set_manual_boxes([])
            self.app.set_manual_selected_ids(set())
            self.app.set_overlays([CropBox(name="AI_FallbackCrop", bbox=crop, color="#00FF00")])
            self.app.log("AI Crop: no detections, using fallback center crop.")
            return

        people = [d for d in detections if self._is_person_label(d.label)]
        balls = [d for d in detections if self._is_ball_label(d.label)]

        hero, hero_reason = self._pick_hero_person(people)
        support_ball = self._pick_support_ball(hero, balls)

        chosen: list[Detection] = []
        if hero is not None:
            chosen.append(hero)
        if support_ball is not None and support_ball not in chosen:
            chosen.append(support_ball)

        subject_union = union_boxes([d.bbox for d in chosen]) if chosen else None

        overlays: list[CropBox] = []

        if subject_union is not None:
            overlays.append(CropBox(name="AI_SubjectUnion", bbox=subject_union, color="#FFD400"))
            crop = build_crop_around_subject(
                subject_box=subject_union,
                img_w=self.app.current_image.width,
                img_h=self.app.current_image.height,
                ratio_str=profile.main_ratio,
                margin_pct=profile.margin_buffer,
            )
            overlays.append(CropBox(name="AI_PrimaryCrop", bbox=crop, color="#00FF00"))

            if support_ball is not None:
                self.app.log(f'{detection_mode}: {hero_reason} + support ball="{support_ball.label}"')
            else:
                self.app.log(f"{detection_mode}: {hero_reason}")
        else:
            crop = build_center_crop(
                self.app.current_image.width,
                self.app.current_image.height,
                profile.main_ratio,
                profile.margin_buffer,
            )
            overlays.append(CropBox(name="AI_FallbackCrop", bbox=crop, color="#00FF00"))
            self.app.log(f"{detection_mode}: no usable hero found, using fallback center crop.")

        self.app.set_manual_boxes(detections)
        self.app.set_manual_selected_ids({d.id for d in chosen})
        self.app.set_overlays(overlays)

    def rerun(self):
        self.on_image_changed()

    def _resolve_crop_source_folder(self) -> Path | None:
        if self.app.state.input_folder is None:
            return None
        source_folder = Path(self.app.state.input_folder)
        keep_folder = source_folder / "Output" / "Keep"
        if keep_folder.exists() and keep_folder.is_dir():
            return keep_folder
        return source_folder

    def _get_crop_output_dir(self) -> Path | None:
        crop_source = self._resolve_crop_source_folder()
        if crop_source is None:
            return None
        out_dir = crop_source / "Crops"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _save_current_crops_to_input_crop_folder(self) -> int:
        if self.app.current_image is None or self.app.state.current_image_path is None:
            return 0
        if not self.app.current_overlay_boxes:
            return 0

        output_dir = self._get_crop_output_dir()
        if output_dir is None:
            return 0

        base_name = self.app.state.current_image_path.stem
        saved = 0

        for crop in self.app.current_overlay_boxes:
            if crop.name.startswith("AF_"):
                continue
            out_path = output_dir / f"{base_name}_{crop.name}.jpg"
            self.app.image_repo.save_crop(self.app.current_image, crop.bbox, out_path)
            saved += 1

        return saved

    def stop_auto_crop(self):
        if not self.auto_running:
            return
        self.auto_cancel_requested = True
        self.app.log("AI Crop: stop requested...")

    def auto_crop_input_folder(self):
        if self.auto_running:
            self.app.log("AI Crop: auto crop already running.")
            return

        if self.app.state.input_folder is None:
            self.app.log("AI Crop: please select an input folder first.")
            return

        source_folder = Path(self.app.state.input_folder)
        crop_source = self._resolve_crop_source_folder()
        if crop_source is None:
            self.app.log("AI Crop: no crop source folder available.")
            return

        if source_folder != crop_source:
            self.app.set_input_folder(str(crop_source))
            self.app.start_batch()
        elif not self.app.state.image_paths:
            self.app.log("AI Crop: loading images from input folder...")
            self.app.start_batch()

        self.auto_images = [Path(p) for p in self.app.state.image_paths]
        if not self.auto_images:
            self.app.log("AI Crop: no images found in crop source folder.")
            return

        output_dir = self._get_crop_output_dir()
        if output_dir is None:
            self.app.log("AI Crop: could not create crop output folder.")
            return

        self.auto_index = 0
        self.auto_saved_total = 0
        self.auto_running = True
        self.auto_cancel_requested = False

        if self.auto_button is not None:
            self.auto_button.config(state="disabled")
        if self.stop_button is not None:
            self.stop_button.config(state="normal")

        self.app.log(f"AI Crop: starting auto crop on {len(self.auto_images)} image(s) from {crop_source}...")
        self.app.root.after(10, self._auto_crop_step)

    def _auto_crop_step(self):
        if self.auto_cancel_requested:
            self._finish_auto_crop(cancelled=True)
            return

        if self.auto_index >= len(self.auto_images):
            output_dir = self._get_crop_output_dir()
            self.app.log(
                f"AI Crop: complete. Processed {len(self.auto_images)} image(s), "
                f"saved {self.auto_saved_total} crop(s) to {output_dir}"
            )
            self._finish_auto_crop(cancelled=False)
            return

        image_path = self.auto_images[self.auto_index]
        self.app.state.current_index = self.auto_index
        self.app.load_current_image()

        try:
            self.on_image_changed()
            saved = self._save_current_crops_to_input_crop_folder()
            self.auto_saved_total += saved
            self.app.log(
                f"AI Crop Auto {self.auto_index + 1}/{len(self.auto_images)}: "
                f"{image_path.name}, saved {saved} crop(s)."
            )
        except Exception as exc:
            self.app.log(f"AI Crop: failed on {image_path.name}: {exc}")

        self.auto_index += 1
        self.app.root.after(1, self._auto_crop_step)

    def _finish_auto_crop(self, cancelled: bool):
        self.auto_running = False
        self.auto_cancel_requested = False

        if self.auto_button is not None:
            self.auto_button.config(state="normal")
        if self.stop_button is not None:
            self.stop_button.config(state="disabled")

        if cancelled:
            self.app.log("AI Crop: auto crop cancelled.")

    def approve(self):
        saved = self._save_current_crops_to_input_crop_folder()
        if saved == 0:
            self.app.log("AI Crop: no crops saved.")
        else:
            self.app.log(f"AI Crop: saved {saved} crop(s) to input-folder Crops.")
        self.app.next_image()