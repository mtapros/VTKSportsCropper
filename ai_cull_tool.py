from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ExifTags

from core import (
    compute_iou,
    get_focus_score,
    run_florence_od_detection,
    run_florence_phrase_detection,
)
from models import CropBox, Detection, SportProfile, BoundingBox


class AICullTool:
    tool_id = "ai_cull"
    display_name = "AI Cull Tool"

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

        self.current_score = 0.0
        self.current_decision = "Reject"
        self.current_hero_id: int | None = None
        self.current_ball_id: int | None = None

        self.auto_running = False
        self.auto_cancel_requested = False
        self.auto_images: list[Path] = []
        self.auto_results: list[dict] = []
        self.auto_index = 0
        self.auto_button = None
        self.stop_button = None

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

        return self.panel

    def apply_profile(self, profile: SportProfile):
        prompts = list(profile.prompts[:4])
        while len(prompts) < 4:
            prompts.append("")
        for i, var in enumerate(self.prompt_vars):
            var.set(prompts[i])

    def get_profile_data(self) -> SportProfile:
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
                1 if (item["prefer_face"] and item["has_face"]) else 0,
                item["score"],
                item["face_focus"],
                item["hero_focus"],
            )
        return sorted(items, key=key, reverse=True)

    def evaluate_image_for_pipeline(self, image_path, config: dict) -> dict:
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

        if self.app.current_image is None:
            self.app.set_manual_boxes([])
            self.app.set_manual_selected_ids(set())
            self.app.set_overlays([])
            return

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