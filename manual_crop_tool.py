from __future__ import annotations

import tkinter as tk

from core import build_center_crop, build_crop_around_subject, union_boxes
from models import CropBox, Detection, SportProfile


class ManualCropTool:
    tool_id = "manual_crop"
    display_name = "Manual Crop Tool"

    def __init__(self, app):
        self.app = app
        self.panel = None
        self.selected_ids: set[int] = set()

        self.margin_var = tk.StringVar(value="12")
        self.main_ratio_var = tk.StringVar(value="4:5")

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")
        pad = {"padx": 10, "pady": 4}

        tk.Label(self.panel, text="Manual Crop Settings", bg="#2a2a2a", fg="white", font=("Arial", 11, "bold")).pack(anchor="w", **pad)

        tk.Label(self.panel, text="Main Ratio", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.OptionMenu(self.panel, self.main_ratio_var, "4:5", "5:7", "2:3", "1:1", "16:9").pack(fill="x", **pad)

        tk.Label(self.panel, text="Margin Buffer %", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.margin_var).pack(fill="x", **pad)

        tk.Button(self.panel, text="Rebuild Crop From Selection", command=self.rebuild_from_selection).pack(fill="x", padx=10, pady=(10, 4))
        return self.panel

    def apply_profile(self, profile: SportProfile):
        self.margin_var.set(str(profile.margin_buffer))
        self.main_ratio_var.set(str(profile.main_ratio))

    def get_profile_data(self) -> SportProfile:
        profile_name = self.app.get_selected_profile_name() or "Manual Crop"
        return SportProfile(
            name=profile_name,
            prompts=[],
            focus_min=0.0,
            focus_relative=0.0,
            edge_margin=0,
            margin_buffer=float(self.margin_var.get().strip() or "12"),
            main_ratio=self.main_ratio_var.get().strip() or "4:5",
            auto_rotate=False,
            join_descriptors=False,
            safe_ratios={},
        )

    def on_image_changed(self):
        self.selected_ids.clear()
        self.app.set_manual_selected_ids(self.selected_ids)

        detections = self.app.current_overlay_boxes or []
        if not detections:
            self.app.set_overlays([])
            return

    def toggle_detection(self, detection_id: int):
        if detection_id in self.selected_ids:
            self.selected_ids.remove(detection_id)
        else:
            self.selected_ids.add(detection_id)
        self.app.set_manual_selected_ids(self.selected_ids)
        self.rebuild_from_selection()

    def rebuild_from_selection(self):
        if self.app.current_image is None:
            return

        detections: list[Detection] = self.app.current_manual_boxes or []
        selected = [d for d in detections if d.id in self.selected_ids]

        if not selected:
            crop = build_center_crop(
                self.app.current_image.width,
                self.app.current_image.height,
                self.main_ratio_var.get().strip() or "4:5",
                float(self.margin_var.get().strip() or "12"),
            )
            self.app.set_overlays([CropBox(name="Manual_PrimaryCrop", bbox=crop, color="#00FF00")])
            self.app.log("Manual Crop: no selection, using center crop.")
            return

        union = union_boxes([d.bbox for d in selected])
        crop = build_crop_around_subject(
            subject_box=union,
            img_w=self.app.current_image.width,
            img_h=self.app.current_image.height,
            ratio_str=self.main_ratio_var.get().strip() or "4:5",
            margin_pct=float(self.margin_var.get().strip() or "12"),
        )

        overlays = [
            CropBox(name="Manual_SubjectUnion", bbox=union, color="#FFD400"),
            CropBox(name="Manual_PrimaryCrop", bbox=crop, color="#00FF00"),
        ]
        self.app.set_overlays(overlays)
        self.app.log(f"Manual Crop: rebuilt from {len(selected)} selected detection(s).")

    def approve(self):
        self.app.save_current_overlays(prefer_name="Manual_PrimaryCrop")
        self.app.next_image()