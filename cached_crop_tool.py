from __future__ import annotations

import csv
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from core import build_crop_around_subject
from models import CropBox, SportProfile, BoundingBox


RATIO_COLORS = {
    "4:5": "#FFD400",  # yellow
    "5:7": "#FF3B30",  # red
    "2:3": "#2D7FF9",  # blue
}


class CachedCropTool:
    tool_id = "cached_crop"
    display_name = "Cached Crop Tool"

    def __init__(self, app):
        self.app = app
        self.panel = None

        self.keepers_folder_var = tk.StringVar()
        self.cache_json_var = tk.StringVar()
        self.overwrite_var = tk.BooleanVar(value=False)

        self.main_ratio_var = tk.StringVar(value="4:5")
        self.margin_var = tk.StringVar(value="12")

        self.is_running = False
        self.cancel_requested = False

        self.images: list[Path] = []
        self.cached_cull_entries: dict[str, dict] = {}
        self.crop_rows: list[list[str]] = []
        self.crop_index = 0

        self.crops_dir: Path | None = None
        self.reports_dir: Path | None = None
        self.original_active_tool = None

        self.commit_button = None
        self.run_all_button = None
        self.stop_button = None

        self.current_preview_crop = None
        self.current_preview_reason = "not ready"

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")
        pad = {"padx": 10, "pady": 4}

        tk.Label(
            self.panel,
            text="Cached Crop Tool",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", **pad)

        tk.Label(self.panel, text="Keepers Folder", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        keepers_row = tk.Frame(self.panel, bg="#2a2a2a")
        keepers_row.pack(fill="x", padx=10, pady=4)
        tk.Entry(keepers_row, textvariable=self.keepers_folder_var).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(keepers_row, text="Browse", command=self.browse_keepers_folder).pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(self.panel, text="Florence / VL Cache JSON", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        cache_row = tk.Frame(self.panel, bg="#2a2a2a")
        cache_row.pack(fill="x", padx=10, pady=4)
        tk.Entry(cache_row, textvariable=self.cache_json_var).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(cache_row, text="Browse", command=self.browse_cache_json).pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(
            self.panel,
            text="Crop Ratio",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(10, 4), padx=10)

        ratio_frame = tk.Frame(self.panel, bg="#2a2a2a")
        ratio_frame.pack(fill="x", padx=10, pady=4)

        for ratio in ["4:5", "5:7", "2:3"]:
            tk.Radiobutton(
                ratio_frame,
                text=ratio,
                value=ratio,
                variable=self.main_ratio_var,
                command=self.refresh_preview,
                bg="#2a2a2a",
                fg="white",
                selectcolor="#444",
                activebackground="#2a2a2a",
                activeforeground="white",
            ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(self.panel, text="Margin Buffer %", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        margin_entry = tk.Entry(self.panel, textvariable=self.margin_var)
        margin_entry.pack(fill="x", **pad)
        margin_entry.bind("<KeyRelease>", lambda e: self.refresh_preview())

        tk.Checkbutton(
            self.panel,
            text="Overwrite Existing Outputs",
            variable=self.overwrite_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        self.commit_button = tk.Button(
            self.panel,
            text="Commit Current",
            command=self.commit_current_and_next,
        )
        self.commit_button.pack(fill="x", padx=10, pady=(12, 4))

        self.run_all_button = tk.Button(
            self.panel,
            text="Run Continuous",
            command=self.run_cached_crop_batch,
        )
        self.run_all_button.pack(fill="x", padx=10, pady=(0, 4))

        self.stop_button = tk.Button(
            self.panel,
            text="Stop Cached Crop",
            command=self.stop_cached_crop,
            state="disabled",
            bg="#8b1e1e",
            fg="white",
        )
        self.stop_button.pack(fill="x", padx=10, pady=(0, 4))

        return self.panel

    def apply_profile(self, profile: SportProfile):
        self.margin_var.set(str(profile.margin_buffer))
        ratio = profile.main_ratio if profile.main_ratio in {"4:5", "5:7", "2:3"} else "4:5"
        self.main_ratio_var.set(ratio)

    def get_profile_data(self) -> SportProfile:
        profile_name = self.app.get_selected_profile_name() or "Cached Crop"
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

    def browse_keepers_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.keepers_folder_var.set(folder)
            self.app.set_input_folder(folder)
            self.images = list(self.app.state.image_paths)
            self.app.ui.set_thumbnail_paths(self.images)
            self.refresh_preview()

    def browse_cache_json(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.cache_json_var.set(path)
            self._load_cache_json_into_memory()
            if self.app.state.image_paths and self.app.current_image is None:
                self.app.load_current_image()
            self.refresh_preview()

    def on_image_changed(self):
        self.refresh_preview()

    def stop_cached_crop(self):
        if not self.is_running:
            return
        self.cancel_requested = True
        self.app.log("Cached Crop: stop requested...")

    def _find_images(self, folder: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

    def _load_cache_json_into_memory(self):
        self.cached_cull_entries = {}
        cache_json_raw = self.cache_json_var.get().strip()
        if not cache_json_raw:
            return

        cache_path = Path(cache_json_raw)
        if not cache_path.exists():
            self.app.log("Cached Crop: cache JSON not found.")
            return

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.app.log(f"Cached Crop: failed reading cache JSON ({exc}).")
            return

        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            self.app.log("Cached Crop: cache JSON has invalid entries object.")
            return

        by_name: dict[str, dict] = {}
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                continue

            image_name = ""
            image_path = entry.get("image_path")
            if isinstance(image_path, str) and image_path.strip():
                image_name = Path(image_path).name

            if not image_name:
                image_name = str(key).split("|", 1)[0]

            if not image_name:
                continue

            by_name[image_name.lower()] = entry

        self.cached_cull_entries = by_name
        self.app.log(f"Cached Crop: loaded {len(by_name)} cached entries from {cache_path}.")

    def _get_current_image_path(self) -> Path | None:
        if not self.app.state.image_paths:
            return None
        idx = self.app.state.current_index
        if idx < 0 or idx >= len(self.app.state.image_paths):
            return None
        return Path(self.app.state.image_paths[idx])

    def _get_cached_subject_bbox(self, image_path: Path):
        ai_crop = self.app.tools_by_id.get("ai_crop")
        if ai_crop is None:
            return None, "ai_crop unavailable"

        entry = self.cached_cull_entries.get(image_path.name.lower())
        if entry is None:
            return None, "cache miss"

        bbox = ai_crop._bbox_from_cached_entry(image_path, entry)
        if bbox is None:
            return None, "cache invalid"

        return bbox, "cache bbox"

    def _compute_selected_crop(self, image_path: Path):
        if self.app.current_image is None:
            return None, "no image"

        bbox, reason = self._get_cached_subject_bbox(image_path)
        if bbox is None:
            return None, reason

        ai_crop = self.app.tools_by_id.get("ai_crop")
        if ai_crop is None:
            return None, "ai_crop unavailable"

        ratio_str = self.main_ratio_var.get().strip() or "4:5"
        try:
            margin_pct = float(self.margin_var.get().strip() or "12")
        except Exception:
            margin_pct = 12.0

        img_w = self.app.current_image.width
        img_h = self.app.current_image.height

        config = {
            "main_ratio": ratio_str,
            "margin_buffer": margin_pct,
        }

        if ai_crop._is_bbox_near_edge(bbox, img_w, img_h):
            crop = ai_crop._tight_edge_crop(bbox, img_w, img_h, config)
            return crop, "cache edge-tight"

        crop = build_crop_around_subject(
            subject_box=bbox,
            img_w=img_w,
            img_h=img_h,
            ratio_str=ratio_str,
            margin_pct=margin_pct,
        )
        return crop, "cache full-body"

    def _ratio_to_float(self, ratio_str: str) -> float:
        left, right = ratio_str.split(":")
        return float(left) / float(right)

    def _inscribe_ratio_inside_box(self, outer_box: BoundingBox, target_ratio: str) -> BoundingBox:
        outer_w = outer_box.width
        outer_h = outer_box.height
        ratio = self._ratio_to_float(target_ratio)

        candidate_w = outer_w
        candidate_h = candidate_w / ratio

        if candidate_h > outer_h:
            candidate_h = outer_h
            candidate_w = candidate_h * ratio

        cx = (outer_box.x1 + outer_box.x2) / 2.0
        cy = (outer_box.y1 + outer_box.y2) / 2.0

        x1 = int(round(cx - candidate_w / 2.0))
        y1 = int(round(cy - candidate_h / 2.0))
        x2 = int(round(cx + candidate_w / 2.0))
        y2 = int(round(cy + candidate_h / 2.0))

        return BoundingBox(x1, y1, x2, y2)

    def _build_guides(self, selected_crop: BoundingBox):
        selected_ratio = self.main_ratio_var.get().strip() or "4:5"
        overlays: list[CropBox] = []

        overlays.append(
            CropBox(
                name="",
                bbox=selected_crop,
                color=RATIO_COLORS[selected_ratio],
            )
        )

        for ratio in ["4:5", "5:7", "2:3"]:
            if ratio == selected_ratio:
                continue
            inner = self._inscribe_ratio_inside_box(selected_crop, ratio)
            overlays.append(
                CropBox(
                    name="",
                    bbox=inner,
                    color=RATIO_COLORS[ratio],
                )
            )

        return overlays

    def refresh_preview(self):
        image_path = self._get_current_image_path()
        if image_path is None or self.app.current_image is None:
            self.app.set_overlays([])
            return

        crop_box, reason = self._compute_selected_crop(image_path)
        self.current_preview_crop = crop_box
        self.current_preview_reason = reason

        if crop_box is None:
            self.app.set_overlays([])
            return

        self.app.set_overlays(self._build_guides(crop_box))

    def _save_crop_image(self, src_path: Path, crop_box, crops_dir: Path, overwrite: bool) -> Path:
        crops_dir.mkdir(parents=True, exist_ok=True)
        dst = crops_dir / src_path.name
        if dst.exists() and not overwrite:
            return dst

        image = self.app.image_repo.load_image(src_path)
        self.app.image_repo.save_crop(image, crop_box, dst)
        return dst

    def crop_paths_with_cached_process(
        self,
        target_paths: list[Path],
        cached_cull_entries: dict[str, dict],
        crops_dir: Path,
        *,
        overwrite: bool = True,
        ratio_override: str | None = None,
        margin_override: float | None = None,
        log_prefix: str = "Cached Crop",
    ) -> int:
        if not target_paths:
            self.app.log(f"{log_prefix}: no images available to crop.")
            return 0

        self.cached_cull_entries = dict(cached_cull_entries or {})
        if ratio_override:
            self.main_ratio_var.set(str(ratio_override))
        if margin_override is not None:
            self.margin_var.set(str(margin_override))

        self.images = [Path(p) for p in target_paths]
        self.app.state.image_paths = list(self.images)
        self.app.ui.set_thumbnail_paths(self.images)

        saved_count = 0
        total = len(self.images)
        self.app.log(f"{log_prefix}: cropping {total} image(s) with cached process...")
        for idx, image_path in enumerate(self.images, start=1):
            self.app.state.current_index = idx - 1
            self.app.load_current_image()

            crop_box, reason = self._compute_selected_crop(image_path)
            if crop_box is None:
                self.app.log(f"{log_prefix} {idx}/{total}: {image_path.name} skipped ({reason})")
                continue

            self.app.set_overlays(self._build_guides(crop_box))
            saved = self._save_crop_image(image_path, crop_box, crops_dir, overwrite)
            saved_count += 1
            self.app.log(f"{log_prefix} {idx}/{total}: {image_path.name} -> {saved.name} ({reason})")
            try:
                self.app.root.update_idletasks()
                self.app.root.update()
            except Exception:
                pass

        self.app.log(f"{log_prefix}: saved {saved_count}/{total} crop(s) to {crops_dir}.")
        return saved_count

    def _write_crop_report(self):
        if self.reports_dir is None:
            return
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        crop_csv = self.reports_dir / "cached_crop_results.csv"
        with crop_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "saved_crop", "hero_reason"])
            writer.writerows(self.crop_rows)

    def _prepare_run(self):
        keepers_folder_raw = self.keepers_folder_var.get().strip()
        cache_json_raw = self.cache_json_var.get().strip()

        if not keepers_folder_raw:
            messagebox.showerror("Missing keepers folder", "Please choose a keepers folder.")
            return False

        if not cache_json_raw:
            messagebox.showerror("Missing cache JSON", "Please choose a Florence/VL cache JSON file.")
            return False

        keepers_folder = Path(keepers_folder_raw)
        if not keepers_folder.exists() or not keepers_folder.is_dir():
            messagebox.showerror("Invalid folder", "Please choose a valid keepers folder.")
            return False

        cache_json = Path(cache_json_raw)
        if not cache_json.exists() or not cache_json.is_file():
            messagebox.showerror("Invalid cache JSON", "Please choose a valid Florence/VL cache JSON file.")
            return False

        self.images = self._find_images(keepers_folder)
        if not self.images:
            messagebox.showinfo("No images", "No supported images found in the selected keepers folder.")
            return False

        self._load_cache_json_into_memory()

        output_root = keepers_folder / "Output"
        self.crops_dir = output_root / "Crops"
        self.reports_dir = output_root / "Reports"

        self.app.state.input_folder = keepers_folder
        self.app.state.output_folder = output_root
        self.app.state.image_paths = self.images
        self.app.ui.set_thumbnail_paths(self.images)

        if self.app.state.current_index >= len(self.images):
            self.app.state.current_index = 0

        if self.images:
            self.app.load_current_image()

        return True

    def commit_current_and_next(self):
        if not self._prepare_run():
            return

        image_path = self._get_current_image_path()
        if image_path is None:
            return

        crop_box, reason = self._compute_selected_crop(image_path)
        if crop_box is None:
            self.app.log(f"Cached Crop Commit: skipped {image_path.name} ({reason})")
            self.app.next_image()
            return

        saved = self._save_crop_image(
            image_path,
            crop_box,
            self.crops_dir,
            self.overwrite_var.get(),
        )
        self.app.log(f"Cached Crop Commit: {image_path.name} -> {saved.name} ({reason})")
        self.app.next_image()

    def run_cached_crop_batch(self):
        if self.is_running:
            self.app.log("Cached Crop: batch already running.")
            return

        if not self._prepare_run():
            return

        self.crop_rows = []
        self.crop_index = 0
        self.original_active_tool = self.app.state.active_tool_id
        self.is_running = True
        self.cancel_requested = False

        if self.commit_button is not None:
            self.commit_button.config(state="disabled")
        if self.run_all_button is not None:
            self.run_all_button.config(state="disabled")
        if self.stop_button is not None:
            self.stop_button.config(state="normal")

        self.app.log(f"Cached Crop: starting batch on {len(self.images)} image(s)")
        self.app.root.after(10, self._step)

    def _step(self):
        if self.cancel_requested:
            self._finish_cancelled()
            return

        if self.crop_index >= len(self.images):
            self._write_crop_report()
            self._finish_success()
            return

        image_path = self.images[self.crop_index]
        self.app.state.current_index = self.crop_index
        self.app.load_current_image()

        try:
            crop_box, reason = self._compute_selected_crop(image_path)
            if crop_box is None:
                self.crop_rows.append([image_path.name, "", reason])
                self.app.log(
                    f"Cached Crop {self.crop_index + 1}/{len(self.images)}: "
                    f"{image_path.name} skipped ({reason})"
                )
            else:
                saved = self._save_crop_image(
                    image_path,
                    crop_box,
                    self.crops_dir,
                    self.overwrite_var.get(),
                )
                self.crop_rows.append([image_path.name, str(saved), reason])
                self.app.log(
                    f"Cached Crop {self.crop_index + 1}/{len(self.images)}: "
                    f"{image_path.name} -> {saved.name} ({reason})"
                )
        except Exception as exc:
            self.app.log(f"Cached Crop: failed on {image_path.name}: {exc}")

        self.crop_index += 1
        self.app.root.after(1, self._step)

    def _finish_success(self):
        self.is_running = False
        self.cancel_requested = False

        if self.commit_button is not None:
            self.commit_button.config(state="normal")
        if self.run_all_button is not None:
            self.run_all_button.config(state="normal")
        if self.stop_button is not None:
            self.stop_button.config(state="disabled")

        if self.original_active_tool in self.app.tools:
            self.app.set_active_tool(self.original_active_tool)

        report_path = self.reports_dir / "cached_crop_results.csv" if self.reports_dir else "report unavailable"
        self.app.log(
            f"Cached Crop: complete. Processed {len(self.images)} image(s). "
            f"Saved report to {report_path}"
        )
        messagebox.showinfo("Cached Crop Complete", f"Processed {len(self.images)} images.")

    def _finish_cancelled(self):
        self.is_running = False
        self.cancel_requested = False

        if self.commit_button is not None:
            self.commit_button.config(state="normal")
        if self.run_all_button is not None:
            self.run_all_button.config(state="normal")
        if self.stop_button is not None:
            self.stop_button.config(state="disabled")

        try:
            self._write_crop_report()
        except Exception as exc:
            self.app.log(f"Cached Crop: report write after cancel failed: {exc}")

        if self.original_active_tool in self.app.tools:
            self.app.set_active_tool(self.original_active_tool)

        self.app.log("Cached Crop: cancelled by user.")

    def approve(self):
        self.commit_current_and_next()