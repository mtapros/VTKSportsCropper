from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image


class PipelineTool:
    tool_id = "pipeline"
    display_name = "Full Trial Pipeline"

    def __init__(self, app):
        self.app = app
        self.panel = None

        self.source_folder_var = tk.StringVar()
        self.run_crop_var = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=False)

        self.is_running = False
        self.cancel_requested = False
        self.original_active_tool = None

        self.images: list[Path] = []
        self.results: list[dict] = []
        self.keeper_paths: list[Path] = []
        self.routed_count = 0
        self.bucket_counts = {"Keep": 0, "Maybe": 0, "Reject": 0, "Reject-Bursts": 0}
        self.crop_rows: list[list[str]] = []

        self.cull_config: dict = {}
        self.crop_config: dict = {}
        self.cached_cull_entries: dict[str, dict] = {}

        self.output_root: Path | None = None
        self.keep_dir: Path | None = None
        self.maybe_dir: Path | None = None
        self.reject_dir: Path | None = None
        self.reject_bursts_dir: Path | None = None
        self.crops_dir: Path | None = None
        self.reports_dir: Path | None = None

        self.phase = "idle"
        self.cull_index = 0
        self.review_index = 0
        self.crop_index = 0

        self.run_button = None
        self.stop_button = None

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")
        pad = {"padx": 10, "pady": 4}

        tk.Label(
            self.panel,
            text="Full Trial Pipeline",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", **pad)

        tk.Label(self.panel, text="Source Folder", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        row = tk.Frame(self.panel, bg="#2a2a2a")
        row.pack(fill="x", padx=10, pady=4)

        tk.Entry(row, textvariable=self.source_folder_var).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(row, text="Browse", command=self.browse_folder).pack(side=tk.LEFT, padx=(6, 0))

        tk.Checkbutton(
            self.panel,
            text="Run AI Crop on Output/Keep",
            variable=self.run_crop_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        tk.Checkbutton(
            self.panel,
            text="Overwrite Existing Outputs",
            variable=self.overwrite_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        self.run_button = tk.Button(self.panel, text="Run Full Trial", command=self.run_full_trial)
        self.run_button.pack(fill="x", padx=10, pady=(12, 4))

        self.stop_button = tk.Button(
            self.panel,
            text="Stop Pipeline",
            command=self.stop_pipeline,
            state="disabled",
            bg="#8b1e1e",
            fg="white",
        )
        self.stop_button.pack(fill="x", padx=10, pady=(0, 4))

        return self.panel

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_folder_var.set(folder)

    def stop_pipeline(self):
        if not self.is_running:
            return
        self.cancel_requested = True
        self.app.log("Pipeline: stop requested...")

    def _find_images(self, folder: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

    def _copy_to_bucket(self, src: Path, bucket_dir: Path, overwrite: bool) -> Path:
        bucket_dir.mkdir(parents=True, exist_ok=True)
        dst = bucket_dir / src.name
        if dst.exists() and not overwrite:
            return dst
        shutil.copy2(src, dst)
        return dst

    def _bucket_for_result(self, result: dict) -> tuple[str, Path]:
        if result.get("burst_suppressed", False):
            return "Reject-Bursts", self.reject_bursts_dir
        decision = str(result.get("decision", "Reject"))
        if decision == "Keep":
            return "Keep", self.keep_dir
        if decision == "Maybe":
            return "Maybe", self.maybe_dir
        return "Reject", self.reject_dir

    def _save_crop_image(self, src_path: Path, crop_box, crops_dir: Path, overwrite: bool):
        crops_dir.mkdir(parents=True, exist_ok=True)
        dst = crops_dir / src_path.name
        if dst.exists() and not overwrite:
            return dst

        with Image.open(src_path) as img:
            cropped = img.crop((crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2))
            cropped.save(dst)
        return dst

    def _show_image_in_tool(self, tool_id: str, image_path: Path, index: int):
        if self.app.state.active_tool_id != tool_id:
            self.app.set_active_tool(tool_id)

        self.app.state.image_paths = self.images
        self.app.state.current_index = index
        self.app.load_image(image_path)

        self.app.current_overlay_boxes = []
        self.app.current_manual_boxes = []
        self.app.ui.set_manual_boxes([])
        self.app.ui.set_manual_selected_ids(set())
        self.app.ui.set_overlay_boxes([])
        self.app.ui.show_image(self.app.current_image)
        self.app.ui.highlight_thumbnail_index(index)
        self.app._refresh_header_info()

        tool = self.app.tools[tool_id]
        if hasattr(tool, "on_image_changed"):
            tool.on_image_changed()

        self.app.root.update_idletasks()

    def _write_cull_report(self):
        report_csv = self.reports_dir / "cull_results.csv"
        with report_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "filename",
                    "decision",
                    "score",
                    "hero_focus",
                    "has_face",
                    "face_focus",
                    "burst_suppressed",
                    "burst_winner_paths",
                    "final_bucket",
                ]
            )
            for r in self.results:
                writer.writerow(
                    [
                        r["path"].name,
                        r["decision"],
                        f'{r["score"]:.2f}',
                        f'{r["hero_focus"]:.2f}',
                        r["has_face"],
                        f'{r["face_focus"]:.2f}',
                        bool(r.get("burst_suppressed", False)),
                        "; ".join(r.get("burst_winner_paths", [])),
                        str(r.get("final_bucket", "")),
                    ]
                )

    def _write_crop_report(self):
        crop_csv = self.reports_dir / "crop_results.csv"
        with crop_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "saved_crop", "hero_reason"])
            writer.writerows(self.crop_rows)

    def _load_cached_cull_entries_by_filename(self, source_folder: Path) -> dict[str, dict]:
        cache_path = source_folder / "VL_Debug" / "dance_cull_cache.json"
        if not cache_path.exists():
            self.app.log("Pipeline: dance cull cache not found; crop will use normal detection.")
            return {}

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.app.log(f"Pipeline: failed reading dance cull cache ({exc}); crop will use normal detection.")
            return {}

        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            self.app.log("Pipeline: dance cull cache has invalid entries object; crop will use normal detection.")
            return {}

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

        self.app.log(f"Pipeline: loaded {len(by_name)} cached cull entries from {cache_path}.")
        return by_name

    def run_full_trial(self):
        if self.is_running:
            self.app.log("Pipeline already running.")
            return

        source_folder = Path(self.source_folder_var.get().strip())
        if not source_folder.exists() or not source_folder.is_dir():
            messagebox.showerror("Invalid folder", "Please choose a valid source folder.")
            return

        ai_cull = self.app.tools_by_id.get("ai_cull")
        ai_crop = self.app.tools_by_id.get("ai_crop")

        if ai_cull is None or ai_crop is None:
            messagebox.showerror("Missing tools", "AI Cull and AI Crop must both be loaded.")
            return

        self.images = self._find_images(source_folder)
        if not self.images:
            messagebox.showinfo("No images", "No supported images found in the selected folder.")
            return

        self.app.state.input_folder = source_folder

        self.cull_config = ai_cull.get_runtime_config()
        burst_summary = ai_cull.prepare_burst_groups_for_paths(
            self.images,
            self.cull_config,
            source_folder=source_folder,
        )
        self.crop_config = ai_crop.get_runtime_config()
        self.cached_cull_entries = self._load_cached_cull_entries_by_filename(source_folder)
        self.crop_config["cached_cull_entries"] = self.cached_cull_entries
        self.crop_config["use_dance_scene_classifier"] = self.cull_config.get("use_dance_scene_classifier", False)

        self.output_root = source_folder / "Output"
        self.keep_dir = self.output_root / "Keep"
        self.maybe_dir = self.output_root / "Maybe"
        self.reject_dir = self.output_root / "Reject"
        self.reject_bursts_dir = self.output_root / "Reject-Bursts"
        self.crops_dir = self.keep_dir / "Crops"
        self.reports_dir = self.output_root
        for folder in [self.keep_dir, self.maybe_dir, self.reject_dir, self.reject_bursts_dir]:
            folder.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.keeper_paths = []
        self.routed_count = 0
        self.bucket_counts = {"Keep": 0, "Maybe": 0, "Reject": 0, "Reject-Bursts": 0}
        self.crop_rows = []

        self.phase = "cull"
        self.cull_index = 0
        self.review_index = 0
        self.crop_index = 0

        self.original_active_tool = self.app.state.active_tool_id
        self.is_running = True
        self.cancel_requested = False

        if self.run_button is not None:
            self.run_button.config(state="disabled")
        if self.stop_button is not None:
            self.stop_button.config(state="normal")

        self.app.state.output_folder = self.output_root
        self.app.state.image_paths = self.images
        self.app.state.current_index = 0
        self.app.ui.set_thumbnail_paths(self.images)
        self.app._refresh_header_info()

        self.app.log(f"Pipeline: starting full trial on {len(self.images)} images")
        self.app.log(
            f"Pipeline Burst Preflight: groups={burst_summary['group_count']} "
            f"burst-images={burst_summary['burst_images']} fps={burst_summary['fps']:.2f}"
        )
        self.app.log(f"Pipeline: cull config={self.cull_config}")
        self.app.log(f"Pipeline: crop config={self.crop_config}")

        self.app.root.after(10, self._step)

    def _step(self):
        if self.cancel_requested:
            self._finish_cancelled()
            return

        try:
            if self.phase == "cull":
                self._step_cull()
            elif self.phase == "review":
                self._step_review()
            elif self.phase == "crop":
                self._step_crop()
            elif self.phase == "done":
                self._finish_success()
        except Exception as exc:
            self._finish_error(exc)

    def _step_cull(self):
        ai_cull = self.app.tools_by_id["ai_cull"]

        if self.cull_index >= len(self.images):
            self.results = ai_cull.apply_burst_suppression_for_pipeline(self.results, self.cull_config)
            if hasattr(ai_cull, "persist_burst_suppression_results"):
                ai_cull.persist_burst_suppression_results(self.results)
            self._write_cull_report()
            self.app.log("Pipeline: cull pass complete")
            self.phase = "review"
            self.review_index = 0
            self.app.root.after(10, self._step)
            return

        image_path = self.images[self.cull_index]
        self._show_image_in_tool("ai_cull", image_path, self.cull_index)

        result = ai_cull.evaluate_image_for_pipeline(image_path, self.cull_config)
        self.results.append(result)

        self.app.log(
            f"Pipeline Cull {self.cull_index + 1}/{len(self.images)}: "
            f"{image_path.name} -> {result['decision']} score={result['score']:.1f}"
        )

        self.cull_index += 1
        self.app.root.after(1, self._step)

    def _step_review(self):
        if self.review_index >= len(self.results):
            if self.routed_count != len(self.images):
                raise RuntimeError(
                    f"Pipeline routing mismatch: routed={self.routed_count} expected={len(self.images)}"
                )
            if self.run_crop_var.get() and self.keeper_paths:
                self.phase = "crop"
                self.crop_index = 0
                self.app.log(f"Pipeline: starting crop pass on {len(self.keeper_paths)} keep image(s)")
            else:
                self.phase = "done"
            self.app.root.after(10, self._step)
            return

        r = self.results[self.review_index]
        image_path = Path(r["path"])
        display_index = self.review_index if self.review_index < len(self.images) else 0
        self._show_image_in_tool("ai_cull", image_path, display_index)

        bucket_name, bucket_dir = self._bucket_for_result(r)
        copied = self._copy_to_bucket(image_path, bucket_dir, self.overwrite_var.get())
        r["final_bucket"] = bucket_name
        self.routed_count += 1
        self.bucket_counts[bucket_name] = self.bucket_counts.get(bucket_name, 0) + 1

        if bucket_name == "Keep":
            self.keeper_paths.append(copied)

        self.app.log(
            f"Pipeline Route {self.review_index + 1}/{len(self.results)}: "
            f"{image_path.name} -> {bucket_name}"
        )

        self.review_index += 1
        self.app.root.after(1, self._step)

    def _step_crop(self):
        ai_crop = self.app.tools_by_id["ai_crop"]

        if self.crop_index >= len(self.keeper_paths):
            self._write_crop_report()
            self.phase = "done"
            self.app.root.after(10, self._step)
            return

        keeper_path = self.keeper_paths[self.crop_index]
        display_index = self.crop_index if self.crop_index < len(self.images) else 0
        self._show_image_in_tool("ai_crop", keeper_path, display_index)

        crop_result = ai_crop.evaluate_image_for_pipeline(keeper_path, self.crop_config)
        saved = self._save_crop_image(
            keeper_path,
            crop_result["crop"],
            self.crops_dir,
            self.overwrite_var.get(),
        )

        self.crop_rows.append([keeper_path.name, str(saved), crop_result["hero_reason"]])
        self.app.log(
            f"Pipeline Crop {self.crop_index + 1}/{len(self.keeper_paths)}: "
            f"{keeper_path.name} -> {saved.name}"
        )

        self.crop_index += 1
        self.app.root.after(1, self._step)

    def _finish_success(self):
        self.is_running = False
        self.cancel_requested = False

        if self.run_button is not None:
            self.run_button.config(state="normal")
        if self.stop_button is not None:
            self.stop_button.config(state="disabled")

        if self.original_active_tool in self.app.tools:
            self.app.set_active_tool(self.original_active_tool)

        counts = dict(self.bucket_counts)
        total_output = sum(counts.values())
        self.app.log(
            f"Pipeline: full trial complete. Keep={counts['Keep']} Maybe={counts['Maybe']} "
            f"Reject={counts['Reject']} Reject-Bursts={counts['Reject-Bursts']} total={total_output}"
        )

        messagebox.showinfo("Full Trial Complete", f"Processed {len(self.images)} images.")

    def _finish_cancelled(self):
        self.is_running = False
        self.cancel_requested = False

        if self.run_button is not None:
            self.run_button.config(state="normal")
        if self.stop_button is not None:
            self.stop_button.config(state="disabled")

        try:
            if self.reports_dir is not None:
                self._write_cull_report()
                if self.crop_rows:
                    self._write_crop_report()
        except Exception as exc:
            self.app.log(f"Pipeline: report write after cancel failed: {exc}")

        if self.original_active_tool in self.app.tools:
            self.app.set_active_tool(self.original_active_tool)

        self.app.log("Pipeline: cancelled by user.")

    def _finish_error(self, exc: Exception):
        self.is_running = False
        self.cancel_requested = False

        if self.run_button is not None:
            self.run_button.config(state="normal")
        if self.stop_button is not None:
            self.stop_button.config(state="disabled")

        self.app.log(f"Pipeline ERROR: {exc}")

        if self.original_active_tool in self.app.tools:
            self.app.set_active_tool(self.original_active_tool)

        messagebox.showerror("Pipeline Error", str(exc))