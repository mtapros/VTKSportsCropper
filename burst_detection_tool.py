from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageOps, ImageTk

from burst_service import BurstAnalysis, analyze_bursts
from lmstudio_client import LMStudioClient


class BurstDetectionTool:
    tool_id = "burst_detection"
    display_name = "Burst Detection Tool"

    def __init__(self, app):
        self.app = app
        self.panel = None

        default_folder = str(app.state.input_folder) if app.state.input_folder else ""
        self.source_folder_var = tk.StringVar(value=default_folder)
        self.fps_threshold_var = tk.StringVar(value="8")
        self.keep_per_burst_var = tk.StringVar(value="1")
        self.summary_var = tk.StringVar(value="No burst analysis yet.")

        self.analysis: BurstAnalysis | None = None
        self.rows: list[dict] = []

        self.preview_frame = None
        self.preview_canvas = None
        self.preview_scrollbar = None
        self.preview_rows_frame = None
        self.preview_rows_window = None

        self.row_thumbnail_refs: dict[int, list[ImageTk.PhotoImage]] = {}
        self._active = False
        self._vl_running = False

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")
        pad = {"padx": 10, "pady": 4}

        tk.Button(self.panel, text="Select Source Folder", command=self._choose_source_folder).pack(fill="x", padx=10, pady=(6, 4))
        tk.Label(
            self.panel,
            text="Source Folder",
            bg="#2a2a2a",
            fg="white",
        ).pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.source_folder_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="FPS Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.fps_threshold_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="# Keep Per Burst", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.keep_per_burst_var).pack(fill="x", **pad)

        tk.Button(self.panel, text="Analyze Timestamps", command=self.analyze_timestamps).pack(fill="x", padx=10, pady=(8, 6))

        tk.Label(
            self.panel,
            textvariable=self.summary_var,
            bg="#2a2a2a",
            fg="#d3f2ff",
            justify="left",
            wraplength=320,
        ).pack(anchor="w", padx=10, pady=(4, 8))

        tk.Button(self.panel, text="Select All", command=self.select_all_rows).pack(fill="x", padx=10, pady=(0, 4))
        tk.Button(self.panel, text="Deselect All", command=self.deselect_all_rows).pack(fill="x", padx=10, pady=(0, 6))
        tk.Button(self.panel, text="Run Selected Rows Through VL", command=self.run_selected_rows_through_vl).pack(fill="x", padx=10, pady=(0, 8))

        return self.panel

    def on_activate(self):
        self._active = True
        self._ensure_preview_widget()
        self._refresh_summary()

    def on_deactivate(self):
        self._active = False
        self.app.ui.set_preview_widget(None)

    def apply_profile(self, profile):
        pass

    def on_image_changed(self):
        pass

    def _ensure_preview_widget(self):
        if self.preview_frame is None or not self.preview_frame.winfo_exists():
            self.preview_frame = tk.Frame(self.app.ui.preview_host, bg="#131313")

            header = tk.Frame(self.preview_frame, bg="#1f1f1f")
            header.pack(fill="x", padx=8, pady=(8, 6))
            tk.Label(header, text="Burst Rows", bg="#1f1f1f", fg="white", font=("Arial", 10, "bold")).pack(side="left")
            tk.Button(header, text="Select All", command=self.select_all_rows).pack(side="right", padx=(6, 0))
            tk.Button(header, text="Deselect All", command=self.deselect_all_rows).pack(side="right")

            rows_host = tk.Frame(self.preview_frame, bg="#131313")
            rows_host.pack(fill="both", expand=True, padx=8, pady=(0, 8))

            self.preview_canvas = tk.Canvas(rows_host, bg="#131313", highlightthickness=0)
            self.preview_scrollbar = tk.Scrollbar(rows_host, orient="vertical", command=self.preview_canvas.yview)
            self.preview_canvas.configure(yscrollcommand=self.preview_scrollbar.set)

            self.preview_scrollbar.pack(side="right", fill="y")
            self.preview_canvas.pack(side="left", fill="both", expand=True)

            self.preview_rows_frame = tk.Frame(self.preview_canvas, bg="#131313")
            self.preview_rows_window = self.preview_canvas.create_window((0, 0), window=self.preview_rows_frame, anchor="nw")
            self.preview_rows_frame.bind("<Configure>", self._on_preview_rows_configure)
            self.preview_canvas.bind("<Configure>", self._on_preview_canvas_configure)
            self.preview_canvas.bind("<MouseWheel>", self._on_preview_mousewheel)

        self.app.ui.set_preview_widget(self.preview_frame)
        self._render_rows()

    def _on_preview_rows_configure(self, event=None):
        if self.preview_canvas is not None:
            self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))

    def _on_preview_canvas_configure(self, event):
        if self.preview_canvas is not None and self.preview_rows_window is not None:
            self.preview_canvas.itemconfigure(self.preview_rows_window, width=event.width)

    def _on_preview_mousewheel(self, event):
        if self.preview_canvas is None:
            return
        self.preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _choose_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_folder_var.set(folder)

    def _parse_fps_threshold(self) -> float:
        try:
            return max(0.1, float(self.fps_threshold_var.get().strip() or "8"))
        except Exception:
            return 8.0

    def _parse_keep_per_burst(self) -> int:
        try:
            return max(1, int(self.keep_per_burst_var.get().strip() or "1"))
        except Exception:
            return 1

    def analyze_timestamps(self):
        source_folder = Path(self.source_folder_var.get().strip())
        if not source_folder.exists() or not source_folder.is_dir():
            self.app.log("Burst Detection: choose a valid source folder.")
            return

        fps_threshold = self._parse_fps_threshold()
        keep_per_burst = self._parse_keep_per_burst()
        self.app.log(
            f"Burst Detection: analyzing {source_folder} @ {fps_threshold:.2f} FPS (keep={keep_per_burst})."
        )

        self.analysis = analyze_bursts(source_folder, fps_threshold)
        self.rows = []
        for index, group in enumerate(self.analysis.burst_groups, start=1):
            self.rows.append(
                {
                    "index": index,
                    "paths": list(group),
                    "selected_var": tk.BooleanVar(value=True),
                    "expanded": False,
                    "winner": None,
                    "winner_source": None,
                    "row_frame": None,
                    "thumbs_frame": None,
                }
            )

        self._ensure_preview_widget()
        self._refresh_summary()
        self.app.log(
            f"Burst Detection: complete. images={self.analysis.total_images}, burst_groups={len(self.analysis.burst_groups)}."
        )

    def _refresh_summary(self):
        total_images = self.analysis.total_images if self.analysis else 0
        burst_groups = len(self.analysis.burst_groups) if self.analysis else 0
        burst_images = self.analysis.burst_images if self.analysis else 0
        non_burst_images = self.analysis.non_burst_images if self.analysis else 0
        selected_rows = sum(1 for row in self.rows if row["selected_var"].get())

        self.summary_var.set(
            f"Total images: {total_images}\n"
            f"Burst groups: {burst_groups}\n"
            f"Burst images: {burst_images}\n"
            f"Non-burst images: {non_burst_images}\n"
            f"Selected rows: {selected_rows}"
        )

    def _render_rows(self):
        if self.preview_rows_frame is None:
            return

        for child in self.preview_rows_frame.winfo_children():
            child.destroy()
        self.row_thumbnail_refs = {}

        if not self.rows:
            tk.Label(
                self.preview_rows_frame,
                text="Run Analyze Timestamps to show burst rows.",
                bg="#131313",
                fg="#b7c9d4",
                font=("Arial", 11),
            ).pack(anchor="w", padx=10, pady=10)
            self._on_preview_rows_configure()
            return

        for row in self.rows:
            row_frame = tk.Frame(self.preview_rows_frame, bg="#1a1a1a", bd=1, relief="solid", highlightbackground="#303030")
            row_frame.pack(fill="x", padx=6, pady=6)
            row["row_frame"] = row_frame

            controls = tk.Frame(row_frame, bg="#1a1a1a")
            controls.pack(fill="x", padx=6, pady=(6, 4))

            tk.Checkbutton(
                controls,
                variable=row["selected_var"],
                command=self._refresh_summary,
                bg="#1a1a1a",
                fg="white",
                selectcolor="#404040",
            ).pack(side="left")

            tk.Label(
                controls,
                text=f"Burst Group {row['index']} ({len(row['paths'])} frame(s))",
                bg="#1a1a1a",
                fg="white",
                font=("Arial", 10, "bold"),
            ).pack(side="left", padx=(6, 12))

            tk.Button(
                controls,
                text="Collapse" if row["expanded"] else "Expand",
                command=lambda current_row=row: self._toggle_row_expand(current_row),
            ).pack(side="right", padx=(6, 0))
            tk.Button(
                controls,
                text="Send Row To VL",
                command=lambda current_row=row: self.run_single_row_through_vl(current_row["index"] - 1),
            ).pack(side="right")

            thumbs_frame = tk.Frame(row_frame, bg="#181818")
            thumbs_frame.pack(fill="x", padx=6, pady=(0, 6))
            row["thumbs_frame"] = thumbs_frame

            self._render_row_thumbnails(row)

        self._on_preview_rows_configure()
        if self.preview_canvas is not None:
            self.preview_canvas.yview_moveto(0)

    def _toggle_row_expand(self, row: dict):
        row["expanded"] = not bool(row.get("expanded"))
        self._render_rows()

    def _render_row_thumbnails(self, row: dict):
        thumbs_frame = row.get("thumbs_frame")
        if thumbs_frame is None:
            return

        for child in thumbs_frame.winfo_children():
            child.destroy()

        thumb_refs: list[ImageTk.PhotoImage] = []
        thumb_side = 160 if row.get("expanded") else 96
        total = len(row["paths"])
        winner_path = row.get("winner")

        for idx, image_path in enumerate(row["paths"], start=1):
            is_winner = winner_path is not None and Path(winner_path).resolve() == Path(image_path).resolve()
            card_bg = "#0f2f0f" if is_winner else "#202020"
            card = tk.Frame(thumbs_frame, bg=card_bg, bd=2, relief="solid", highlightbackground="#2ecc71" if is_winner else "#444444")
            card.pack(side="left", padx=4, pady=4)

            preview = self._build_thumbnail(Path(image_path), thumb_side)
            image_label = tk.Label(card, image=preview, bg=card_bg, cursor="hand2")
            image_label.pack(padx=4, pady=(4, 2))
            image_label.bind("<Button-1>", lambda _event, row_index=row["index"] - 1, p=Path(image_path): self._manual_pick_winner(row_index, p))

            tk.Label(card, text=Path(image_path).name, bg=card_bg, fg="white", font=("Arial", 9)).pack(padx=4, pady=(0, 1))
            tk.Label(
                card,
                text=f"Burst candidate {idx}/{total}",
                bg=card_bg,
                fg="#cde8ff",
                font=("Arial", 8),
            ).pack(padx=4, pady=(0, 4))

            thumb_refs.append(preview)

        self.row_thumbnail_refs[row["index"] - 1] = thumb_refs

    def _build_thumbnail(self, image_path: Path, side: int) -> ImageTk.PhotoImage:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((side, side), Image.LANCZOS)
            canvas = Image.new("RGB", (side, side), (20, 20, 20))
            offset = ((side - img.width) // 2, (side - img.height) // 2)
            canvas.paste(img, offset)
        return ImageTk.PhotoImage(canvas)

    def _manual_pick_winner(self, row_index: int, image_path: Path):
        if not (0 <= row_index < len(self.rows)):
            return
        row = self.rows[row_index]
        row["winner"] = Path(image_path)
        row["winner_source"] = "manual"
        self._render_row_thumbnails(row)
        self.app.log(f"Burst Detection: manual winner for row {row_index + 1} -> {Path(image_path).name}")

    def select_all_rows(self):
        for row in self.rows:
            row["selected_var"].set(True)
        self._refresh_summary()

    def deselect_all_rows(self):
        for row in self.rows:
            row["selected_var"].set(False)
        self._refresh_summary()

    def run_single_row_through_vl(self, row_index: int):
        if not (0 <= row_index < len(self.rows)):
            return
        self._run_rows_with_vl([row_index])

    def run_selected_rows_through_vl(self):
        selected = [idx for idx, row in enumerate(self.rows) if row["selected_var"].get()]
        if not selected:
            self.app.log("Burst Detection: no rows selected for VL.")
            return
        self._run_rows_with_vl(selected)

    def _lmstudio_settings(self) -> tuple[str, str, float, float, int] | None:
        tool = self.app.tools_by_id.get("lmstudio")
        if tool is None:
            self.app.log("Burst Detection: LM Studio tool is unavailable.")
            return None

        base_url = tool.base_url_var.get().strip()
        model = tool.model_var.get().strip()
        if not base_url:
            self.app.log("Burst Detection: LM Studio base URL is empty.")
            return None
        if not model:
            self.app.log("Burst Detection: choose an LM Studio model first.")
            return None

        try:
            timeout = float(tool.timeout_var.get().strip() or "60")
        except Exception:
            timeout = 60.0

        try:
            temperature = float(tool.temperature_var.get().strip() or "0.2")
        except Exception:
            temperature = 0.2

        try:
            max_tokens = int(tool.max_tokens_var.get().strip() or "512")
        except Exception:
            max_tokens = 512

        return base_url, model, timeout, temperature, max_tokens

    def _run_rows_with_vl(self, row_indices: list[int]):
        if self._vl_running:
            self.app.log("Burst Detection: VL run already in progress.")
            return

        settings = self._lmstudio_settings()
        if settings is None:
            return

        self._vl_running = True
        base_url, model, timeout, temperature, max_tokens = settings
        row_indices = [idx for idx in row_indices if 0 <= idx < len(self.rows)]
        if not row_indices:
            self._vl_running = False
            return

        self.app.log(f"Burst Detection: running VL for {len(row_indices)} row(s).")

        def worker():
            try:
                client = LMStudioClient(base_url=base_url, timeout=timeout)
                for row_index in row_indices:
                    row = self.rows[row_index]
                    frame_paths = [Path(p) for p in row["paths"]]
                    self.app.root.after(0, lambda idx=row_index: self.app.log(f"Burst Detection: VL row {idx + 1} started."))
                    try:
                        response = client.burst_select_frames(
                            model=model,
                            image_paths=[str(p) for p in frame_paths],
                            temperature=min(max(0.0, temperature), 0.2),
                            max_tokens=max(256, min(max_tokens, 450)),
                        )
                    except Exception as exc:
                        self.app.root.after(
                            0,
                            lambda idx=row_index, err=exc: self.app.log(
                                f"Burst Detection: VL failed for row {idx + 1} ({err})."
                            ),
                        )
                        continue

                    best_frame = str(response.get("best_frame", "")).strip()
                    winner_path = self._match_best_frame(frame_paths, best_frame)
                    if winner_path is None:
                        self.app.root.after(
                            0,
                            lambda idx=row_index, best=best_frame: self.app.log(
                                f"Burst Detection: VL row {idx + 1} returned no valid winner ({best or 'missing best_frame'})."
                            ),
                        )
                        continue

                    self.app.root.after(0, lambda idx=row_index, wp=winner_path: self._apply_vl_winner(idx, wp))
            finally:
                self.app.root.after(0, self._finish_vl_run)

        threading.Thread(target=worker, daemon=True).start()

    def _match_best_frame(self, frame_paths: list[Path], best_frame: str) -> Path | None:
        if not best_frame:
            return None
        best_name = Path(best_frame).name.lower()
        for path in frame_paths:
            if path.name.lower() == best_name:
                return path
        return None

    def _apply_vl_winner(self, row_index: int, winner_path: Path):
        if not (0 <= row_index < len(self.rows)):
            return
        row = self.rows[row_index]
        row["winner"] = Path(winner_path)
        row["winner_source"] = "vl"
        self._render_row_thumbnails(row)
        self.app.log(f"Burst Detection: VL row {row_index + 1} winner -> {Path(winner_path).name}")

    def _finish_vl_run(self):
        self._vl_running = False
        self.app.log("Burst Detection: VL processing complete.")
