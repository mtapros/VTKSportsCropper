from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageDraw, ImageFont, ImageTk

from burst_service import (
    BurstAnalysis,
    BurstSettingsStore,
    BurstToolSettings,
    build_square_thumbnail,
    default_winner_criteria_text,
    group_adjacent_images,
    list_supported_images,
    load_rgb_image,
    normalize_winner_criteria_lines,
)
from lmstudio_client import LMStudioClient


_RENDER_BATCH_SIZE = 5
_RENDER_BATCH_DELAY_MS = 10
_LOG_PROGRESS_EVERY = 25
_THUMBNAIL_BATCH_SIZE = 10
_THUMBNAIL_BATCH_SLEEP_SEC = 0.03
_MAIN_THUMB_SIDE = 192
_REVIEW_THUMB_SIDE = 440
_MAIN_THUMB_CANVAS_EXTRA_HEIGHT = 92
_THUMBNAIL_PRIORITY_ROW_MULTIPLIER = 1000
_THUMBNAIL_PRIORITY_REVIEW_OFFSET = -100000
_PREVIEW_SCREEN_FRACTION = 0.92
_LIME = "#7CFF4D"
_LIME_RGB = (124, 255, 77)


class BurstDetectionTool:
    tool_id = "burst_detection"
    display_name = "Burst Detection Tool"

    VL_BURST_PANEL_LABELS = ("A", "B", "C", "D")
    VL_BURST_MAX_ROUND_IMAGES = 4
    VL_BURST_MAX_NEW_CHALLENGERS = 3
    VL_BURST_MAX_TEMPERATURE = 0.2
    VL_BURST_MIN_TOKENS = 256
    VL_BURST_MAX_TOKENS = 450

    def __init__(self, app):
        self.app = app
        self.panel = None
        self.settings_store = BurstSettingsStore(self.app.base_dir / "burst_detection_settings.json")

        default_folder = str(app.state.input_folder) if app.state.input_folder else ""
        self.source_folder_var = tk.StringVar(value=default_folder)
        self.fps_threshold_var = tk.StringVar(value="8")
        self.keep_per_burst_var = tk.StringVar(value="1")
        self.summary_var = tk.StringVar(value="No burst analysis yet.")

        self.criteria_text: ScrolledText | None = None
        self._winner_criteria_buffer = default_winner_criteria_text()

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
        self._analysis_running = False
        self._render_cancelled = False
        self._render_gen = 0
        self._analyze_btn: tk.Button | None = None
        self._profile_name = self.app.get_selected_profile_name() or "Generic Sport"

        self._thumbnail_cache: dict[tuple[str, int], Image.Image] = {}
        self._thumbnail_cache_lock = threading.Lock()
        self._thumbnail_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._thumbnail_job_seq = 0
        self._thumbnail_epoch = 0
        self._thumbnail_tokens: dict[str, tuple[int, int]] = {}
        self._thumbnail_worker = threading.Thread(target=self._thumbnail_worker_loop, daemon=True)
        self._thumbnail_worker.start()

        self._preview_windows: dict[tuple[int, str], dict] = {}

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")
        pad = {"padx": 10, "pady": 4}

        tk.Button(self.panel, text="Select Source Folder", command=self._choose_source_folder).pack(fill="x", padx=10, pady=(6, 4))
        tk.Label(self.panel, text="Source Folder", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.source_folder_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="FPS Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.fps_threshold_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="# Keep Per Burst", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.keep_per_burst_var).pack(fill="x", **pad)

        buttons = tk.Frame(self.panel, bg="#2a2a2a")
        buttons.pack(fill="x", padx=10, pady=(6, 2))
        tk.Button(buttons, text="Save Burst Setup", command=self.save_burst_setup).pack(side="left", expand=True, fill="x")
        tk.Button(buttons, text="Reload Burst Setup", command=self.reload_burst_setup).pack(side="left", expand=True, fill="x", padx=(6, 0))

        tk.Label(self.panel, text="Winner Criteria", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        self.criteria_text = ScrolledText(self.panel, height=7, bg="#161616", fg="white", insertbackground="white", wrap="word")
        self.criteria_text.pack(fill="both", padx=10, pady=(0, 8))
        self._set_winner_criteria_text(self._winner_criteria_buffer)
        self.criteria_text.bind("<KeyRelease>", lambda _event: self._sync_winner_criteria_buffer())

        self._analyze_btn = tk.Button(self.panel, text="Analyze Timestamps", command=self.analyze_timestamps)
        self._analyze_btn.pack(fill="x", padx=10, pady=(4, 6))

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
        self._sync_source_folder_from_app()
        self._ensure_preview_widget()
        self._refresh_summary()

    def on_deactivate(self):
        self._active = False
        self._render_cancelled = True
        self._thumbnail_epoch += 1
        self._close_aux_windows()
        self.app.ui.set_preview_widget(None)
        self.preview_frame = None
        self.preview_canvas = None
        self.preview_scrollbar = None
        self.preview_rows_frame = None
        self.preview_rows_window = None

    def apply_profile(self, profile):
        self._profile_name = getattr(profile, "name", "Generic Sport") or "Generic Sport"
        settings = self.settings_store.load_profile(self._profile_name)
        self.fps_threshold_var.set(str(settings.fps_threshold))
        self.keep_per_burst_var.set(str(settings.keep_per_burst))
        self._set_winner_criteria_text(settings.winner_criteria)
        self._sync_source_folder_from_app()

    def get_profile_data(self):
        self.save_burst_setup(log=False)
        current_profile = self.app.get_current_profile()
        profile_name = self.app.get_selected_profile_name() or current_profile.name
        return replace(current_profile, name=profile_name)

    def on_image_changed(self):
        pass

    def save_burst_setup(self, log: bool = True):
        profile_name = self.app.get_selected_profile_name() or self._profile_name or "Generic Sport"
        settings = BurstToolSettings(
            fps_threshold=self._parse_fps_threshold(),
            keep_per_burst=self._parse_keep_per_burst(),
            winner_criteria=self._read_winner_criteria_text(),
        )
        self.settings_store.save_profile(profile_name, settings)
        if log:
            self.app.log(f"Burst Detection: saved setup for profile '{profile_name}'.")

    def reload_burst_setup(self):
        self.apply_profile(self.app.get_current_profile())
        self.app.log(f"Burst Detection: loaded setup for profile '{self.app.get_selected_profile_name()}'.")

    def _sync_source_folder_from_app(self):
        if not self.source_folder_var.get().strip() and self.app.state.input_folder:
            self.source_folder_var.set(str(self.app.state.input_folder))

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
        self._start_incremental_render()

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

    def _read_winner_criteria_text(self) -> str:
        if self.criteria_text is not None and self.criteria_text.winfo_exists():
            self._winner_criteria_buffer = self.criteria_text.get("1.0", "end").strip() or default_winner_criteria_text()
        return self._winner_criteria_buffer

    def _set_winner_criteria_text(self, text: str):
        self._winner_criteria_buffer = str(text or "").strip() or default_winner_criteria_text()
        if self.criteria_text is not None and self.criteria_text.winfo_exists():
            self.criteria_text.delete("1.0", "end")
            self.criteria_text.insert("1.0", self._winner_criteria_buffer)

    def _sync_winner_criteria_buffer(self):
        self._winner_criteria_buffer = self._read_winner_criteria_text()

    def analyze_timestamps(self):
        if self._analysis_running:
            self.app.log("Burst Detection: analysis already running.")
            return

        source_folder = Path(self.source_folder_var.get().strip())
        if not source_folder.exists() or not source_folder.is_dir():
            self.app.log("Burst Detection: choose a valid source folder.")
            return

        self.save_burst_setup(log=False)
        fps_threshold = self._parse_fps_threshold()
        keep_per_burst = self._parse_keep_per_burst()

        self._analysis_running = True
        self._render_cancelled = True
        self._thumbnail_epoch += 1
        self._close_aux_windows()
        if self._analyze_btn is not None:
            self._analyze_btn.config(state="disabled")

        self.app.log(
            f"Burst Detection: starting analysis of {source_folder} "
            f"@ {fps_threshold:.2f} FPS (keep={keep_per_burst})."
        )

        def worker():
            try:
                ordered_paths = list_supported_images(source_folder)
                n_images = len(ordered_paths)
                self.app.root.after(
                    0,
                    lambda n=n_images: self.app.log(
                        f"Burst Detection: loaded {n} total image(s) from the folder."
                    ),
                )
                self.app.root.after(0, lambda: self.app.log("Burst Detection: analyzing burst groups..."))
                all_groups = group_adjacent_images(ordered_paths, fps_threshold)
                burst_groups = [g for g in all_groups if len(g) > 1]
                analysis = BurstAnalysis(
                    ordered_paths=ordered_paths,
                    all_groups=all_groups,
                    burst_groups=burst_groups,
                )
                self.app.root.after(0, lambda a=analysis: self._on_analysis_complete(a))
            except Exception as exc:
                self.app.root.after(0, lambda e=exc: self._on_analysis_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_analysis_complete(self, analysis: BurstAnalysis):
        self.analysis = analysis
        n_groups = len(analysis.burst_groups)
        self.app.log(f"Burst Detection: identified {n_groups} burst group(s).")

        self.rows = []
        for index, group in enumerate(analysis.burst_groups, start=1):
            row = {
                "index": index,
                "paths": list(group),
                "selected_var": tk.BooleanVar(value=True),
                "winner_paths": [],
                "winner_source": None,
                "row_frame": None,
                "thumbs_frame": None,
                "status_var": tk.StringVar(value="No winners selected."),
                "main_cards": {},
                "review_cards": {},
                "review_window": None,
                "review_inner": None,
                "review_vl_button": None,
                "review_status_label": None,
                "vl_meta": {},
            }
            self.rows.append(row)

        self._analysis_running = False
        if self._analyze_btn is not None:
            self._analyze_btn.config(state="normal")

        self._render_cancelled = False
        self._refresh_summary()
        self._ensure_preview_widget()

    def _on_analysis_error(self, exc: Exception):
        self._analysis_running = False
        if self._analyze_btn is not None:
            self._analyze_btn.config(state="normal")
        self.app.log(f"Burst Detection: analysis failed ({exc}).")

    def _refresh_summary(self):
        total_images = self.analysis.total_images if self.analysis else 0
        burst_groups = len(self.analysis.burst_groups) if self.analysis else 0
        burst_images = self.analysis.burst_images if self.analysis else 0
        non_burst_images = self.analysis.non_burst_images if self.analysis else 0
        selected_rows = sum(1 for row in self.rows if row["selected_var"].get())
        rows_with_winners = sum(1 for row in self.rows if row.get("winner_paths"))

        self.summary_var.set(
            f"Total images: {total_images}\n"
            f"Burst groups: {burst_groups}\n"
            f"Burst images: {burst_images}\n"
            f"Non-burst images: {non_burst_images}\n"
            f"Selected rows: {selected_rows}\n"
            f"Rows with winners: {rows_with_winners}"
        )

    def _start_incremental_render(self):
        if self.preview_rows_frame is None:
            return

        self._thumbnail_epoch += 1
        for child in self.preview_rows_frame.winfo_children():
            child.destroy()
        self.row_thumbnail_refs = {}

        for row in self.rows:
            row["row_frame"] = None
            row["thumbs_frame"] = None
            row["main_cards"] = {}

        self._render_cancelled = False

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

        self.app.log(f"Burst Detection: rendering preview for {len(self.rows)} burst group(s)...")
        self.app.log("Burst Detection: thumbnail placeholders ready; loading thumbnails in background...")
        self._render_gen += 1
        gen = self._render_gen
        self.app.root.after(0, lambda g=gen: self._render_batch(0, g))

    def _render_batch(self, start_index: int, gen: int):
        if self._render_cancelled or gen != self._render_gen:
            return
        if self.preview_rows_frame is None or not self.preview_rows_frame.winfo_exists():
            return

        end_index = min(start_index + _RENDER_BATCH_SIZE, len(self.rows))
        for i in range(start_index, end_index):
            self._render_single_row(self.rows[i])

        self._on_preview_rows_configure()

        if end_index < len(self.rows):
            if end_index % _LOG_PROGRESS_EVERY == 0:
                self.app.log(f"Burst Detection: rendered {end_index}/{len(self.rows)} rows...")
            self.app.root.after(_RENDER_BATCH_DELAY_MS, lambda: self._render_batch(end_index, gen))
        else:
            self.app.log(f"Burst Detection: preview complete ({len(self.rows)} row(s) rendered).")
            if self.preview_canvas is not None:
                self.preview_canvas.yview_moveto(0)

    def _render_single_row(self, row: dict):
        row_frame = tk.Frame(
            self.preview_rows_frame,
            bg="#1a1a1a",
            bd=1,
            relief="solid",
            highlightbackground="#303030",
        )
        row_frame.pack(fill="x", padx=6, pady=6)
        row["row_frame"] = row_frame

        controls = tk.Frame(row_frame, bg="#1a1a1a")
        controls.pack(fill="x", padx=6, pady=(6, 2))

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
            text="Expand",
            command=lambda current_row=row: self._open_row_review(current_row),
        ).pack(side="right", padx=(6, 0))

        tk.Button(
            controls,
            text="Send Row To VL",
            command=lambda current_row=row: self.run_single_row_through_vl(current_row["index"] - 1),
        ).pack(side="right")

        status = tk.Label(
            row_frame,
            textvariable=row["status_var"],
            bg="#1a1a1a",
            fg="#d7f6f8",
            anchor="w",
            justify="left",
        )
        status.pack(fill="x", padx=10, pady=(0, 4))

        thumbs_host = tk.Frame(row_frame, bg="#181818")
        thumbs_host.pack(fill="x", padx=6, pady=(0, 6))
        thumbs_canvas = tk.Canvas(
            thumbs_host,
            bg="#181818",
            highlightthickness=0,
            height=_MAIN_THUMB_SIDE + _MAIN_THUMB_CANVAS_EXTRA_HEIGHT,
        )
        thumbs_scroll = tk.Scrollbar(thumbs_host, orient="horizontal", command=thumbs_canvas.xview)
        thumbs_canvas.configure(xscrollcommand=thumbs_scroll.set)
        thumbs_canvas.pack(fill="x", expand=True)
        thumbs_scroll.pack(fill="x")

        thumbs_frame = tk.Frame(thumbs_canvas, bg="#181818")
        thumbs_canvas.create_window((0, 0), window=thumbs_frame, anchor="nw")
        thumbs_frame.bind("<Configure>", lambda _event, c=thumbs_canvas: c.configure(scrollregion=c.bbox("all")))

        row["thumbs_frame"] = thumbs_frame
        row["thumbs_canvas"] = thumbs_canvas
        row["thumbs_scroll"] = thumbs_scroll

        self._render_row_cards(row, thumbs_frame, _MAIN_THUMB_SIDE, context="main")
        self._update_row_status(row)

    def _render_row_cards(self, row: dict, host, side: int, context: str):
        for child in host.winfo_children():
            child.destroy()

        cards: dict[str, dict] = {}
        total = len(row["paths"])
        row_index = self._row_to_index(row)
        columns = 2 if context == "review" else 1

        for idx, image_path in enumerate(row["paths"], start=1):
            path = Path(image_path)
            path_key = self._path_key(path)

            card = tk.Frame(host, bg="#202020", bd=2, relief="solid", highlightbackground="#444444", highlightthickness=2)
            if context == "main":
                card.pack(side="left", padx=8, pady=6)
            else:
                grid_row = (idx - 1) // columns
                grid_col = (idx - 1) % columns
                card.grid(row=grid_row, column=grid_col, padx=8, pady=8, sticky="n")

            header = tk.Frame(card, bg="#202020")
            header.pack(fill="x", padx=4, pady=(4, 2))
            winner_button = tk.Button(
                header,
                text="☐",
                command=lambda ridx=row_index, p=path: self._toggle_manual_winner(ridx, p),
                bg="#404040",
                fg="white",
                activebackground="#555555",
                activeforeground="white",
                relief="flat",
                padx=8,
            )
            winner_button.pack(side="right")

            image_label = tk.Label(
                card,
                text="Loading...",
                width=max(10, side // 9),
                height=max(4, side // 20),
                bg="#202020",
                fg="#cde8ff",
                cursor="hand2",
            )
            image_label.pack(padx=4, pady=(2, 2))
            image_label.bind("<Button-1>", lambda _event, ridx=row_index, p=path: self._open_image_preview(ridx, p))

            name_label = tk.Label(card, text=path.name, bg="#202020", fg="white", font=("Arial", 9), wraplength=side + 20)
            name_label.pack(padx=4, pady=(0, 1))
            meta_label = tk.Label(
                card,
                text=f"Burst candidate {idx}/{total}",
                bg="#202020",
                fg="#cde8ff",
                font=("Arial", 8),
            )
            meta_label.pack(padx=4, pady=(0, 4))

            cards[path_key] = {
                "card": card,
                "header": header,
                "image_label": image_label,
                "winner_button": winner_button,
                "name_label": name_label,
                "meta_label": meta_label,
                "context": context,
                "side": side,
            }

            priority = (
                (row_index * _THUMBNAIL_PRIORITY_ROW_MULTIPLIER)
                + idx
                + (_THUMBNAIL_PRIORITY_REVIEW_OFFSET if context == "review" else 0)
            )
            self._queue_thumbnail_request(image_label, path, side, priority)

        if context == "main":
            row["main_cards"] = cards
        else:
            row["review_cards"] = cards
        self._refresh_row_views(row_index)

    def _queue_thumbnail_request(self, image_label: tk.Label, image_path: Path, side: int, priority: int):
        token = (self._thumbnail_epoch, self._thumbnail_job_seq + 1)
        self._thumbnail_job_seq += 1
        widget_key = str(image_label)
        self._thumbnail_tokens[widget_key] = token

        cache_key = (self._path_key(image_path), int(side))
        with self._thumbnail_cache_lock:
            cached = self._thumbnail_cache.get(cache_key)
        if cached is not None:
            self.app.root.after(0, lambda lbl=image_label, img=cached.copy(), tok=token: self._apply_thumbnail_to_label(lbl, img, tok))
            return

        self._thumbnail_queue.put(
            (
                priority,
                self._thumbnail_job_seq,
                {
                    "path": Path(image_path),
                    "side": int(side),
                    "cache_key": cache_key,
                    "callback": lambda img, tok=token, lbl=image_label: self._apply_thumbnail_to_label(lbl, img, tok),
                    "display_name": Path(image_path).name,
                },
            )
        )

    def _thumbnail_worker_loop(self):
        processed = 0
        while True:
            _, _, job = self._thumbnail_queue.get()
            path = Path(job["path"])
            side = int(job["side"])
            cache_key = job["cache_key"]
            callback = job["callback"]

            with self._thumbnail_cache_lock:
                cached = self._thumbnail_cache.get(cache_key)
            if cached is None:
                try:
                    image = build_square_thumbnail(path, side)
                except Exception as exc:
                    self._log_async(
                        f"Burst Detection: thumbnail load failed for {job.get('display_name', path.name)} ({exc})."
                    )
                    image = Image.new("RGB", (side, side), (38, 38, 38))
                with self._thumbnail_cache_lock:
                    self._thumbnail_cache[cache_key] = image
                cached = image

            self.app.root.after(0, lambda img=cached.copy(), cb=callback: cb(img))
            processed += 1
            if processed % _THUMBNAIL_BATCH_SIZE == 0:
                time.sleep(_THUMBNAIL_BATCH_SLEEP_SEC)

    def _apply_thumbnail_to_label(self, image_label: tk.Label, pil_image: Image.Image, token):
        try:
            if not image_label.winfo_exists():
                return
        except Exception:
            return
        if self._thumbnail_tokens.get(str(image_label)) != token:
            return
        photo = ImageTk.PhotoImage(pil_image)
        image_label.configure(image=photo, text="")
        image_label.image = photo

    def _path_key(self, image_path: Path | str) -> str:
        return str(Path(image_path).resolve())

    def _selected_keys_for_row(self, row: dict) -> set[str]:
        return {self._path_key(path) for path in row.get("winner_paths", [])}

    def _toggle_manual_winner(self, row_index: int, image_path: Path):
        if not (0 <= row_index < len(self.rows)):
            return
        row = self.rows[row_index]
        keep_limit = self._parse_keep_per_burst()
        current = [Path(p) for p in row.get("winner_paths", [])]
        current_keys = {self._path_key(p) for p in current}
        target_key = self._path_key(image_path)

        if target_key in current_keys:
            new_winners = [p for p in current if self._path_key(p) != target_key]
        elif keep_limit <= 1:
            new_winners = [Path(image_path)]
        else:
            if len(current) >= keep_limit:
                self.app.log(
                    f"Burst Detection: row {row_index + 1} already has {keep_limit} winner(s); deselect one first."
                )
                return
            new_winners = current + [Path(image_path)]

        self._set_row_winners(row_index, new_winners, source="manual")
        winner_names = ", ".join(Path(p).name for p in new_winners) or "none"
        self.app.log(f"Burst Detection: manual winner(s) for row {row_index + 1} -> {winner_names}")

    def _row_to_index(self, row: dict) -> int:
        return int(row.get("index", 1)) - 1

    def _set_row_winners(self, row_index: int, winner_paths: list[Path], source: str | None, vl_meta: dict | None = None):
        if not (0 <= row_index < len(self.rows)):
            return
        row = self.rows[row_index]
        keep_limit = self._parse_keep_per_burst()
        normalized: list[Path] = []
        seen: set[str] = set()
        row_paths = {self._path_key(p): Path(p) for p in row["paths"]}
        for path in winner_paths:
            key = self._path_key(path)
            if key in seen or key not in row_paths:
                continue
            seen.add(key)
            normalized.append(row_paths[key])
            if len(normalized) >= keep_limit:
                break

        row["winner_paths"] = normalized
        row["winner_source"] = source if normalized else None
        if vl_meta is not None:
            row["vl_meta"] = dict(vl_meta)
        self._update_row_status(row)
        self._refresh_row_views(row_index)
        self._refresh_summary()

    def _update_row_status(self, row: dict):
        winners = [Path(p).name for p in row.get("winner_paths", [])]
        source = row.get("winner_source") or "manual"
        if not winners:
            text = "No winners selected."
        elif len(winners) == 1:
            text = f"Winner ({source}): {winners[0]}"
        else:
            text = f"Winners ({source}): {', '.join(winners)}"
        row["status_var"].set(text)
        status_label = row.get("review_status_label")
        if status_label is not None and status_label.winfo_exists():
            status_label.configure(text=text)

    def _refresh_row_views(self, row_index: int):
        if not (0 <= row_index < len(self.rows)):
            return
        row = self.rows[row_index]
        selected_keys = self._selected_keys_for_row(row)

        for cards in (row.get("main_cards", {}), row.get("review_cards", {})):
            for path_key, card_info in cards.items():
                self._set_card_state(card_info, path_key in selected_keys)

        for key, info in list(self._preview_windows.items()):
            if info.get("row_index") == row_index:
                self._refresh_preview_window_state(key)

    def _set_card_state(self, card_info: dict, is_selected: bool):
        bg = "#1f3320" if is_selected else "#202020"
        border = _LIME if is_selected else "#444444"
        button_bg = _LIME if is_selected else "#404040"
        button_fg = "#101010" if is_selected else "white"
        button_text = "✓" if is_selected else "☐"

        for widget_name in ("card", "header", "image_label", "name_label", "meta_label"):
            widget = card_info.get(widget_name)
            if widget is not None and widget.winfo_exists():
                widget.configure(bg=bg)
        card = card_info.get("card")
        if card is not None and card.winfo_exists():
            card.configure(highlightbackground=border, highlightcolor=border)
        button = card_info.get("winner_button")
        if button is not None and button.winfo_exists():
            button.configure(text=button_text, bg=button_bg, fg=button_fg, activebackground=button_bg, activeforeground=button_fg)

    def select_all_rows(self):
        for row in self.rows:
            row["selected_var"].set(True)
        self._refresh_summary()

    def deselect_all_rows(self):
        for row in self.rows:
            row["selected_var"].set(False)
        self._refresh_summary()

    def _open_row_review(self, row: dict):
        existing = row.get("review_window")
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return

        window = tk.Toplevel(self.app.root)
        window.title(f"Burst Review - Row {row['index']}")
        sw = max(900, int(window.winfo_screenwidth() * 0.9))
        sh = max(650, int(window.winfo_screenheight() * 0.85))
        x = max(20, (window.winfo_screenwidth() - sw) // 2)
        y = max(20, (window.winfo_screenheight() - sh) // 2)
        window.geometry(f"{sw}x{sh}+{x}+{y}")
        window.configure(bg="#101010")

        header = tk.Frame(window, bg="#171717")
        header.pack(fill="x")
        tk.Label(
            header,
            text=f"Burst Group {row['index']} Review ({len(row['paths'])} frame(s))",
            bg="#171717",
            fg="white",
            font=("Arial", 12, "bold"),
        ).pack(side="left", padx=12, pady=10)
        review_row_index = self._row_to_index(row)
        review_vl_button = tk.Button(
            header,
            text="Run VL For Row",
            command=lambda ridx=review_row_index: self.run_single_row_through_vl(ridx),
        )
        review_vl_button.pack(side="right", padx=(6, 6))
        status_label = tk.Label(header, text=row["status_var"].get(), bg="#171717", fg="#d7f6f8")
        status_label.pack(side="right", padx=12)
        row["review_vl_button"] = review_vl_button
        row["review_status_label"] = status_label

        body = tk.Frame(window, bg="#101010")
        body.pack(fill="both", expand=True, padx=8, pady=8)

        canvas = tk.Canvas(body, bg="#101010", highlightthickness=0)
        scrollbar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg="#101010")
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(canvas_window, width=event.width))

        row["review_window"] = window
        row["review_inner"] = inner
        row["review_cards"] = {}
        self._render_row_cards(row, inner, _REVIEW_THUMB_SIDE, context="review")

        def close_review():
            row["review_cards"] = {}
            row["review_inner"] = None
            row["review_window"] = None
            row["review_vl_button"] = None
            row["review_status_label"] = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_review)

    def _open_image_preview(self, row_index: int, image_path: Path):
        if not (0 <= row_index < len(self.rows)):
            return
        path = Path(image_path)
        key = (row_index, self._path_key(path))
        existing = self._preview_windows.get(key)
        if existing and existing["window"].winfo_exists():
            existing["window"].deiconify()
            existing["window"].lift()
            existing["window"].focus_force()
            return

        window = tk.Toplevel(self.app.root)
        window.title(path.name)
        sw = max(900, int(window.winfo_screenwidth() * _PREVIEW_SCREEN_FRACTION))
        sh = max(700, int(window.winfo_screenheight() * _PREVIEW_SCREEN_FRACTION))
        x = max(20, (window.winfo_screenwidth() - sw) // 2)
        y = max(20, (window.winfo_screenheight() - sh) // 2)
        window.geometry(f"{sw}x{sh}+{x}+{y}")
        window.configure(bg="#111111")

        header = tk.Frame(window, bg="#161616")
        header.pack(fill="x")
        tk.Label(header, text=path.name, bg="#161616", fg="white", font=("Arial", 12, "bold")).pack(side="left", padx=12, pady=10)
        state_label = tk.Label(header, text="Loading preview...", bg="#161616", fg="#d7f6f8")
        state_label.pack(side="right", padx=12)

        image_host = tk.Frame(window, bg="#111111")
        image_host.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        image_label = tk.Label(image_host, text="Loading preview...", bg="#111111", fg="#d7f6f8")
        image_label.pack(fill="both", expand=True)

        controls = tk.Frame(window, bg="#161616")
        controls.pack(fill="x")
        set_button = tk.Button(
            controls,
            text="Set as Winner",
            command=lambda ridx=row_index, p=path: self._set_preview_winner(ridx, p),
            bg=_LIME,
            fg="#101010",
        )
        set_button.pack(side="right", padx=(6, 12), pady=10)
        close_button = tk.Button(controls, text="Close")
        close_button.pack(side="right", pady=10)

        info = {
            "window": window,
            "image_host": image_host,
            "image_label": image_label,
            "state_label": state_label,
            "winner_button": set_button,
            "row_index": row_index,
            "path": path,
            "loaded_image": None,
            "photo": None,
            "render_after": None,
        }
        self._preview_windows[key] = info
        self._refresh_preview_window_state(key)

        def close_preview():
            self._preview_windows.pop(key, None)
            window.destroy()

        close_button.configure(command=close_preview)
        window.protocol("WM_DELETE_WINDOW", close_preview)
        image_host.bind("<Configure>", lambda _event, preview_key=key: self._schedule_preview_render(preview_key))

        def worker():
            try:
                image = load_rgb_image(path)
            except Exception as exc:
                self.app.root.after(0, lambda err=exc: self._preview_load_failed(key, err))
                return
            self.app.root.after(0, lambda img=image: self._on_preview_image_loaded(key, img))

        threading.Thread(target=worker, daemon=True).start()

    def _set_preview_winner(self, row_index: int, image_path: Path):
        if not (0 <= row_index < len(self.rows)):
            return
        row = self.rows[row_index]
        keep_limit = self._parse_keep_per_burst()
        current = [Path(p) for p in row.get("winner_paths", [])]
        current_keys = {self._path_key(p) for p in current}
        target_key = self._path_key(image_path)

        if target_key in current_keys:
            new_winners = current
        elif keep_limit <= 1:
            new_winners = [Path(image_path)]
        else:
            if len(current) >= keep_limit:
                self.app.log(
                    f"Burst Detection: row {row_index + 1} already has {keep_limit} winner(s); deselect one first."
                )
                return
            new_winners = current + [Path(image_path)]

        self._set_row_winners(row_index, new_winners, source="manual")
        self.app.log(
            f"Burst Detection: preview winner(s) for row {row_index + 1} -> "
            f"{', '.join(Path(p).name for p in new_winners)}"
        )

    def _preview_load_failed(self, key: tuple[int, str], exc: Exception):
        info = self._preview_windows.get(key)
        if not info:
            return
        window = info["window"]
        if not window.winfo_exists():
            return
        info["image_label"].configure(text=f"Failed to load preview: {exc}")
        info["state_label"].configure(text="Preview unavailable", fg="#ff8a8a")

    def _on_preview_image_loaded(self, key: tuple[int, str], image: Image.Image):
        info = self._preview_windows.get(key)
        if not info:
            return
        if not info["window"].winfo_exists():
            return
        info["loaded_image"] = image
        self._refresh_preview_window_state(key)
        self._schedule_preview_render(key, delay=10)

    def _schedule_preview_render(self, key: tuple[int, str], delay: int = 60):
        info = self._preview_windows.get(key)
        if not info:
            return
        after_id = info.get("render_after")
        if after_id is not None:
            try:
                info["window"].after_cancel(after_id)
            except Exception:
                pass
        info["render_after"] = info["window"].after(delay, lambda preview_key=key: self._render_preview_image(preview_key))

    def _render_preview_image(self, key: tuple[int, str]):
        info = self._preview_windows.get(key)
        if not info:
            return
        window = info["window"]
        if not window.winfo_exists():
            return
        image = info.get("loaded_image")
        if image is None:
            return
        host = info["image_host"]
        max_w = max(100, host.winfo_width() - 20)
        max_h = max(100, host.winfo_height() - 20)
        if max_w <= 0 or max_h <= 0:
            return

        resized = image.copy()
        resized.thumbnail((max_w, max_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        label = info["image_label"]
        label.configure(image=photo, text="")
        label.image = photo
        info["photo"] = photo
        info["render_after"] = None

    def _refresh_preview_window_state(self, key: tuple[int, str]):
        info = self._preview_windows.get(key)
        if not info:
            return
        row_index = info["row_index"]
        if not (0 <= row_index < len(self.rows)):
            return
        row = self.rows[row_index]
        selected = self._path_key(info["path"]) in self._selected_keys_for_row(row)
        state_label = info["state_label"]
        if selected:
            state_label.configure(text="Winner selected", fg=_LIME)
            info["winner_button"].configure(text="Set as Winner", bg=_LIME, fg="#101010")
        else:
            source = row.get("winner_source")
            text = f"Current selection source: {source}" if source else "Not selected"
            state_label.configure(text=text, fg="#d7f6f8")
            info["winner_button"].configure(text="Set as Winner", bg=_LIME, fg="#101010")

    def _close_aux_windows(self):
        for key, info in list(self._preview_windows.items()):
            window = info.get("window")
            if window is not None and window.winfo_exists():
                window.destroy()
            self._preview_windows.pop(key, None)

        for row in self.rows:
            review_window = row.get("review_window")
            if review_window is not None and review_window.winfo_exists():
                review_window.destroy()
            row["review_window"] = None
            row["review_inner"] = None
            row["review_cards"] = {}
            row["review_vl_button"] = None
            row["review_status_label"] = None

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
        keep_per_burst = self._parse_keep_per_burst()
        criteria_lines = normalize_winner_criteria_lines(self._read_winner_criteria_text(), include_defaults=True)
        row_indices = [idx for idx in row_indices if 0 <= idx < len(self.rows)]
        if not row_indices:
            self._vl_running = False
            return

        self.app.log(f"Burst Detection: running VL tournament for {len(row_indices)} row(s).")

        def worker():
            try:
                client = LMStudioClient(base_url=base_url, timeout=timeout)
                for row_index in row_indices:
                    row = self.rows[row_index]
                    frame_paths = [Path(p) for p in row["paths"]]
                    self._log_async(
                        f"Burst Detection: VL row {row_index + 1} started with {len(frame_paths)} candidate(s): "
                        f"{', '.join(path.name for path in frame_paths)}"
                    )
                    try:
                        winners, meta = self._select_burst_winners_with_vl(
                            client=client,
                            model=model,
                            frame_paths=frame_paths,
                            keep_per_burst=keep_per_burst,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            criteria_lines=criteria_lines,
                            row_index=row_index,
                        )
                    except Exception as exc:
                        self._log_async(f"Burst Detection: VL failed for row {row_index + 1} ({exc}).")
                        continue

                    if not winners:
                        self._log_async(f"Burst Detection: VL row {row_index + 1} returned no valid winner.")
                        continue

                    self.app.root.after(0, lambda idx=row_index, wp=winners, meta=meta: self._apply_vl_winners(idx, wp, meta))
            finally:
                self.app.root.after(0, self._finish_vl_run)

        threading.Thread(target=worker, daemon=True).start()

    def _log_async(self, message: str):
        self.app.root.after(0, lambda m=message: self.app.log(m))

    def _select_burst_winners_with_vl(
        self,
        client: LMStudioClient,
        model: str,
        frame_paths: list[Path],
        keep_per_burst: int,
        temperature: float,
        max_tokens: int,
        criteria_lines: list[str],
        row_index: int,
    ) -> tuple[list[Path], dict]:
        remaining = list(frame_paths)
        chosen: list[Path] = []
        rounds: list[dict] = []
        round_index = 1
        keep_target = min(max(1, int(keep_per_burst)), len(remaining))

        while remaining and len(chosen) < keep_target:
            if len(remaining) == 1:
                winner = remaining.pop(0)
                chosen.append(winner)
                self._log_async(f"Burst Detection: VL row {row_index + 1} final carry-over winner -> {winner.name}")
                continue

            winner, round_meta, round_index = self._select_single_burst_winner_via_tournament(
                client=client,
                model=model,
                items=remaining,
                start_round=round_index,
                temperature=temperature,
                max_tokens=max_tokens,
                criteria_lines=criteria_lines,
                row_index=row_index,
            )
            if winner is None:
                raise ValueError("VL burst tournament returned no winner.")
            chosen.append(winner)
            rounds.extend(round_meta)
            winner_key = self._path_key(winner)
            remaining = [item for item in remaining if self._path_key(item) != winner_key]

        selected_paths = [str(path) for path in chosen]
        rejected_paths = [str(path) for path in frame_paths if self._path_key(path) not in {self._path_key(p) for p in chosen}]
        round_confidences = [float(r.get("confidence", 0.0)) for r in rounds if isinstance(r, dict)]
        avg_confidence = (sum(round_confidences) / len(round_confidences)) if round_confidences else 0.0

        meta = {
            "best_frame": selected_paths[0] if selected_paths else "",
            "alternates": selected_paths[1:],
            "rejects": rejected_paths,
            "reason": "VL tournament burst selection.",
            "selection_mode": "vl_tournament_grid",
            "selected_frames": selected_paths,
            "rounds": rounds,
            "confidence": max(0.0, min(1.0, avg_confidence)),
            "criteria": criteria_lines,
        }
        self._log_async(
            f"Burst Detection: VL row {row_index + 1} final winner(s): {', '.join(Path(p).name for p in selected_paths)}"
        )
        return chosen, meta

    def _select_single_burst_winner_via_tournament(
        self,
        client: LMStudioClient,
        model: str,
        items: list[Path],
        start_round: int,
        temperature: float,
        max_tokens: int,
        criteria_lines: list[str],
        row_index: int,
    ) -> tuple[Path | None, list[dict], int]:
        if not items:
            return None, [], start_round
        if len(items) == 1:
            return items[0], [], start_round

        round_index = start_round
        round_meta: list[dict] = []
        cursor = 0
        chunk = items[: min(self.VL_BURST_MAX_ROUND_IMAGES, len(items))]

        winner, meta = self._run_burst_vl_round(
            client=client,
            model=model,
            round_paths=chunk,
            round_index=round_index,
            temperature=temperature,
            max_tokens=max_tokens,
            criteria_lines=criteria_lines,
            row_index=row_index,
        )
        round_meta.append(meta)
        cursor = len(chunk)
        round_index += 1

        while cursor < len(items):
            opponents = items[cursor:cursor + self.VL_BURST_MAX_NEW_CHALLENGERS]
            if not opponents:
                break
            winner, meta = self._run_burst_vl_round(
                client=client,
                model=model,
                round_paths=[winner] + opponents,
                round_index=round_index,
                temperature=temperature,
                max_tokens=max_tokens,
                criteria_lines=criteria_lines,
                row_index=row_index,
            )
            round_meta.append(meta)
            cursor += len(opponents)
            round_index += 1

        return winner, round_meta, round_index

    def _run_burst_vl_round(
        self,
        client: LMStudioClient,
        model: str,
        round_paths: list[Path],
        round_index: int,
        temperature: float,
        max_tokens: int,
        criteria_lines: list[str],
        row_index: int,
    ) -> tuple[Path, dict]:
        grid_path, label_to_path = self._build_burst_round_grid(round_paths, row_index, round_index)
        option_lines = "\n".join(f"- Panel {label}: {path.name}" for label, path in label_to_path.items())
        criteria_prompt = "\n".join(f"- {line}" for line in criteria_lines)

        self._log_async(
            f"Burst Detection: VL row {row_index + 1} round {round_index} candidates ({len(round_paths)}): "
            f"{', '.join(path.name for path in round_paths)}"
        )

        system_prompt = (
            "You are a sports burst-frame comparison assistant.\n"
            "You will see one labeled comparison grid image with panels named Panel A, Panel B, Panel C, or Panel D.\n"
            "Choose the single best deliverable frame from the provided candidates.\n"
            "Apply these selection criteria in order of importance:\n"
            f"{criteria_prompt}\n"
            "Return EXACTLY one valid JSON object and nothing else.\n"
            "Required JSON keys:\n"
            "- winner_label (string, one of A/B/C/D that is present)\n"
            "- runner_up_labels (array of zero or more labels from A/B/C/D that are present)\n"
            "- confidence (number 0..1)\n"
            "- reason (one short sentence)\n"
            "No markdown. No code fences. No extra text."
        )
        user_prompt = (
            "Select the best panel label from this burst round grid.\n"
            "Valid options for this round:\n"
            f"{option_lines}\n"
            "Output exactly one JSON object only."
        )

        raw_text = client.vision_chat_text(
            model=model,
            image_path=grid_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=min(max(0.0, temperature), self.VL_BURST_MAX_TEMPERATURE),
            max_tokens=max(self.VL_BURST_MIN_TOKENS, min(max_tokens, self.VL_BURST_MAX_TOKENS)),
        )
        parsed = client._extract_json_object(raw_text)

        winner = self._resolve_burst_panel_choice(parsed.get("winner_label", ""), label_to_path, round_paths)
        if winner is None:
            winner = self._resolve_burst_panel_choice(parsed.get("best_frame", ""), label_to_path, round_paths)
        if winner is None:
            raise ValueError(f"No valid winner label returned for burst round {round_index}: {parsed}")

        runner_ups: list[str] = []
        for value in parsed.get("runner_up_labels", []) or []:
            pick = self._resolve_burst_panel_choice(value, label_to_path, round_paths)
            if pick is not None:
                runner_ups.append(str(pick))

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        winner_label = ""
        for label, path in label_to_path.items():
            if self._path_key(path) == self._path_key(winner):
                winner_label = label
                break
        if winner_label:
            try:
                self._annotate_burst_round_winner(grid_path, winner_label, len(round_paths))
            except Exception:
                pass

        self._log_async(
            f"Burst Detection: VL row {row_index + 1} round {round_index} winner -> {winner.name}"
        )

        return winner, {
            "round": round_index,
            "grid_path": str(grid_path),
            "winner_path": str(winner),
            "winner_label": winner_label,
            "runner_up_paths": runner_ups,
            "reason": str(parsed.get("reason", "")).strip(),
            "confidence": max(0.0, min(1.0, confidence)),
            "candidate_map": {label: str(path) for label, path in label_to_path.items()},
            "candidate_paths": [str(path) for path in round_paths],
        }

    def _resolve_burst_panel_choice(self, value, label_to_path: dict[str, Path], round_paths: list[Path]) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        upper = text.upper()
        if upper in label_to_path:
            return label_to_path[upper]
        if upper.startswith("PANEL "):
            suffix = upper.split()[-1]
            if suffix in label_to_path:
                return label_to_path[suffix]
        target_name = Path(text).name.lower()
        for path in round_paths:
            if path.name.lower() == target_name:
                return path
        return None

    def _burst_round_layout(self, count: int) -> tuple[int, int]:
        if count <= 2:
            return (2, 1)
        if count == 3:
            return (3, 1)
        return (2, 2)

    def _build_burst_round_grid(self, round_paths: list[Path], row_index: int, round_index: int) -> tuple[Path, dict[str, Path]]:
        if not round_paths:
            raise ValueError("No burst round items")
        if len(round_paths) > self.VL_BURST_MAX_ROUND_IMAGES:
            raise ValueError(f"Burst round supports at most {self.VL_BURST_MAX_ROUND_IMAGES} items")

        panel_w = 560
        panel_h = 480
        gap = 18
        cols, rows = self._burst_round_layout(len(round_paths))
        canvas_w = cols * panel_w + (cols + 1) * gap
        canvas_h = rows * panel_h + (rows + 1) * gap
        canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
        draw = ImageDraw.Draw(canvas, "RGBA")
        label_font = ImageFont.load_default()
        name_font = ImageFont.load_default()

        label_to_path: dict[str, Path] = {}
        labels = self.VL_BURST_PANEL_LABELS[: len(round_paths)]

        for idx, image_path in enumerate(round_paths):
            label = labels[idx]
            label_to_path[label] = image_path
            row = idx // cols
            col = idx % cols
            px = gap + col * (panel_w + gap)
            py = gap + row * (panel_h + gap)
            panel_box = [px, py, px + panel_w, py + panel_h]

            draw.rounded_rectangle(panel_box, radius=16, fill=(38, 38, 38, 255), outline=(215, 215, 215, 255), width=4)

            try:
                image = load_rgb_image(image_path)
            except Exception as exc:
                raise ValueError(f"Failed to load burst round image '{image_path.name}': {exc}") from exc
            max_content_w = panel_w - 28
            max_content_h = panel_h - 118
            image.thumbnail((max_content_w, max_content_h), Image.LANCZOS)
            ix = px + (panel_w - image.width) // 2
            iy = py + 76 + (max_content_h - image.height) // 2
            canvas.paste(image, (ix, iy))

            self._draw_badge(draw, px + 12, py + 12, f"Panel {label}", fill=(0, 0, 0, 235), outline=(255, 255, 255, 255), text_fill=(255, 255, 255, 255), font=label_font)
            self._draw_badge(draw, px + 12, py + panel_h - 60, image_path.name, fill=(0, 0, 0, 220), outline=(170, 170, 170, 255), text_fill=(240, 240, 240, 255), font=name_font)

        debug_dir = round_paths[0].parent / "VL_Debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        grid_path = debug_dir / f"burst_row_{row_index + 1:03d}_round_{round_index:02d}.jpg"
        canvas.save(grid_path, format="JPEG", quality=92)
        return grid_path, label_to_path

    def _draw_badge(self, draw, x: int, y: int, text: str, fill, outline, text_fill, font):
        bbox = draw.textbbox((x, y), text, font=font)
        pad_x = 12
        pad_y = 8
        badge = [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y]
        draw.rounded_rectangle(badge, radius=12, fill=fill, outline=outline, width=3)
        draw.text((x, y), text, fill=text_fill, font=font)

    def _annotate_burst_round_winner(self, grid_path: Path, winner_label: str, candidate_count: int) -> None:
        label = str(winner_label or "").strip().upper()
        if not label:
            return
        labels = self.VL_BURST_PANEL_LABELS[: max(0, int(candidate_count))]
        if label not in labels:
            return

        panel_w = 560
        panel_h = 480
        gap = 18
        cols, _ = self._burst_round_layout(len(labels))
        panel_index = labels.index(label)
        row = panel_index // cols
        col = panel_index % cols
        px = gap + col * (panel_w + gap)
        py = gap + row * (panel_h + gap)

        with Image.open(grid_path) as img:
            canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")
        panel_box = [px + 6, py + 6, px + panel_w - 6, py + panel_h - 6]
        draw.rounded_rectangle(panel_box, radius=18, outline=_LIME_RGB + (255,), width=10)

        marker_r = 34
        marker_cx = px + panel_w - 56
        marker_cy = py + 56
        draw.ellipse(
            [marker_cx - marker_r, marker_cy - marker_r, marker_cx + marker_r, marker_cy + marker_r],
            fill=_LIME_RGB + (245,),
            outline=(238, 255, 244, 255),
            width=4,
        )
        draw.line(
            [
                (marker_cx - 15, marker_cy + 2),
                (marker_cx - 3, marker_cy + 16),
                (marker_cx + 18, marker_cy - 10),
            ],
            fill=(16, 16, 16, 255),
            width=8,
            joint="curve",
        )
        canvas.save(grid_path, format="JPEG", quality=92)

    def _apply_vl_winners(self, row_index: int, winner_paths: list[Path], meta: dict):
        self._set_row_winners(row_index, [Path(p) for p in winner_paths], source="vl", vl_meta=meta)
        self.app.log(
            f"Burst Detection: VL row {row_index + 1} winner(s) -> {', '.join(Path(p).name for p in winner_paths)}"
        )

    def _finish_vl_run(self):
        self._vl_running = False
        self.app.log("Burst Detection: VL processing complete.")
