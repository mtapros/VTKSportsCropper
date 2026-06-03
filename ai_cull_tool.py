from __future__ import annotations

import hashlib
import inspect
import json
import math
import queue
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

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
SCENE_TYPE_VALUES = {"intro_pose", "finale_pose", "group_static_pose", "action", "unknown"}
MIN_STATIC_GROUP_BURST_KEEP = 2


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
    BURST_EVAL_WINDOW_GEOMETRY = "520x420"

    DANCE_PICK_COLORS = [
        ("red", "#FF4D4D"),
        ("yellow", "#FFD84D"),
        ("cyan", "#33D6FF"),
        ("lime", "#7CFF4D"),
        ("magenta", "#FF5CFF"),
        ("orange", "#FF9A3D"),
    ]

    VL_TARGET_LONG_EDGE = 1024
    MAX_VL_BURST_CANDIDATES = 6
    VL_BURST_SCORE_TIE_THRESHOLD = 8.0

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
        self.hide_burst_suppressed_var = tk.BooleanVar(value=True)
        self.prefer_face_var = tk.BooleanVar(value=True)
        self.use_object_cull_var = tk.BooleanVar(value=True)
        self.use_vision_cull_var = tk.BooleanVar(value=False)

        self.use_dance_vl_var = tk.BooleanVar(value=True)
        self.use_dance_vl_subject_picker_var = tk.BooleanVar(value=True)
        self.use_dance_scene_classifier_var = tk.BooleanVar(value=True)
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
        self.current_scene_classification: dict | None = None

        self.auto_running = False
        self.auto_cancel_requested = False
        self.auto_images: list[Path] = []
        self.auto_results: list[dict] = []
        self.auto_index = 0
        self.auto_mode = "auto_cull"
        self.auto_button = None
        self.run_cull_button = None
        self.run_current_button = None
        self.run_full_button = None
        self.select_folder_button = None
        self.burst_button = None
        self.evaluate_bursts_button = None
        self.use_object_checkbox = None
        self.use_vision_checkbox = None
        self.object_settings_button = None
        self.vision_settings_button = None
        self.crop_keep_button = None
        self.lmstudio_settings_button = None
        self.stop_button = None
        self.burst_eval_window = None
        self.object_settings_window = None
        self.vision_settings_window = None
        self.input_folder_var = tk.StringVar(value="Selected folder: —")
        self.loaded_count_var = tk.StringVar(value="Loaded: 0 images")
        self.burst_eval_details_var = tk.StringVar(value="Burst preflight not run.")
        self.burst_accounting_var = tk.StringVar(
            value="Burst images = 0\nBurst images removed = 0\nRemaining burst images = 0\nNon-burst images = 0\nTotal remaining images = 0"
        )
        self.mid_process_var = tk.StringVar(
            value="Starting images = 0\nKeepers = 0\nMaybe = 0\nReject = 0\nUnprocessed = 0"
        )
        self.final_accounting_var = tk.StringVar(
            value="Input folder images = 0\nRejected by burst cull = 0\nRejected by AI cull = 0\nKept = 0\nMaybe = 0\nUnprocessed = 0"
        )
        self.vision_warning_var = tk.StringVar(value="")
        self.loaded_image_count = 0
        self.accounting_folder_key: str | None = None
        self.removed_by_bursts_count: int | None = None
        self.remaining_after_burst_count: int | None = None
        self.vision_model_loaded = False
        self.burst_analysis_complete = False
        self.burst_paths: set[str] = set()
        self.burst_removed_paths: set[str] = set()
        self.burst_remaining_paths: list[Path] = []
        self.burst_summary: dict = {}
        self.ai_processed_results: dict[str, dict] = {}
        self.precomputed_burst_state: dict | None = None
        self.auto_all_images: list[Path] = []
        self.auto_precomputed_bursts: list[list[Path]] = []
        self._worker_thread: threading.Thread | None = None
        self._worker_queue: queue.Queue = queue.Queue()
        self._pending_full_workflow_after_burst = False
        self._cancel_event = threading.Event()

        self.dance_frame = None

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")
        pad = {"padx": 10, "pady": 4}

        self.select_folder_button = tk.Button(self.panel, text="Select Folder", command=self.select_input_folder)
        self.select_folder_button.pack(fill="x", padx=10, pady=(6, 4))
        tk.Label(self.panel, textvariable=self.input_folder_var, bg="#2a2a2a", fg="#d9d9d9", justify="left", wraplength=320).pack(anchor="w", padx=10, pady=(0, 2))
        tk.Label(self.panel, textvariable=self.loaded_count_var, bg="#2a2a2a", fg="#c9d7ff").pack(anchor="w", padx=10, pady=(0, 6))
        ttk.Separator(self.panel, orient="horizontal").pack(fill="x", padx=10, pady=(2, 8))

        self.evaluate_bursts_button = tk.Button(self.panel, text="Evaluate Bursts", command=self.evaluate_bursts)
        self.evaluate_bursts_button.pack(fill="x", padx=10, pady=(0, 6))
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
            text="Hide Burst-Suppressed Images",
            variable=self.hide_burst_suppressed_var,
            command=self.refresh_burst_browser_view,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Label(self.panel, textvariable=self.burst_accounting_var, bg="#2a2a2a", fg="#d3e3ff", justify="left", wraplength=320).pack(anchor="w", padx=10, pady=(4, 6))
        ttk.Separator(self.panel, orient="horizontal").pack(fill="x", padx=10, pady=(2, 8))

        self.use_object_checkbox = tk.Checkbutton(
            self.panel,
            text="Use Object Cull",
            variable=self.use_object_cull_var,
            command=self._on_object_checkbox_toggled,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        )
        self.use_object_checkbox.pack(anchor="w", **pad)
        self.object_settings_button = tk.Button(self.panel, text="Cull by Object", command=self.cull_current_by_object)
        self.object_settings_button.pack(fill="x", padx=10, pady=(0, 4))
        self.use_vision_checkbox = tk.Checkbutton(
            self.panel,
            text="Use Vision Cull",
            variable=self.use_vision_cull_var,
            command=self._on_vision_checkbox_toggled,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        )
        self.use_vision_checkbox.pack(anchor="w", **pad)
        self.vision_settings_button = tk.Button(self.panel, text="Cull by Vision", command=self.cull_current_by_vision)
        self.vision_settings_button.pack(fill="x", padx=10, pady=(0, 2))
        tk.Label(self.panel, textvariable=self.vision_warning_var, bg="#2a2a2a", fg="#ffd27f", justify="left", wraplength=320).pack(anchor="w", padx=10, pady=(0, 8))
        ttk.Separator(self.panel, orient="horizontal").pack(fill="x", padx=10, pady=(2, 8))

        self.run_cull_button = tk.Button(self.panel, text="Run Cull", command=self.run_cull)
        self.run_cull_button.pack(fill="x", padx=10, pady=(0, 4))
        self.run_current_button = tk.Button(self.panel, text="Run Current Image", command=self.run_current_image)
        self.run_current_button.pack(fill="x", padx=10, pady=(0, 6))

        tk.Label(self.panel, textvariable=self.mid_process_var, bg="#2a2a2a", fg="#d3e3ff", justify="left", wraplength=320).pack(anchor="w", padx=10, pady=(0, 6))
        ttk.Separator(self.panel, orient="horizontal").pack(fill="x", padx=10, pady=(2, 8))

        self.run_full_button = tk.Button(self.panel, text="Run Full Workflow", command=self.run_full_workflow)
        self.run_full_button.pack(fill="x", padx=10, pady=(0, 6))

        tk.Label(self.panel, textvariable=self.final_accounting_var, bg="#2a2a2a", fg="#c6ffc6", justify="left", wraplength=320).pack(anchor="w", padx=10, pady=(0, 8))
        self.stop_button = tk.Button(self.panel, text="Stop", command=self.stop_auto_cull, state="disabled", bg="#8b1e1e", fg="white")
        self.stop_button.pack(fill="x", padx=10, pady=(0, 4))

        self._sync_folder_and_loaded_state()
        self._update_vision_warning()
        self._refresh_accounting_labels()
        self._refresh_dynamic_sections()
        return self.panel

    def _current_profile_is_dance(self) -> bool:
        profile = self.app.get_current_profile()
        sport_type = getattr(profile, "sport_type", "").strip().lower()
        if sport_type == "dance":
            return True
        return "dance" in profile.name.lower()

    def _get_loaded_image_paths(self) -> list:
        all_paths = getattr(self.app.state, "all_image_paths", None)
        visible_paths = getattr(self.app.state, "image_paths", None)
        return list(all_paths or visible_paths or [])

    def _sync_folder_and_loaded_state(self):
        folder = self.app.state.input_folder
        folder_key = str(folder.resolve()) if folder is not None else None
        if folder_key != self.accounting_folder_key:
            self.accounting_folder_key = folder_key
            self.precomputed_burst_state = None
            self.burst_analysis_complete = False
            self.burst_paths = set()
            self.burst_removed_paths = set()
            self.burst_remaining_paths = []
            self.burst_summary = {}
            self.ai_processed_results = {}
            self._pending_full_workflow_after_burst = False
        if folder is None:
            self.input_folder_var.set("Selected folder: —")
        else:
            self.input_folder_var.set(str(folder))
        self.loaded_image_count = len(self._get_loaded_image_paths())
        self.loaded_count_var.set(f"Loaded: {self.loaded_image_count} images")
        self._refresh_accounting_labels()

    def _update_vision_warning(self):
        tool = self.app.tools_by_id.get("lmstudio")
        model = ""
        if tool is not None and hasattr(tool, "model_var"):
            try:
                model = str(tool.model_var.get() or "").strip()
            except Exception:
                model = ""
        self.vision_model_loaded = bool(model)
        if self.vision_model_loaded:
            self.vision_warning_var.set("")
        else:
            self.use_vision_cull_var.set(False)
            self.vision_warning_var.set("No vision model loaded.")
        self._refresh_vision_object_controls()

    def _refresh_accounting_labels(self):
        burst_images = int(self.burst_summary.get("burst_images", 0))
        burst_removed = int(self.burst_summary.get("burst_images_removed", 0))
        burst_remaining = int(self.burst_summary.get("remaining_burst_images", 0))
        non_burst = int(self.burst_summary.get("non_burst_images", 0))
        total_remaining = int(self.burst_summary.get("total_remaining_images", 0))
        self.burst_accounting_var.set(
            f"Burst images = {burst_images}\n"
            f"Burst images removed = {burst_removed}\n"
            f"Remaining burst images = {burst_remaining}\n"
            f"Non-burst images = {non_burst}\n"
            f"Total remaining images = {total_remaining}"
        )

        remaining_keys = {str(Path(p).resolve()) for p in self.burst_remaining_paths}
        keepers = 0
        maybe = 0
        reject = 0
        for key, item in self.ai_processed_results.items():
            if key not in remaining_keys:
                continue
            decision = str(item.get("decision", "Reject"))
            if decision == "Keep":
                keepers += 1
            elif decision == "Maybe":
                maybe += 1
            else:
                reject += 1
        starting = total_remaining
        unprocessed = max(0, starting - keepers - maybe - reject)
        self.mid_process_var.set(
            f"Starting images = {starting}\nKeepers = {keepers}\nMaybe = {maybe}\nReject = {reject}\nUnprocessed = {unprocessed}"
        )

        input_total = self.loaded_image_count
        rejected_by_burst = burst_removed
        rejected_by_ai = reject
        kept = keepers
        maybe_final = maybe
        final_unprocessed = max(0, input_total - rejected_by_burst - rejected_by_ai - kept - maybe_final)
        self.final_accounting_var.set(
            f"Input folder images = {input_total}\n"
            f"Rejected by burst cull = {rejected_by_burst}\n"
            f"Rejected by AI cull = {rejected_by_ai}\n"
            f"Kept = {kept}\n"
            f"Maybe = {maybe_final}\n"
            f"Unprocessed = {final_unprocessed}"
        )

    def _reset_accounting_for_new_folder(self):
        self.accounting_folder_key = None
        self._sync_folder_and_loaded_state()

    def select_input_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.app.set_input_folder(folder)
        self._reset_accounting_for_new_folder()
        self._sync_folder_and_loaded_state()
        self._update_vision_warning()

    def _refresh_dynamic_sections(self):
        return

    def _refresh_vision_object_controls(self):
        vision_enabled = bool(self.use_vision_cull_var.get()) and self.vision_model_loaded
        if vision_enabled:
            self.use_object_cull_var.set(True)
        if self.use_vision_checkbox is not None:
            self.use_vision_checkbox.configure(state="normal" if self.vision_model_loaded else "disabled")
        if self.vision_settings_button is not None:
            self.vision_settings_button.configure(state="normal" if self.vision_model_loaded else "disabled")
        object_state = "disabled" if vision_enabled else "normal"
        if self.use_object_checkbox is not None:
            self.use_object_checkbox.configure(state=object_state)

    def _on_vision_checkbox_toggled(self):
        if self.use_vision_cull_var.get():
            self.use_object_cull_var.set(True)
        self._refresh_vision_object_controls()

    def _on_object_checkbox_toggled(self):
        if not self.use_object_cull_var.get() and self.use_vision_cull_var.get():
            self.use_object_cull_var.set(True)
        self._refresh_vision_object_controls()

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
        vision_enabled = bool(self.use_vision_cull_var.get()) and self.vision_model_loaded
        config = {
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
            "use_object_cull": bool(self.use_object_cull_var.get()),
            "use_vision_cull": vision_enabled,
            "use_dance_vl": bool(self.use_dance_vl_var.get()) and vision_enabled,
            "use_dance_vl_subject_picker": bool(self.use_dance_vl_subject_picker_var.get()) and vision_enabled,
            "use_dance_scene_classifier": bool(self.use_dance_scene_classifier_var.get()) and vision_enabled,
            "save_vl_debug_images": bool(self.save_vl_debug_images_var.get()),
            "show_dance_debug_preview": bool(self.show_dance_debug_preview_var.get()),
        }
        if self.precomputed_burst_state and self.app.state.input_folder is not None:
            state_folder = str(self.precomputed_burst_state.get("folder", ""))
            if state_folder == str(self.app.state.input_folder.resolve()):
                config["precomputed_burst_groups"] = list(self.precomputed_burst_state.get("groups", []))
                config["precomputed_burst_fps"] = float(self.precomputed_burst_state.get("fps", config["burst_fps"]))
        return config

    def open_lmstudio_settings_window(self):
        tool = self.app.tools_by_id.get("lmstudio")
        if tool is None or not hasattr(tool, "open_settings_window"):
            self.app.log("AI Cull: LM Studio tool unavailable.")
            return
        tool.open_settings_window()

    def open_object_settings_window(self):
        if self.object_settings_window is not None and self.object_settings_window.winfo_exists():
            self.object_settings_window.deiconify()
            self.object_settings_window.lift()
            self.object_settings_window.focus_force()
            return

        win = tk.Toplevel(self.app.root)
        win.title("Cull by Object")
        win.geometry("480x620")
        win.configure(bg="#2a2a2a")
        self.object_settings_window = win
        win.protocol("WM_DELETE_WINDOW", lambda: (setattr(self, "object_settings_window", None), win.destroy()))

        pad = {"padx": 10, "pady": 4}
        tk.Label(win, text="Object Culling Settings", bg="#2a2a2a", fg="white", font=("Arial", 11, "bold")).pack(anchor="w", **pad)
        tk.Label(win, text="Detection Mode", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        ttk.Combobox(
            win,
            textvariable=self.detection_mode_var,
            values=["Hybrid", "Phrase Only"],
            state="readonly",
        ).pack(fill="x", **pad)

        tk.Label(win, text="Prompts", bg="#2a2a2a", fg="white", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 4), padx=10)
        for i, var in enumerate(self.prompt_vars, start=1):
            tk.Label(win, text=f"Prompt {i}", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
            tk.Entry(win, textvariable=var).pack(fill="x", **pad)

        tk.Label(win, text="Keep Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(win, textvariable=self.keep_threshold_var).pack(fill="x", **pad)
        tk.Label(win, text="Maybe Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(win, textvariable=self.maybe_threshold_var).pack(fill="x", **pad)
        tk.Label(win, text="Blur Penalty Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(win, textvariable=self.blur_penalty_threshold_var).pack(fill="x", **pad)
        tk.Label(win, text="Blur Reject Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(win, textvariable=self.blur_reject_threshold_var).pack(fill="x", **pad)
        tk.Label(win, text="Blur Penalty Points", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(win, textvariable=self.blur_penalty_points_var).pack(fill="x", **pad)
        tk.Checkbutton(
            win,
            text="Prefer Visible Face",
            variable=self.prefer_face_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Button(win, text="Close", command=win.destroy).pack(fill="x", padx=10, pady=(8, 8))

    def open_vision_settings_window(self):
        if self.vision_settings_window is not None and self.vision_settings_window.winfo_exists():
            self.vision_settings_window.deiconify()
            self.vision_settings_window.lift()
            self.vision_settings_window.focus_force()
            return

        win = tk.Toplevel(self.app.root)
        win.title("Cull by Vision")
        win.geometry("480x360")
        win.configure(bg="#2a2a2a")
        self.vision_settings_window = win
        win.protocol("WM_DELETE_WINDOW", lambda: (setattr(self, "vision_settings_window", None), win.destroy()))

        pad = {"padx": 10, "pady": 4}
        tk.Label(win, text="Vision Culling Settings", bg="#2a2a2a", fg="white", font=("Arial", 11, "bold")).pack(anchor="w", **pad)
        tk.Checkbutton(
            win,
            text="Use LM Studio Vision Rubric",
            variable=self.use_dance_vl_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Checkbutton(
            win,
            text="Use Vision Subject Picker",
            variable=self.use_dance_vl_subject_picker_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Checkbutton(
            win,
            text="Classify Intro/Finale/Group Poses",
            variable=self.use_dance_scene_classifier_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Checkbutton(
            win,
            text="Save Vision Debug Images",
            variable=self.save_vl_debug_images_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Checkbutton(
            win,
            text="Show 4-Panel Debug Preview",
            variable=self.show_dance_debug_preview_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Button(win, text="LM Studio Settings…", command=self.open_lmstudio_settings_window).pack(fill="x", padx=10, pady=(8, 4))
        self._update_vision_warning()
        if self.vision_warning_var.get().strip():
            tk.Label(
                win,
                textvariable=self.vision_warning_var,
                bg="#2a2a2a",
                fg="#ffd27f",
                justify="left",
                wraplength=440,
            ).pack(anchor="w", padx=10, pady=(0, 6))
        tk.Button(win, text="Close", command=win.destroy).pack(fill="x", padx=10, pady=(6, 8))

    def open_keep_bucket_for_cropping(self):
        if self.app.state.input_folder is None:
            self.app.log("AI Cull: select an input folder first.")
            return
        keep_folder = self.app.state.input_folder / "Output" / "Keep"
        if not keep_folder.exists() or not keep_folder.is_dir():
            self.app.log("AI Cull: Output/Keep not found. Run unified pipeline first.")
            return
        self.app.set_input_folder(str(keep_folder))
        self.app.set_active_tool("ai_crop")
        self.app.log(f"AI Cull: ready to crop keep bucket at {keep_folder}.")

    def _summarize_burst_groups(self, groups: list[list[Path]], total_images: int) -> dict:
        burst_groups = [g for g in groups if len(g) > 1]
        burst_images = sum(len(g) for g in burst_groups)
        largest = max((len(g) for g in burst_groups), default=0)
        singleton_count = max(0, total_images - burst_images)
        return {
            "group_count": len(burst_groups),
            "burst_images": burst_images,
            "singleton_images": singleton_count,
            "largest_group": largest,
        }

    def prepare_burst_groups_for_paths(
        self,
        ordered_paths: list[Path],
        config: dict | None = None,
        source_folder: Path | None = None,
    ) -> dict:
        runtime = config if config is not None else self.get_runtime_config()
        fps = float(runtime.get("burst_fps", 8.0))
        groups = self._build_bursts([Path(p) for p in ordered_paths], fps)
        summary = self._summarize_burst_groups(groups, len(ordered_paths))
        keep_per_burst = max(1, int(runtime.get("keep_per_burst", 1)))
        removed_by_bursts = sum(max(0, len(group) - keep_per_burst) for group in groups if len(group) > 1)
        remaining_after_burst = max(0, len(ordered_paths) - removed_by_bursts)
        folder = source_folder or self.app.state.input_folder
        if folder is not None:
            self.precomputed_burst_state = {
                "folder": str(folder.resolve()),
                "fps": fps,
                "groups": [[str(p) for p in g] for g in groups],
            }
            if config is not None:
                config["precomputed_burst_groups"] = list(self.precomputed_burst_state["groups"])
                config["precomputed_burst_fps"] = fps
        self.removed_by_bursts_count = removed_by_bursts
        self.remaining_after_burst_count = remaining_after_burst
        self._refresh_accounting_labels()
        self.burst_eval_details_var.set(
            f"Burst evaluation @ {fps:.2f} FPS: groups={summary['group_count']}, "
            f"burst-images={summary['burst_images']}, removed-by-bursts={removed_by_bursts}, "
            f"remaining-after-filter={remaining_after_burst}, largest={summary['largest_group']}"
        )
        return {
            "fps": fps,
            "groups": groups,
            "removed_by_bursts": removed_by_bursts,
            "remaining_after_burst": remaining_after_burst,
            **summary,
        }

    def _resolve_burst_groups(self, ordered_paths: list[Path], config: dict) -> list[list[Path]]:
        resolved_paths = [Path(p) for p in ordered_paths]
        raw_groups = config.get("precomputed_burst_groups")
        if not isinstance(raw_groups, list):
            return self._build_bursts(resolved_paths, float(config.get("burst_fps", 8.0)))

        path_lookup = {str(p.resolve()): p for p in resolved_paths}
        groups: list[list[Path]] = []
        used: set[Path] = set()
        for raw_group in raw_groups:
            if not isinstance(raw_group, list):
                continue
            group: list[Path] = []
            for raw_path in raw_group:
                key = str(Path(raw_path).resolve())
                path = path_lookup.get(key)
                if path is None or path in used:
                    continue
                group.append(path)
                used.add(path)
            if group:
                groups.append(group)

        if not groups:
            return self._build_bursts(resolved_paths, float(config.get("burst_fps", 8.0)))
        return groups

    def _persist_non_burst_defaults(self, all_paths: list[Path], grouped_paths: set[Path]) -> None:
        for path in all_paths:
            if path in grouped_paths:
                continue
            self._put_cached_entry(Path(path), self._default_burst_metadata_updates())

    @staticmethod
    def _default_burst_metadata_updates() -> dict:
        return {
            "burst_group_id": None,
            "burst_rank": 0,
            "burst_size": 1,
            "burst_suppressed": False,
            "burst_winner_paths": [],
            "burst_vl_selector_used": False,
            "burst_keep_target": 1,
            "burst_conservative_scene_mode": False,
        }

    def _compute_burst_summary(self, paths: list[Path], fps: float, keep_per_burst: int, enable_burst: bool) -> tuple[dict, list[list[Path]], set[str], list[Path]]:
        groups = self._build_bursts(paths, fps)
        burst_groups = [group for group in groups if len(group) > 1]
        burst_images = sum(len(group) for group in burst_groups)
        removed_paths: set[str] = set()

        if enable_burst:
            for group in burst_groups:
                for suppressed_path in group[max(1, keep_per_burst):]:
                    removed_paths.add(str(Path(suppressed_path).resolve()))

        remaining_paths = [p for p in paths if str(Path(p).resolve()) not in removed_paths]
        summary = {
            "burst_images": burst_images,
            "burst_images_removed": len(removed_paths),
            "remaining_burst_images": max(0, burst_images - len(removed_paths)),
            "non_burst_images": max(0, len(paths) - burst_images),
            "total_remaining_images": len(remaining_paths),
        }
        return summary, burst_groups, removed_paths, remaining_paths

    def _persist_burst_evaluation_cache(self, all_paths: list[Path], burst_groups: list[list[Path]], removed_paths: set[str], keep_per_burst: int):
        grouped_paths: set[Path] = set()
        for idx, group in enumerate(burst_groups, start=1):
            kept_paths = [str(p) for p in group[:max(1, keep_per_burst)]]
            for rank, path in enumerate(group, start=1):
                p = Path(path)
                grouped_paths.add(p)
                resolved = str(p.resolve())
                self._put_cached_entry(
                    p,
                    {
                        "burst_group_id": f"burst_{idx:04d}",
                        "burst_rank": rank,
                        "burst_size": len(group),
                        "burst_suppressed": resolved in removed_paths,
                        "burst_winner_paths": kept_paths,
                        "burst_vl_selector_used": False,
                        "burst_keep_target": max(1, keep_per_burst),
                        "burst_conservative_scene_mode": False,
                    },
                )
        self._persist_non_burst_defaults(all_paths, grouped_paths)

    def evaluate_bursts(self):
        if self.auto_running:
            self.app.log("AI Cull: another batch job is already running.")
            return
        if self.app.state.input_folder is None:
            self.app.log("AI Cull: select an input folder first.")
            return

        self._sync_folder_and_loaded_state()
        if not self._get_loaded_image_paths():
            self.app.start_batch()
            self._sync_folder_and_loaded_state()
        paths = self._folder_batch_source_paths()
        if not paths:
            self.app.log("AI Cull: no images found in input folder.")
            return

        fps = float(self.burst_fps_var.get().strip() or "8")
        keep_per_burst = int(self.keep_per_burst_var.get().strip() or "1")
        enable_burst = bool(self.enable_burst_var.get())
        self.auto_cancel_requested = False
        self._cancel_event.clear()
        self._set_running_state(True, "evaluate_bursts")

        def worker():
            try:
                summary, burst_groups, removed_paths, remaining_paths = self._compute_burst_summary(paths, fps, keep_per_burst, enable_burst)
                self._worker_queue.put(("burst_done", paths, fps, keep_per_burst, summary, burst_groups, removed_paths, remaining_paths))
            except Exception as exc:
                self._worker_queue.put(("burst_error", str(exc)))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
        self.app.root.after(40, self._poll_worker_queue)

    def _run_burst_preflight_scan(self):
        self.evaluate_bursts()
        return self.burst_summary or None

    def open_burst_evaluation_window(self):
        if self.burst_eval_window is not None and self.burst_eval_window.winfo_exists():
            self.burst_eval_window.deiconify()
            self.burst_eval_window.lift()
            self.burst_eval_window.focus_force()
            return

        win = tk.Toplevel(self.app.root)
        win.title("Evaluate Bursts")
        win.geometry(self.BURST_EVAL_WINDOW_GEOMETRY)
        win.configure(bg="#2a2a2a")
        self.burst_eval_window = win

        pad = {"padx": 10, "pady": 4}
        tk.Label(
            win,
            text="Fast timestamp-only burst scan",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", **pad)
        tk.Checkbutton(
            win,
            text="Enable Burst Suppression",
            variable=self.enable_burst_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Label(win, text="Burst FPS Threshold", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(win, textvariable=self.burst_fps_var).pack(fill="x", **pad)
        tk.Label(win, text="Keep Per Burst", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(win, textvariable=self.keep_per_burst_var).pack(fill="x", **pad)
        tk.Checkbutton(
            win,
            text="Use VL Burst Tie-Breaker",
            variable=self.use_vl_burst_tiebreaker_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Checkbutton(
            win,
            text="Hide Burst-Suppressed Images",
            variable=self.hide_burst_suppressed_var,
            command=self.refresh_burst_browser_view,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)
        tk.Button(win, text="Run Burst Scan", command=self._run_burst_preflight_scan).pack(fill="x", padx=10, pady=(8, 6))
        tk.Label(
            win,
            textvariable=self.burst_eval_details_var,
            bg="#2a2a2a",
            fg="#c9d7ff",
            justify="left",
            wraplength=440,
        ).pack(anchor="w", padx=10, pady=(4, 6))
        tk.Button(win, text="Close", command=win.destroy).pack(fill="x", padx=10, pady=(6, 8))

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

    def _get_cached_entry(self, image_path: Path, rules_hash: str | None = None) -> dict:
        try:
            entry = self._get_dance_cull_cache(image_path).get(image_path, rules_hash or self._dance_rules_hash())
        except Exception:
            entry = None
        return dict(entry) if isinstance(entry, dict) else {}

    def _put_cached_entry(self, image_path: Path, updates: dict, rules_hash: str | None = None) -> dict:
        active_rules_hash = rules_hash or self._dance_rules_hash()
        cache = self._get_dance_cull_cache(image_path)
        merged = self._get_cached_entry(image_path, active_rules_hash)
        merged.update(updates)
        merged["image_path"] = str(image_path)
        cache.put(image_path, active_rules_hash, merged)
        return merged

    def _get_burst_cache_entry(self, image_path: Path) -> dict:
        entry = self._get_cached_entry(image_path)
        if "burst_group_id" not in entry and "burst_suppressed" not in entry:
            return {}
        return entry

    def get_folder_browser_state(self, ordered_paths: list[Path]) -> tuple[list[Path], list[str]]:
        visible_paths: list[Path] = []
        labels: list[str] = []
        hide_suppressed = bool(self.hide_burst_suppressed_var.get())

        for raw_path in ordered_paths:
            path = Path(raw_path)
            burst_entry = self._get_burst_cache_entry(path)
            is_suppressed = bool(burst_entry.get("burst_suppressed", False))
            label = path.name
            if is_suppressed:
                label = f"{path.name} [Burst Suppressed]"
            elif int(burst_entry.get("burst_size", 1)) > 1:
                label = f"{path.name} [Burst Kept]"

            if is_suppressed and hide_suppressed:
                continue

            visible_paths.append(path)
            labels.append(label)

        if not visible_paths and ordered_paths:
            return list(ordered_paths), [Path(p).name for p in ordered_paths]
        return visible_paths, labels

    def refresh_burst_browser_view(self):
        if self.app.state.input_folder is None:
            return
        self.app.refresh_image_browser()

    def _log_current_burst_state(self):
        image_path = self.app.state.current_image_path
        if image_path is None:
            return

        burst_entry = self._get_burst_cache_entry(Path(image_path))
        burst_size = int(burst_entry.get("burst_size", 1))
        if burst_size <= 1:
            return

        burst_rank = int(burst_entry.get("burst_rank", 0))
        burst_group_id = str(burst_entry.get("burst_group_id", "burst"))
        if bool(burst_entry.get("burst_suppressed", False)):
            winners = [Path(p).name for p in burst_entry.get("burst_winner_paths", []) if p]
            self.app.log(
                f"AI Cull Burst: {Path(image_path).name} suppressed in {burst_group_id} "
                f"(rank {burst_rank}/{burst_size}; kept: {', '.join(winners) or 'none'})."
            )
        else:
            self.app.log(
                f"AI Cull Burst: {Path(image_path).name} kept in {burst_group_id} "
                f"(rank {burst_rank}/{burst_size})."
            )

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
        blur_penalty_threshold: float | None = None,
        blur_penalty_points: float | None = None,
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

        if blur_penalty_threshold is None:
            blur_penalty_threshold = float(self.blur_penalty_threshold_var.get().strip() or "12")
        if blur_penalty_points is None:
            blur_penalty_points = float(self.blur_penalty_points_var.get().strip() or "20")

        if hero is not None and focus_score < blur_penalty_threshold:
            breakdown["blur_penalty"] = -blur_penalty_points

        if crop_center_penalty < 0.0:
            breakdown["crop_center"] = crop_center_penalty

        total = sum(breakdown.values())
        return total, breakdown

    def _decision_from_score(
        self,
        score: float,
        hero_focus_score: float,
        hero_exists: bool,
        keep_threshold: float | None = None,
        maybe_threshold: float | None = None,
        blur_reject_threshold: float | None = None,
    ) -> tuple[str, str | None]:
        if not hero_exists:
            return "Reject", "no_hero"

        if blur_reject_threshold is None:
            blur_reject_threshold = float(self.blur_reject_threshold_var.get().strip() or "6")
        if keep_threshold is None:
            keep_threshold = float(self.keep_threshold_var.get().strip() or "80")
        if maybe_threshold is None:
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

    def _default_scene_classification(self) -> dict:
        return {
            "scene_type": "unknown",
            "is_group_pose": False,
            "is_static_pose": False,
            "should_keep_full_frame": False,
            "should_avoid_subject_crop": False,
            "reason": "",
            "confidence": 0.0,
        }

    def _normalize_scene_classification(self, value: dict | None) -> dict:
        merged = self._default_scene_classification()
        if isinstance(value, dict):
            merged.update(value)

        scene_type = str(merged.get("scene_type", "unknown")).strip().lower()
        if scene_type not in SCENE_TYPE_VALUES:
            scene_type = "unknown"

        try:
            confidence = float(merged.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        return {
            "scene_type": scene_type,
            "is_group_pose": LMStudioClient._to_bool(merged.get("is_group_pose", False)),
            "is_static_pose": LMStudioClient._to_bool(merged.get("is_static_pose", False)),
            "should_keep_full_frame": LMStudioClient._to_bool(merged.get("should_keep_full_frame", False)),
            "should_avoid_subject_crop": LMStudioClient._to_bool(merged.get("should_avoid_subject_crop", False)),
            "reason": str(merged.get("reason", "")).strip(),
            "confidence": max(0.0, min(1.0, confidence)),
        }

    def _scene_requires_composition_preservation(self, scene: dict | None) -> bool:
        normalized = self._normalize_scene_classification(scene)
        if normalized["scene_type"] in LMStudioClient.COMPOSITION_PRESERVE_SCENE_TYPES:
            return True
        return bool(normalized["should_keep_full_frame"] or normalized["should_avoid_subject_crop"])

    def _scene_is_static_group_pose(self, item: dict) -> bool:
        scene = item.get("scene_classification")
        if not isinstance(scene, dict):
            scene = {
                "scene_type": item.get("scene_type", "unknown"),
                "is_group_pose": item.get("is_group_pose", False),
                "is_static_pose": item.get("is_static_pose", False),
                "should_keep_full_frame": item.get("should_keep_full_frame", False),
                "should_avoid_subject_crop": item.get("should_avoid_subject_crop", False),
            }
        scene = self._normalize_scene_classification(scene)
        if self._scene_requires_composition_preservation(scene):
            return True
        return bool(scene.get("is_group_pose", False) and scene.get("is_static_pose", False))

    def _classify_scene_with_vl(self, image_path: Path) -> dict:
        base_url, model, timeout, temperature, max_tokens = self._dance_lmstudio_settings()
        client = LMStudioClient(base_url=base_url, timeout=timeout)
        scene = client.classify_scene_type(
            model=model,
            image_path=image_path,
            temperature=min(temperature, 0.2),
            max_tokens=max(256, min(max_tokens, 450)),
        )
        return self._normalize_scene_classification(scene)

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
            scene_classification = self._default_scene_classification()
            if bool(config.get("use_dance_scene_classifier", False)):
                try:
                    scene_classification = self._classify_scene_with_vl(Path(image_path))
                    self.app.log(
                        "AI Cull Dance VL scene: "
                        f"type={scene_classification.get('scene_type', 'unknown')} "
                        f"full_frame={scene_classification.get('should_keep_full_frame', False)} "
                        f"avoid_subject_crop={scene_classification.get('should_avoid_subject_crop', False)} "
                        f"confidence={float(scene_classification.get('confidence', 0.0)):.2f}"
                    )
                except Exception as exc:
                    self.app.log(f"AI Cull Dance VL scene classifier failed: {exc}")

            if self._scene_requires_composition_preservation(scene_classification):
                crop_proposal = BoundingBox(0, 0, img_w, img_h) if img_w > 0 and img_h > 0 else None
                crop_center_penalty = 0.0
                self.app.log(
                    "AI Cull Dance: preserving full-frame composition for "
                    f"{scene_classification.get('scene_type', 'unknown')} scene."
                )
            else:
                crop_proposal, crop_center_penalty = self._compute_crop_proposal(
                    chosen, img_w, img_h, profile.main_ratio or "4:5"
                )

            result = self._evaluate_dance_with_vl(Path(image_path))
            final_score = float(result["score"]) + crop_center_penalty
            final_decision = str(result["decision"])

            rules_hash = self._dance_rules_hash()
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
                "scene_classification": scene_classification,
                "scene_type": scene_classification.get("scene_type", "unknown"),
                "is_group_pose": bool(scene_classification.get("is_group_pose", False)),
                "is_static_pose": bool(scene_classification.get("is_static_pose", False)),
                "should_keep_full_frame": bool(scene_classification.get("should_keep_full_frame", False)),
                "should_avoid_subject_crop": bool(scene_classification.get("should_avoid_subject_crop", False)),
                "scene_reason": str(scene_classification.get("reason", "")),
                "scene_confidence": float(scene_classification.get("confidence", 0.0)),
            }
            self._put_cached_entry(Path(image_path), cache_entry, rules_hash)

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
                "scene_classification": scene_classification,
                "scene_type": scene_classification.get("scene_type", "unknown"),
                "is_group_pose": bool(scene_classification.get("is_group_pose", False)),
                "is_static_pose": bool(scene_classification.get("is_static_pose", False)),
                "should_keep_full_frame": bool(scene_classification.get("should_keep_full_frame", False)),
                "should_avoid_subject_crop": bool(scene_classification.get("should_avoid_subject_crop", False)),
                "scene_reason": str(scene_classification.get("reason", "")),
                "scene_confidence": float(scene_classification.get("confidence", 0.0)),
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
                blur_penalty_threshold=float(config["blur_penalty_threshold"]),
                blur_penalty_points=float(config["blur_penalty_points"]),
            )
            decision, override_reason = self._decision_from_score(
                score,
                focus_score,
                hero is not None,
                keep_threshold=float(config["keep_threshold"]),
                maybe_threshold=float(config["maybe_threshold"]),
                blur_reject_threshold=float(config["blur_reject_threshold"]),
            )

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

    def _extract_capture_timestamp(self, image_path: Path) -> tuple[float, str]:
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
            if exif:
                dt_value = exif.get(36867) or exif.get(36868) or exif.get(306)
                if dt_value:
                    base = str(dt_value).strip()
                    dt = datetime.strptime(base, "%Y:%m:%d %H:%M:%S")
                    frac = 0.0
                    subsec = exif.get(37521)
                    if subsec is not None:
                        digits = "".join(ch for ch in str(subsec) if ch.isdigit())
                        if digits:
                            frac = float(f"0.{digits}")
                    return dt.timestamp() + frac, "exif"
        except Exception:
            pass

        try:
            return float(image_path.stat().st_mtime), "mtime"
        except Exception:
            return 0.0, "none"

    def _build_bursts(self, ordered_paths: list[Path], burst_fps: float) -> list[list[Path]]:
        if not ordered_paths:
            return []

        threshold_sec = 1.0 / max(0.1, float(burst_fps))
        bursts: list[list[Path]] = []
        current: list[Path] = []
        prev_ts: float | None = None
        prev_source: str | None = None
        prev_path: Path | None = None

        for path in ordered_paths:
            ts, source = self._extract_capture_timestamp(Path(path))
            if not current:
                current = [Path(path)]
                prev_ts = ts
                prev_source = source
                prev_path = Path(path)
                continue

            delta = ts - float(prev_ts or 0.0)
            same_burst = False
            if delta >= 0:
                if source == "mtime" and prev_source == "mtime" and delta == 0:
                    # Avoid grouping unrelated files with identical mtime when EXIF is unavailable.
                    same_burst = self._looks_like_sequential_burst_names(prev_path, Path(path))
                else:
                    same_burst = delta <= threshold_sec

            if same_burst:
                current.append(Path(path))
            else:
                bursts.append(current)
                current = [Path(path)]

            prev_ts = ts
            prev_source = source
            prev_path = Path(path)

        if current:
            bursts.append(current)

        return bursts

    def _looks_like_sequential_burst_names(self, previous_path: Path | None, current_path: Path) -> bool:
        if previous_path is None:
            return False

        prev_stem = previous_path.stem
        curr_stem = current_path.stem

        prev_digits = ""
        i = len(prev_stem) - 1
        while i >= 0 and prev_stem[i].isdigit():
            prev_digits = prev_stem[i] + prev_digits
            i -= 1

        curr_digits = ""
        j = len(curr_stem) - 1
        while j >= 0 and curr_stem[j].isdigit():
            curr_digits = curr_stem[j] + curr_digits
            j -= 1

        if not prev_digits or not curr_digits:
            return False

        prev_prefix = prev_stem[: len(prev_stem) - len(prev_digits)]
        curr_prefix = curr_stem[: len(curr_stem) - len(curr_digits)]
        if prev_prefix != curr_prefix:
            return False

        try:
            delta = int(curr_digits) - int(prev_digits)
        except Exception:
            return False
        return 0 < delta <= 3

    def _rank_burst_candidates(self, burst_results: list[dict]) -> list[dict]:
        def key(item: dict) -> tuple[float, float, float, float, float]:
            decision = str(item.get("decision", "Reject"))
            score = float(item.get("score", 0.0))
            hero_focus = float(item.get("hero_focus", 0.0))
            has_face = 1.0 if item.get("has_face", False) else 0.0
            face_focus = float(item.get("face_focus", 0.0))
            return (float(self._decision_rank(decision)), score, hero_focus, has_face, face_focus)

        return sorted(burst_results, key=key, reverse=True)

    def _burst_vl_candidates(self, ranked: list[dict], keep_per_burst: int) -> list[dict]:
        plausible = [r for r in ranked if str(r.get("decision", "Reject")) in {"Keep", "Maybe"}]
        if len(plausible) >= 2:
            return plausible[: self.MAX_VL_BURST_CANDIDATES]

        if len(ranked) <= max(1, keep_per_burst):
            return []

        cutoff = max(1, keep_per_burst)
        if cutoff >= len(ranked):
            return []

        score_a = float(ranked[cutoff - 1].get("score", 0.0))
        score_b = float(ranked[cutoff].get("score", 0.0))
        # Only call VL when the heuristic scores at the keep/reject boundary are close.
        if abs(score_a - score_b) > self.VL_BURST_SCORE_TIE_THRESHOLD:
            return []

        return ranked[: min(self.MAX_VL_BURST_CANDIDATES, cutoff + 2)]

    def _resolve_vl_frame_choice(self, value: str, candidates: list[dict]) -> dict | None:
        candidate_by_name: dict[str, dict] = {}
        candidate_by_path: dict[str, dict] = {}
        for item in candidates:
            path = Path(item["path"])
            candidate_by_name[path.name.lower()] = item
            candidate_by_path[str(path).lower()] = item

        raw = str(value or "").strip()
        if not raw:
            return None

        normalized = raw.lower()
        if normalized in candidate_by_name:
            return candidate_by_name[normalized]
        if normalized in candidate_by_path:
            return candidate_by_path[normalized]

        if ":" in raw:
            tail = raw.split(":", 1)[1].strip().lower()
            if tail in candidate_by_name:
                return candidate_by_name[tail]
            if tail in candidate_by_path:
                return candidate_by_path[tail]

        return None

    def _select_burst_winners_with_vl(
        self,
        ranked: list[dict],
        keep_per_burst: int,
        config: dict,
    ) -> tuple[list[dict] | None, dict]:
        if not bool(config.get("use_vl_burst_tiebreaker", False)):
            return None, {}

        candidates = self._burst_vl_candidates(ranked, keep_per_burst)
        if len(candidates) < 2:
            return None, {}

        try:
            base_url, model, timeout, temperature, max_tokens = self._dance_lmstudio_settings()
            client = LMStudioClient(base_url=base_url, timeout=timeout)
            selection = client.burst_select_frames(
                model=model,
                image_paths=[Path(item["path"]) for item in candidates],
                temperature=max(0.0, min(temperature, 0.4)),
                max_tokens=max(256, min(max_tokens, 600)),
            )
        except Exception as exc:
            self.app.log(f"AI Cull: VL burst tie-breaker unavailable ({exc}); using heuristic ranking.")
            return None, {"error": str(exc)}

        chosen: list[dict] = []
        best = self._resolve_vl_frame_choice(str(selection.get("best_frame", "")), candidates)
        if best is not None:
            chosen.append(best)

        for alt in selection.get("alternates", []) or []:
            alt_item = self._resolve_vl_frame_choice(str(alt), candidates)
            if alt_item is not None and alt_item not in chosen:
                chosen.append(alt_item)

        if not chosen:
            return None, {"error": "VL selector returned no valid frame choice."}

        for fallback in ranked:
            if len(chosen) >= keep_per_burst:
                break
            if fallback not in chosen:
                chosen.append(fallback)

        meta = {
            "best_frame": str(selection.get("best_frame", "")),
            "alternates": list(selection.get("alternates", []) or []),
            "rejects": list(selection.get("rejects", []) or []),
            "reason": str(selection.get("reason", "")),
        }
        try:
            meta["confidence"] = float(selection.get("confidence", 0.0))
        except Exception:
            meta["confidence"] = 0.0

        return chosen[:keep_per_burst], meta

    def apply_burst_suppression_for_pipeline(self, results: list[dict], config: dict) -> list[dict]:
        if not results or not config.get("enable_burst", False):
            return results

        path_to_result = {Path(r["path"]): r for r in results}
        ordered_paths = [Path(r["path"]) for r in results]
        bursts = self._resolve_burst_groups(ordered_paths, config)
        keep_per_burst = max(1, int(config.get("keep_per_burst", 1)))

        for burst_index, burst_paths in enumerate(bursts, start=1):
            burst_results = [path_to_result[p] for p in burst_paths if p in path_to_result]
            if len(burst_results) < 2:
                for item in burst_results:
                    item.update(self._default_burst_metadata_updates())
                continue
            ranked = self._rank_burst_candidates(burst_results)
            has_static_group_pose = any(self._scene_is_static_group_pose(item) for item in burst_results)
            keep_target = keep_per_burst
            if has_static_group_pose and len(burst_results) > 1:
                # Static intro/finale/group tableaux often have multiple usable ensemble variants;
                # keep at least two to avoid over-suppressing composition-preserving frames.
                keep_target = min(len(burst_results), max(keep_per_burst, MIN_STATIC_GROUP_BURST_KEEP))

            winners, vl_meta = self._select_burst_winners_with_vl(ranked, keep_target, config)
            if not winners:
                winners = ranked[:keep_target]
            winner_paths = {Path(item["path"]) for item in winners}
            winner_path_strings = [str(p) for p in sorted(winner_paths, key=lambda p: p.name.lower())]
            vl_used = bool(vl_meta) and "error" not in vl_meta
            group_id = f"burst_{burst_index:04d}"
            rank_by_path = {Path(item["path"]): rank for rank, item in enumerate(ranked, start=1)}

            for item in burst_results:
                item["burst_size"] = len(burst_results)
                item["burst_group_id"] = group_id
                item["burst_rank"] = rank_by_path.get(Path(item["path"]), 0)
                item["burst_winner_paths"] = winner_path_strings
                item["burst_vl_selector_used"] = vl_used
                item["burst_keep_target"] = keep_target
                item["burst_conservative_scene_mode"] = has_static_group_pose
                if vl_meta:
                    item["burst_vl_selector"] = vl_meta
                if Path(item["path"]) not in winner_paths:
                    item["decision"] = "Reject"
                    item["burst_suppressed"] = True
                else:
                    item["burst_suppressed"] = False

        return results

    def _persist_burst_suppression_results(self, results: list[dict]) -> None:
        for item in results:
            image_path = Path(item["path"])
            burst_updates = {
                "burst_group_id": item.get("burst_group_id"),
                "burst_rank": int(item.get("burst_rank", 0)),
                "burst_size": int(item.get("burst_size", 1)),
                "burst_suppressed": bool(item.get("burst_suppressed", False)),
                "burst_winner_paths": [str(p) for p in item.get("burst_winner_paths", []) or []],
                "burst_vl_selector_used": bool(item.get("burst_vl_selector_used", False)),
                "burst_keep_target": int(item.get("burst_keep_target", 1)),
                "burst_conservative_scene_mode": bool(item.get("burst_conservative_scene_mode", False)),
            }
            if item.get("burst_vl_selector"):
                burst_updates["burst_vl_selector"] = dict(item.get("burst_vl_selector", {}))
            self._put_cached_entry(image_path, burst_updates)

    def persist_burst_suppression_results(self, results: list[dict]) -> None:
        self._persist_burst_suppression_results(results)

    def _folder_batch_source_paths(self) -> list[Path]:
        return [Path(p) for p in self._get_loaded_image_paths()]

    def run_burst_suppression_input_folder(self):
        self.evaluate_bursts()

    def on_image_changed(self):
        self._sync_folder_and_loaded_state()
        self._update_vision_warning()
        self.current_score = 0.0
        self.current_decision = "Reject"
        self.current_hero_id = None
        self.current_ball_id = None
        self.current_vl_rubric = None
        self.current_vl_subject_reason = ""
        self.current_vl_debug_image_path = None
        self.current_vl_mismatch = False
        self.current_vl_mismatch_context = None
        self.current_scene_classification = None
        self.app.clear_debug_views()

        self._refresh_dynamic_sections()

        if self.app.current_image is None:
            self.app.set_manual_boxes([])
            self.app.set_manual_selected_ids(set())
            self.app.set_overlays([])
            return

        self._log_current_burst_state()
        current_path = Path(self.app.state.current_image_path)
        result = self.ai_processed_results.get(str(current_path.resolve()))
        if result is None:
            cached = self._get_cached_entry(current_path)
            if "cull_decision" in cached:
                result = {
                    "decision": str(cached.get("cull_decision", "Reject")),
                    "score": float(cached.get("cull_score", 0.0)),
                }
        if result is not None:
            self.current_decision = str(result.get("decision", "Reject"))
            self.current_score = float(result.get("score", 0.0))

    def _run_current_image_analysis(self):
        """Run expensive Florence/VL analysis on the current image.

        Unlike on_image_changed() which is lightweight and runs on every image
        selection, this method performs the heavy object detection and vision
        scoring. It should only be called when the user explicitly requests
        analysis (e.g. via the Rerun Current Image button or Run All).
        """
        if self.app.current_image is None:
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
            scene_classification = self._default_scene_classification()
            if bool(runtime_config.get("use_dance_scene_classifier", False)):
                try:
                    scene_classification = self._classify_scene_with_vl(self.app.state.current_image_path)
                    self.app.log(
                        "AI Cull Dance VL scene: "
                        f"type={scene_classification.get('scene_type', 'unknown')} "
                        f"full_frame={scene_classification.get('should_keep_full_frame', False)} "
                        f"avoid_subject_crop={scene_classification.get('should_avoid_subject_crop', False)} "
                        f"confidence={float(scene_classification.get('confidence', 0.0)):.2f}"
                    )
                except Exception as exc:
                    self.app.log(f"AI Cull Dance VL scene classifier failed: {exc}")

            self.current_scene_classification = scene_classification

            if self._scene_requires_composition_preservation(scene_classification):
                crop_proposal = BoundingBox(0, 0, img_w, img_h) if img_w > 0 and img_h > 0 else None
                crop_center_penalty = 0.0
                self.app.log(
                    "AI Cull Dance: preserving full-frame composition for "
                    f"{scene_classification.get('scene_type', 'unknown')} scene."
                )
            else:
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
                    ("Object Candidates", florence_preview),
                    ("VL Input", vl_input_preview),
                    ("Final Subject", final_preview),
                    ("Original", original_preview),
                ])

            rules_hash = self._dance_rules_hash()
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
                "scene_classification": scene_classification,
                "scene_type": scene_classification.get("scene_type", "unknown"),
                "is_group_pose": bool(scene_classification.get("is_group_pose", False)),
                "is_static_pose": bool(scene_classification.get("is_static_pose", False)),
                "should_keep_full_frame": bool(scene_classification.get("should_keep_full_frame", False)),
                "should_avoid_subject_crop": bool(scene_classification.get("should_avoid_subject_crop", False)),
                "scene_reason": str(scene_classification.get("reason", "")),
                "scene_confidence": float(scene_classification.get("confidence", 0.0)),
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
                    self._put_cached_entry(self.app.state.current_image_path, cache_entry, rules_hash)
                    return
                except Exception as exc:
                    self.app.log(f"AI Cull Dance VL failed, falling back to heuristic cull: {exc}")

            self._put_cached_entry(self.app.state.current_image_path, cache_entry, rules_hash)

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

    def _can_run_ai_cull(self) -> bool:
        if not self.use_object_cull_var.get() and not self.use_vision_cull_var.get():
            self.app.log("AI Cull: enable object and/or vision cull first.")
            return False
        return True

    def _is_burst_rejected(self, path: Path) -> bool:
        return str(Path(path).resolve()) in self.burst_removed_paths

    def _store_ai_result(self, path: Path, result: dict):
        resolved_path = Path(path).resolve()
        key = str(resolved_path)
        decision = str(result.get("decision", "Reject"))
        score = float(result.get("score", 0.0))
        self.ai_processed_results[key] = {"decision": decision, "score": score}
        self._put_cached_entry(Path(path), {"cull_decision": decision, "cull_score": score})
        if self.app.state.current_image_path is not None and Path(self.app.state.current_image_path).resolve() == resolved_path:
            self.current_decision = decision
            self.current_score = score
        self._refresh_accounting_labels()

    def _populate_cached_processed_results(self):
        for path in self.burst_remaining_paths:
            resolved = str(Path(path).resolve())
            if resolved in self.ai_processed_results:
                continue
            cached = self._get_cached_entry(Path(path))
            if "cull_decision" not in cached:
                continue
            self.ai_processed_results[resolved] = {
                "decision": str(cached.get("cull_decision", "Reject")),
                "score": float(cached.get("cull_score", 0.0)),
            }

    def cull_current_by_object(self):
        self.use_object_cull_var.set(True)
        self.use_vision_cull_var.set(False)
        self._refresh_vision_object_controls()
        self.run_current_image()

    def cull_current_by_vision(self):
        if self.vision_warning_var.get().strip():
            self.app.log("AI Cull: load a vision model before running vision cull.")
            return
        self.use_vision_cull_var.set(True)
        self.use_object_cull_var.set(True)
        self._refresh_vision_object_controls()
        self.run_current_image()

    def _set_running_state(self, running: bool, mode: str = ""):
        self.auto_running = running
        self.auto_mode = mode or self.auto_mode
        if self.evaluate_bursts_button is not None:
            self.evaluate_bursts_button.config(state="disabled" if running else "normal")
        if self.run_cull_button is not None:
            self.run_cull_button.config(state="disabled" if running else "normal")
        if self.run_current_button is not None:
            self.run_current_button.config(state="disabled" if running else "normal")
        if self.run_full_button is not None:
            self.run_full_button.config(state="disabled" if running else "normal")
        if self.auto_button is not None:
            self.auto_button.config(state="disabled" if running else "normal")
        if self.stop_button is not None:
            self.stop_button.config(state="normal" if running else "disabled")

    def _start_ai_worker(self, paths: list[Path], mode: str):
        if self.auto_running:
            self.app.log("AI Cull: another job is already running.")
            return
        if not paths:
            self._refresh_accounting_labels()
            self.app.log("AI Cull: no unprocessed images to run.")
            return

        config = self.get_runtime_config()
        self.auto_cancel_requested = False
        self._cancel_event.clear()
        self._set_running_state(True, mode)

        total = len(paths)

        def worker():
            for idx, image_path in enumerate(paths, start=1):
                if self._cancel_event.is_set():
                    self._worker_queue.put(("done", idx - 1, True, total))
                    return
                try:
                    result = self.evaluate_image_for_pipeline(Path(image_path), config)
                    self._worker_queue.put(("result", Path(image_path), result, idx, total))
                except Exception as exc:
                    self._worker_queue.put(("error", Path(image_path), str(exc), idx, total))
            self._worker_queue.put(("done", total, False, total))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
        self.app.root.after(40, self._poll_worker_queue)

    def _poll_worker_queue(self):
        has_pending = False
        while True:
            try:
                event = self._worker_queue.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "result":
                _, image_path, result, idx, total = event
                self._store_ai_result(Path(image_path), result)
                decision = str(result.get("decision", "Reject"))
                score = float(result.get("score", 0.0))
                self.app.log(f"AI Cull {idx}/{total}: {Path(image_path).name} -> {decision} ({score:.1f})")
            elif kind == "error":
                _, image_path, message, idx, total = event
                self.app.log(f"AI Cull {idx}/{total}: failed on {Path(image_path).name} ({message})")
            elif kind == "burst_done":
                _, paths, fps, keep_per_burst, summary, burst_groups, removed_paths, remaining_paths = event
                self._persist_burst_evaluation_cache(paths, burst_groups, removed_paths, keep_per_burst)
                self.precomputed_burst_state = {"folder": str(self.app.state.input_folder.resolve()), "fps": fps, "groups": [[str(p) for p in g] for g in burst_groups]}
                self.burst_analysis_complete = True
                self.burst_summary = summary
                self.burst_paths = {str(Path(p).resolve()) for group in burst_groups for p in group}
                self.burst_removed_paths = removed_paths
                self.burst_remaining_paths = remaining_paths
                self._populate_cached_processed_results()
                self.app.refresh_image_browser()
                self._refresh_accounting_labels()
                self._finish_auto_cull(cancelled=False)
                self.app.log(
                    f"AI Cull Burst Evaluate: burst={summary['burst_images']} removed={summary['burst_images_removed']} "
                    f"remaining={summary['total_remaining_images']} non-burst={summary['non_burst_images']} fps={fps:.2f}"
                )
                if self._pending_full_workflow_after_burst:
                    self._pending_full_workflow_after_burst = False
                    pending = [Path(p) for p in self.burst_remaining_paths if str(Path(p).resolve()) not in self.ai_processed_results]
                    self._start_ai_worker(pending, "run_full_workflow")
            elif kind == "burst_error":
                _, message = event
                self._finish_auto_cull(cancelled=True)
                self._pending_full_workflow_after_burst = False
                self.app.log(f"AI Cull Burst Evaluate failed: {message}")
            elif kind == "done":
                _, processed, cancelled, total = event
                self._finish_auto_cull(cancelled=bool(cancelled))
                if cancelled:
                    self.app.log(f"AI Cull: stopped after {processed}/{total} image(s).")
                else:
                    self.app.log(f"AI Cull: completed {processed}/{total} image(s).")
            else:
                has_pending = True
        if self.auto_running or has_pending:
            self.app.root.after(40, self._poll_worker_queue)

    def run_current_image(self):
        if self.app.state.current_image_path is None:
            self.app.log("AI Cull: no current image selected.")
            return
        if not self._can_run_ai_cull():
            return
        image_path = Path(self.app.state.current_image_path)
        if self.burst_analysis_complete and self._is_burst_rejected(image_path):
            self.app.log("AI Cull: current image is burst-rejected; skipping Run Current Image.")
            return
        self._start_ai_worker([image_path], "run_current_image")

    def run_cull(self):
        if not self._can_run_ai_cull():
            return
        if not self.burst_analysis_complete:
            self.app.log("AI Cull: run Evaluate Bursts first.")
            return
        pending = [Path(p) for p in self.burst_remaining_paths if str(Path(p).resolve()) not in self.ai_processed_results]
        self._start_ai_worker(pending, "run_cull")

    def run_full_workflow(self):
        if not self._can_run_ai_cull():
            return
        if not self.burst_analysis_complete:
            self._pending_full_workflow_after_burst = True
            self.evaluate_bursts()
            return
        pending = [Path(p) for p in self.burst_remaining_paths if str(Path(p).resolve()) not in self.ai_processed_results]
        self._start_ai_worker(pending, "run_full_workflow")

    def rerun(self):
        self.run_current_image()

    def _decision_bucket_name(self, decision: str, burst_suppressed: bool = False) -> str:
        if burst_suppressed:
            return "Reject-Bursts"
        mapped = str(decision or "Reject").strip()
        if mapped not in {"Keep", "Maybe", "Reject"}:
            mapped = "Reject"
        return mapped

    def _get_cull_output_dir(self, decision: str, burst_suppressed: bool = False) -> Path | None:
        if self.app.state.input_folder is None:
            return None
        mapped = self._decision_bucket_name(decision, burst_suppressed=burst_suppressed)
        out_dir = self.app.state.input_folder / "Output" / mapped
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _copy_image_to_decision_folder(self, source: Path, decision: str, score: float, burst_suppressed: bool = False):
        bucket = self._decision_bucket_name(decision, burst_suppressed=burst_suppressed)
        output_dir = self._get_cull_output_dir(decision, burst_suppressed=burst_suppressed)
        if output_dir is None:
            self.app.log("AI Cull: no input folder selected.")
            return

        destination = output_dir / source.name
        shutil.copy2(source, destination)
        self.app.log(f"AI Cull: copied {source.name} -> {bucket} ({score:.1f})")

    def stop_auto_cull(self):
        if not self.auto_running:
            return
        self.auto_cancel_requested = True
        self._cancel_event.set()
        self.app.log("AI Cull: stop requested.")

    def auto_cull_input_folder(self):
        self.run_full_workflow()

    def _auto_cull_step(self):
        return

    def _finish_auto_cull(self, cancelled: bool):
        self._set_running_state(False)
        self.auto_cancel_requested = False
        self._cancel_event.clear()
        self.auto_mode = "auto_cull"
        self.auto_all_images = []
        self.auto_precomputed_bursts = []

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