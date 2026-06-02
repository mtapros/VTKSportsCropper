from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog
from tkinter import ttk

from lmstudio_client import LMStudioClient


class LMStudioTool:
    tool_id = "lmstudio"
    display_name = "LM Studio Test"
    SETTINGS_WINDOW_GEOMETRY = "560x520"

    def __init__(self, app):
        self.app = app
        self.panel = None

        self.base_url_var = tk.StringVar(value="http://127.0.0.1:1234/v1")
        self.model_var = tk.StringVar(value="")
        self.timeout_var = tk.StringVar(value="60")
        self.temperature_var = tk.StringVar(value="0.2")
        self.max_tokens_var = tk.StringVar(value="512")

        self.system_prompt_var = tk.StringVar(value="You are a concise local assistant.")
        self.user_prompt_var = tk.StringVar(value="Reply with: LM Studio test successful.")

        self.cull_filename_var = tk.StringVar(value="IMG_0001.JPG")
        self.cull_focus_var = tk.StringVar(value="24.5")
        self.cull_af_match_var = tk.BooleanVar(value=True)
        self.cull_has_face_var = tk.BooleanVar(value=True)
        self.cull_has_ball_var = tk.BooleanVar(value=False)
        self.cull_score_var = tk.StringVar(value="81.0")

        self.vision_instruction_var = tk.StringVar(
            value="Evaluate this image for culling and return only valid JSON."
        )

        self.model_combo = None
        self.response_box = None

        self.good_criteria_box = None
        self.reject_criteria_box = None
        self.vision_schema_box = None
        self.preset_combo = None
        self.settings_window = None

        self.active_preset_var = tk.StringVar(value="default")
        self.presets: dict[str, dict] = {}
        self._load_presets_file()

    def _preset_file(self) -> Path:
        return Path(self.app.base_dir) / "lmstudio_presets.json"

    def _default_settings(self) -> dict:
        return {
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "",
            "timeout": "60",
            "temperature": "0.2",
            "max_tokens": "512",
        }

    def _collect_settings(self) -> dict:
        return {
            "base_url": self.base_url_var.get().strip(),
            "model": self.model_var.get().strip(),
            "timeout": self.timeout_var.get().strip() or "60",
            "temperature": self.temperature_var.get().strip() or "0.2",
            "max_tokens": self.max_tokens_var.get().strip() or "512",
        }

    def _apply_settings(self, settings: dict):
        defaults = self._default_settings()
        merged = dict(defaults)
        if isinstance(settings, dict):
            for key in defaults:
                value = settings.get(key, defaults[key])
                merged[key] = str(value) if value is not None else defaults[key]
        self.base_url_var.set(merged["base_url"])
        self.model_var.set(merged["model"])
        self.timeout_var.set(merged["timeout"])
        self.temperature_var.set(merged["temperature"])
        self.max_tokens_var.set(merged["max_tokens"])

    def _load_presets_file(self):
        defaults = self._default_settings()
        active = "default"
        loaded_presets: dict[str, dict] = {"default": dict(defaults)}
        path = self._preset_file()
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    maybe_presets = raw.get("presets", {})
                    if isinstance(maybe_presets, dict):
                        parsed: dict[str, dict] = {}
                        for name, entry in maybe_presets.items():
                            if not isinstance(entry, dict):
                                continue
                            parsed[str(name)] = {
                                "base_url": str(entry.get("base_url", defaults["base_url"])),
                                "model": str(entry.get("model", defaults["model"])),
                                "timeout": str(entry.get("timeout", defaults["timeout"])),
                                "temperature": str(entry.get("temperature", defaults["temperature"])),
                                "max_tokens": str(entry.get("max_tokens", defaults["max_tokens"])),
                            }
                        if parsed:
                            loaded_presets = parsed
                    active_value = str(raw.get("active_preset", "default")).strip()
                    if active_value:
                        active = active_value
        except Exception as exc:
            self.app.log(f"LM Studio: failed loading presets ({exc}); using defaults.")

        if "default" not in loaded_presets:
            loaded_presets["default"] = dict(defaults)

        if active not in loaded_presets:
            active = "default"

        self.presets = loaded_presets
        self.active_preset_var.set(active)
        self._apply_settings(self.presets.get(active, defaults))

    def _save_presets_file(self):
        path = self._preset_file()
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "active_preset": self.active_preset_var.get().strip() or "default",
                        "presets": self.presets,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            self.app.log(f"LM Studio: failed saving presets ({exc}).")

    def _refresh_preset_controls(self):
        names = sorted(self.presets.keys(), key=lambda n: (n != "default", n.lower()))
        if not names:
            names = ["default"]
            self.presets["default"] = self._default_settings()
        if self.preset_combo is not None:
            self.preset_combo["values"] = names
        active = self.active_preset_var.get().strip() or "default"
        if active not in self.presets:
            active = names[0]
            self.active_preset_var.set(active)

    def _save_active_preset(self):
        name = self.active_preset_var.get().strip() or "default"
        self.presets[name] = self._collect_settings()
        self.active_preset_var.set(name)
        self._refresh_preset_controls()
        self._save_presets_file()
        self.app.log(f'LM Studio: saved preset "{name}".')

    def _save_preset_as(self):
        name = simpledialog.askstring("Save LM Studio Preset", "Preset name:", parent=self.app.root)
        if not name:
            return
        clean = name.strip()
        if not clean:
            return
        self.active_preset_var.set(clean)
        self._save_active_preset()

    def _load_active_preset(self):
        name = self.active_preset_var.get().strip() or "default"
        preset = self.presets.get(name)
        if not isinstance(preset, dict):
            self.app.log(f'LM Studio: preset "{name}" not found.')
            return
        self._apply_settings(preset)
        self.active_preset_var.set(name)
        self._save_presets_file()
        self.app.log(f'LM Studio: loaded preset "{name}".')

    def _build_settings_controls(self, parent):
        pad = {"padx": 10, "pady": 4}

        preset_row = tk.Frame(parent, bg="#2a2a2a")
        preset_row.pack(fill="x", padx=10, pady=(4, 4))
        tk.Label(preset_row, text="Preset", bg="#2a2a2a", fg="white").pack(side=tk.LEFT)
        self.preset_combo = ttk.Combobox(
            preset_row,
            textvariable=self.active_preset_var,
            values=[],
            state="normal",
            width=24,
        )
        self.preset_combo.pack(side=tk.LEFT, fill="x", expand=True, padx=(8, 6))
        tk.Button(preset_row, text="Load", command=self._load_active_preset).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(preset_row, text="Save", command=self._save_active_preset).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(preset_row, text="Save As", command=self._save_preset_as).pack(side=tk.LEFT)
        self._refresh_preset_controls()

        tk.Label(parent, text="Server URL", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(parent, textvariable=self.base_url_var).pack(fill="x", **pad)

        tk.Label(parent, text="Model", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        self.model_combo = ttk.Combobox(parent, textvariable=self.model_var, values=[], state="normal")
        self.model_combo.pack(fill="x", **pad)

        tk.Label(parent, text="Timeout (seconds)", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(parent, textvariable=self.timeout_var).pack(fill="x", **pad)

        tk.Label(parent, text="Temperature", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(parent, textvariable=self.temperature_var).pack(fill="x", **pad)

        tk.Label(parent, text="Max Tokens", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(parent, textvariable=self.max_tokens_var).pack(fill="x", **pad)

        tk.Button(parent, text="Test Connection", command=self.test_connection).pack(fill="x", padx=10, pady=(10, 4))
        tk.Button(parent, text="Refresh Models", command=self.refresh_models).pack(fill="x", padx=10, pady=4)

    def open_settings_window(self):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.deiconify()
            self.settings_window.lift()
            self.settings_window.focus_force()
            return

        self.settings_window = tk.Toplevel(self.app.root)
        self.settings_window.title("LM Studio Settings")
        self.settings_window.geometry(self.SETTINGS_WINDOW_GEOMETRY)
        self.settings_window.configure(bg="#2a2a2a")

        holder = tk.Frame(self.settings_window, bg="#2a2a2a")
        holder.pack(fill="both", expand=True, padx=10, pady=10)
        self._build_settings_controls(holder)

        tk.Button(
            holder,
            text="Close",
            command=self.settings_window.destroy,
            bg="#555555",
            fg="white",
        ).pack(fill="x", padx=10, pady=(12, 6))

    def build_panel(self, parent):
        self.panel = tk.Frame(parent, bg="#2a2a2a")
        pad = {"padx": 10, "pady": 4}

        tk.Label(
            self.panel,
            text="LM Studio Test",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", **pad)

        self._build_settings_controls(self.panel)

        tk.Label(self.panel, text="System Prompt", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.system_prompt_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="User Prompt", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.user_prompt_var).pack(fill="x", **pad)

        tk.Button(self.panel, text="Send Plain Prompt", command=self.send_plain_prompt).pack(fill="x", padx=10, pady=4)

        tk.Label(
            self.panel,
            text="Cull Decision JSON Test",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=10, pady=(14, 4))

        tk.Label(self.panel, text="Filename", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.cull_filename_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="Focus Score", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.cull_focus_var).pack(fill="x", **pad)

        tk.Label(self.panel, text="Heuristic Score", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.cull_score_var).pack(fill="x", **pad)

        tk.Checkbutton(
            self.panel,
            text="AF Match",
            variable=self.cull_af_match_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        tk.Checkbutton(
            self.panel,
            text="Visible Face",
            variable=self.cull_has_face_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        tk.Checkbutton(
            self.panel,
            text="Ball Present",
            variable=self.cull_has_ball_var,
            bg="#2a2a2a",
            fg="white",
            selectcolor="#444",
        ).pack(anchor="w", **pad)

        tk.Button(self.panel, text="Send Cull JSON Test", command=self.send_cull_json_test).pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            self.panel,
            text="Vision JSON Test",
            bg="#2a2a2a",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=10, pady=(14, 4))

        tk.Label(self.panel, text="Vision Instruction", bg="#2a2a2a", fg="white").pack(anchor="w", **pad)
        tk.Entry(self.panel, textvariable=self.vision_instruction_var).pack(fill="x", **pad)

        tk.Label(
            self.panel,
            text='Good Criteria (one per line, e.g. "sharp face", "strong moment", "good composition")',
            bg="#2a2a2a",
            fg="white",
        ).pack(anchor="w", padx=10, pady=(8, 2))
        self.good_criteria_box = tk.Text(self.panel, height=6, bg="#111111", fg="white", insertbackground="white", wrap="word")
        self.good_criteria_box.pack(fill="x", padx=10, pady=(0, 6))
        self.good_criteria_box.insert(
            "1.0",
            "sharp subject\nclear face\nstrong action or performance moment\ngood composition\nusable expression or pose",
        )

        tk.Label(
            self.panel,
            text='Reject / Weak Criteria (one per line, e.g. "motion blur", "bad crop", "missed focus")',
            bg="#2a2a2a",
            fg="white",
        ).pack(anchor="w", padx=10, pady=(8, 2))
        self.reject_criteria_box = tk.Text(self.panel, height=6, bg="#111111", fg="#ffdddd", insertbackground="white", wrap="word")
        self.reject_criteria_box.pack(fill="x", padx=10, pady=(0, 6))
        self.reject_criteria_box.insert(
            "1.0",
            "soft or missed focus\nawkward pose\nbad timing\nweak composition\nsubject obscured",
        )

        tk.Label(
            self.panel,
            text="Requested JSON Keys (one per line)",
            bg="#2a2a2a",
            fg="white",
        ).pack(anchor="w", padx=10, pady=(8, 2))
        self.vision_schema_box = tk.Text(self.panel, height=6, bg="#111111", fg="#ddddff", insertbackground="white", wrap="word")
        self.vision_schema_box.pack(fill="x", padx=10, pady=(0, 6))
        self.vision_schema_box.insert(
            "1.0",
            "decision\nconfidence\nreason\nsharpness\nsubject_visibility\nmoment_quality\ncomposition",
        )

        tk.Button(
            self.panel,
            text="Send Current Image Vision JSON Test",
            command=self.send_current_image_vision_test,
        ).pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(self.panel, text="Response", bg="#2a2a2a", fg="white").pack(anchor="w", padx=10, pady=(10, 4))
        self.response_box = tk.Text(
            self.panel,
            height=18,
            bg="#111111",
            fg="#DDFFDD",
            insertbackground="white",
            wrap="word",
        )
        self.response_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        return self.panel

    def apply_profile(self, profile):
        pass

    def on_image_changed(self):
        pass

    def _client(self) -> LMStudioClient:
        try:
            timeout = float(self.timeout_var.get().strip() or "60")
        except Exception:
            timeout = 60.0
        return LMStudioClient(self.base_url_var.get().strip(), timeout=timeout)

    def _temperature(self) -> float:
        try:
            return float(self.temperature_var.get().strip() or "0.2")
        except Exception:
            return 0.2

    def _max_tokens(self) -> int:
        try:
            return int(self.max_tokens_var.get().strip() or "512")
        except Exception:
            return 512

    def _set_response_text(self, text: str):
        if self.response_box is None:
            return
        self.response_box.delete("1.0", "end")
        self.response_box.insert("1.0", text)

    def _get_text_lines(self, box: tk.Text | None) -> list[str]:
        if box is None:
            return []
        raw = box.get("1.0", "end").strip()
        if not raw:
            return []
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def _ensure_model_selected(self) -> str | None:
        model = self.model_var.get().strip()
        if model:
            return model

        try:
            models = self._client().list_models()
            if self.model_combo is not None:
                self.model_combo["values"] = models

            if len(models) == 1:
                self.model_var.set(models[0])
                return models[0]

            if len(models) > 1:
                self.app.log("LM Studio: multiple models available; please select one.")
                return None

            self.app.log("LM Studio: no models available.")
            return None
        except Exception as exc:
            self.app.log(f"LM Studio: failed to resolve model: {exc}")
            return None

    def test_connection(self):
        ok, message = self._client().test_connection()
        self.app.log(f"LM Studio: {message}")
        self._set_response_text(message)

    def refresh_models(self):
        try:
            models = self._client().list_models()
            if self.model_combo is not None:
                self.model_combo["values"] = models

            if models and not self.model_var.get().strip():
                self.model_var.set(models[0])

            if models:
                text = "Available models:\n" + "\n".join(models)
                self.app.log("LM Studio: models refreshed.")
                self._set_response_text(text)
            else:
                self.app.log("LM Studio: no models returned by server.")
                self._set_response_text("No models returned by server.")
        except Exception as exc:
            self.app.log(f"LM Studio: failed to refresh models: {exc}")
            self._set_response_text(str(exc))

    def send_plain_prompt(self):
        model = self._ensure_model_selected()
        if not model:
            return

        try:
            self.app.log(f"LM Studio: sending plain prompt to {model}...")
            content = self._client().simple_chat_text(
                model=model,
                system_prompt=self.system_prompt_var.get().strip(),
                user_prompt=self.user_prompt_var.get().strip(),
                temperature=self._temperature(),
                max_tokens=self._max_tokens(),
            )
            self.app.log("LM Studio: plain prompt completed.")
            self._set_response_text(content or "[empty response]")
        except Exception as exc:
            self.app.log(f"LM Studio: plain prompt failed: {exc}")
            self._set_response_text(str(exc))

    def send_cull_json_test(self):
        model = self._ensure_model_selected()
        if not model:
            return

        try:
            focus = float(self.cull_focus_var.get().strip() or "0")
        except Exception:
            focus = 0.0

        try:
            score = float(self.cull_score_var.get().strip() or "0")
        except Exception:
            score = 0.0

        payload = {
            "filename": self.cull_filename_var.get().strip(),
            "focus_score": focus,
            "heuristic_score": score,
            "af_match": bool(self.cull_af_match_var.get()),
            "has_face": bool(self.cull_has_face_var.get()),
            "has_ball": bool(self.cull_has_ball_var.get()),
        }

        system_prompt = (
            "You are assisting with sports photo culling.\n"
            "Return only valid JSON.\n"
            "Do not use markdown.\n"
            "Do not include code fences.\n"
            "Do not include any text before or after the JSON.\n\n"
            "Required JSON keys:\n"
            "- decision\n"
            "- confidence\n"
            "- reason\n\n"
            "Rules:\n"
            '- decision must be one of: "Keep", "Maybe", "Reject"\n'
            "- confidence must be a number between 0 and 1\n"
            "- reason must be a short single sentence\n"
        )

        user_prompt = (
            "Evaluate this image summary for photo culling.\n"
            "Return only valid JSON.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

        try:
            self.app.log(f"LM Studio: sending cull JSON test to {model}...")
            content = self._client().simple_chat_text(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature(),
                max_tokens=self._max_tokens(),
            )
            self.app.log("LM Studio: cull JSON test completed.")
            self._set_response_text(content or "[empty response]")
        except Exception as exc:
            self.app.log(f"LM Studio: cull JSON test failed: {exc}")
            self._set_response_text(str(exc))

    def _build_vision_json_system_prompt(self) -> str:
        requested_keys = self._get_text_lines(self.vision_schema_box)
        good_criteria = self._get_text_lines(self.good_criteria_box)
        reject_criteria = self._get_text_lines(self.reject_criteria_box)

        if not requested_keys:
            requested_keys = ["decision", "confidence", "reason"]

        lines = [
            "You are a sports and performance photo culling assistant.",
            "Return only valid JSON.",
            "Do not use markdown.",
            "Do not include code fences.",
            "Do not include any text before or after the JSON.",
            "",
            "Required JSON keys:",
        ]
        lines.extend([f"- {key}" for key in requested_keys])

        lines.extend(
            [
                "",
                "Rules:",
                '- decision must be one of: "Keep", "Maybe", "Reject"',
                "- confidence must be a number between 0 and 1",
                "- reason must be a short single sentence",
            ]
        )

        if "sharpness" in requested_keys:
            lines.append('- sharpness should be one of: "sharp", "acceptable", "soft", "blurry"')
        if "subject_visibility" in requested_keys:
            lines.append('- subject_visibility should be one of: "strong", "good", "partial", "weak"')
        if "moment_quality" in requested_keys:
            lines.append('- moment_quality should be one of: "strong", "good", "average", "weak"')
        if "composition" in requested_keys:
            lines.append('- composition should be one of: "strong", "good", "average", "weak"')

        if good_criteria:
            lines.extend(["", "Good / keeper criteria to value highly:"])
            lines.extend([f"- {item}" for item in good_criteria])

        if reject_criteria:
            lines.extend(["", "Weak / reject criteria to penalize:"])
            lines.extend([f"- {item}" for item in reject_criteria])

        return "\n".join(lines)

    def _build_vision_json_user_prompt(self) -> str:
        return (
            f"{self.vision_instruction_var.get().strip()}\n\n"
            "Evaluate the image using the provided criteria.\n"
            "Return only valid JSON with the required keys."
        )

    def send_current_image_vision_test(self):
        model = self._ensure_model_selected()
        if not model:
            return

        image_path = self.app.state.current_image_path
        if image_path is None:
            self.app.log("LM Studio: no current image loaded for vision test.")
            self._set_response_text("No current image loaded.")
            return

        system_prompt = self._build_vision_json_system_prompt()
        user_prompt = self._build_vision_json_user_prompt()

        try:
            self.app.log(f"LM Studio: sending current image to {model} for vision JSON test...")
            content = self._client().vision_chat_text(
                model=model,
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature(),
                max_tokens=self._max_tokens(),
            )
            self.app.log("LM Studio: vision JSON test completed.")
            self._set_response_text(content or "[empty response]")
        except Exception as exc:
            self.app.log(f"LM Studio: vision JSON test failed: {exc}")
            self._set_response_text(str(exc))