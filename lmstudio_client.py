from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib import error, request

from PIL import Image, ImageOps


class LMStudioClient:
    SCENE_TYPES = {"intro_pose", "finale_pose", "group_static_pose", "action", "unknown"}

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = float(timeout)

    def _http_get_json(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} GET {path} failed: {body}") from exc

    def _http_post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} POST {path} failed: {body}") from exc

    def list_models(self) -> list[str]:
        data = self._http_get_json("/models")
        return [item.get("id", "") for item in data.get("data", []) if item.get("id")]

    def chat(self, model: str, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        return self._http_post_json("/chat/completions", payload)

    def simple_chat_text(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        data = self.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def _image_file_to_data_url(
        self,
        image_path: str | Path,
        max_side: int = 1024,
        jpeg_quality: int = 90,
    ) -> str:
        path = Path(image_path)

        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            w, h = img.size
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / float(longest)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded}"

    def vision_chat_text(
        self,
        model: str,
        image_path: str | Path,
        user_prompt: str,
        system_prompt: str = "You are a concise sports photography vision assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        data_url = self._image_file_to_data_url(image_path)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            },
                        },
                    ],
                },
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        data = self._http_post_json("/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(p for p in parts if p)
        return json.dumps(content, indent=2)

    def vision_chat_text_multi(
        self,
        model: str,
        image_paths: list[str | Path],
        user_prompt: str,
        system_prompt: str = "You are a concise sports photography vision assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        if not image_paths:
            return ""

        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for idx, image_path in enumerate(image_paths, start=1):
            content.append({"type": "text", "text": f"Frame {idx}: {Path(image_path).name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_file_to_data_url(image_path)},
                }
            )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        data = self._http_post_json("/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        response_content = message.get("content", "")
        if isinstance(response_content, str):
            return response_content
        if isinstance(response_content, list):
            parts = []
            for item in response_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(p for p in parts if p)
        return json.dumps(response_content, indent=2)

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

        raise ValueError("No valid JSON object found in response")

    def dance_cull_rubric(
        self,
        model: str,
        image_path: str | Path,
        temperature: float = 0.1,
        max_tokens: int = 700,
    ) -> dict:
        system_prompt = (
            "You are a dance recital photo culling assistant.\n\n"
            "Evaluate the image for keeper quality for dance recital delivery.\n\n"
            "Return only valid JSON.\n"
            "Do not use markdown.\n"
            "Do not include code fences.\n"
            "Do not include any text before or after the JSON.\n\n"
            "Use exactly these keys:\n"
            "- sharpness\n"
            "- subject_visibility\n"
            "- face_visibility\n"
            "- face_facing_camera\n"
            "- full_body_visibility\n"
            "- feet_visibility\n"
            "- hands_visibility\n"
            "- pose_quality\n"
            "- moment_quality\n"
            "- composition_quality\n"
            "- background_distraction\n"
            "- subject_separation\n"
            "- overall_dance_keeper\n"
            "- confidence\n"
            "- summary\n\n"
            "Allowed values:\n"
            "- sharpness: strong, acceptable, soft, blurry\n"
            "- subject_visibility: strong, good, partial, weak\n"
            "- face_visibility: clear, partial, not_visible\n"
            "- face_facing_camera: yes, partial, no, unknown\n"
            "- full_body_visibility: full, mostly_full, partial\n"
            "- feet_visibility: fully_visible, partially_cropped, cropped_out\n"
            "- hands_visibility: fully_visible, partially_cropped, cropped_out\n"
            "- pose_quality: strong, good, awkward, unclear\n"
            "- moment_quality: strong, good, average, weak\n"
            "- composition_quality: strong, good, average, weak\n"
            "- background_distraction: low, moderate, high\n"
            "- subject_separation: good, somewhat_clear, poor\n"
            "- overall_dance_keeper: keep, maybe, reject\n\n"
            "Rules:\n"
            "- confidence must be a number between 0 and 1\n"
            "- summary must be one short sentence\n"
            "- Be conservative and practical for dance recital photo delivery.\n"
            "- If the visible face is turned away or strongly sideways, set face_facing_camera to no.\n"
            "- If the face is only somewhat toward camera, set face_facing_camera to partial.\n"
        )

        user_prompt = (
            "Evaluate this dance recital image using the required rubric.\n\n"
            "Focus on:\n"
            "- whether the dancer is sharp and clearly visible\n"
            "- whether the face is visible\n"
            "- whether the face is facing the camera\n"
            "- whether the full body, feet, and hands are visible\n"
            "- whether the pose is clean and aesthetically usable\n"
            "- whether the moment is strong enough to keep\n"
            "- whether the composition is clean enough for delivery\n\n"
            "Return only valid JSON."
        )

        text = self.vision_chat_text(
            model=model,
            image_path=image_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract_json_object(text)

    def burst_select_frames(
        self,
        model: str,
        image_paths: list[str | Path],
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> dict:
        if not image_paths:
            raise ValueError("No burst frames provided")

        frame_names = [Path(p).name for p in image_paths]
        frame_list_text = "\n".join(f"- {name}" for name in frame_names)

        system_prompt = (
            "You are a sports burst-frame selector.\n"
            "You are given multiple near-duplicate frames from one burst.\n"
            "Choose the best deliverable frame and optional alternates.\n\n"
            "Return only valid JSON (no markdown, no prose outside JSON) with exactly these keys:\n"
            "- best_frame\n"
            "- alternates\n"
            "- rejects\n"
            "- confidence\n"
            "- reason\n\n"
            "Rules:\n"
            "- best_frame must be one of the provided filenames.\n"
            "- alternates and rejects must be arrays of provided filenames (can be empty).\n"
            "- confidence must be a number between 0 and 1.\n"
            "- reason must be one short sentence.\n"
            "- Prefer the sharpest, cleanest, strongest-timing frame.\n"
            "- Prefer clear subject visibility and minimal motion blur.\n"
            "- If unsure, choose the frame that is most reliably usable."
        )

        user_prompt = (
            "Select the best frame from this burst and optional alternates.\n"
            "Only use filenames from this list:\n"
            f"{frame_list_text}"
        )

        text = self.vision_chat_text_multi(
            model=model,
            image_paths=image_paths,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract_json_object(text)

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "y", "on"}

    def classify_scene_type(
        self,
        model: str,
        image_path: str | Path,
        temperature: float = 0.1,
        max_tokens: int = 450,
    ) -> dict:
        system_prompt = (
            "You are a dance recital scene-type classifier.\n\n"
            "Classify the image into one of these scene_type values:\n"
            "- intro_pose\n"
            "- finale_pose\n"
            "- group_static_pose\n"
            "- action\n"
            "- unknown\n\n"
            "Return ONLY valid JSON (no markdown, no extra text) with exactly these keys:\n"
            "- scene_type\n"
            "- is_group_pose\n"
            "- is_static_pose\n"
            "- should_keep_full_frame\n"
            "- should_avoid_subject_crop\n"
            "- reason\n"
            "- confidence\n\n"
            "Rules:\n"
            "- confidence must be a number between 0 and 1.\n"
            "- reason must be one short sentence.\n"
            "- If the whole ensemble/stage composition matters, set should_keep_full_frame=true.\n"
            "- For intro/finale/group static tableau scenes, avoid tight subject crops.\n"
            "- If unsure, use scene_type='unknown'."
        )

        user_prompt = (
            "Classify whether this dance recital frame is intro pose, finale pose, group static pose, or action.\n"
            "Return only the required JSON."
        )

        text = self.vision_chat_text(
            model=model,
            image_path=image_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = self._extract_json_object(text)

        scene_type = str(parsed.get("scene_type", "unknown")).strip().lower()
        if scene_type not in self.SCENE_TYPES:
            scene_type = "unknown"

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "scene_type": scene_type,
            "is_group_pose": self._to_bool(parsed.get("is_group_pose", False)),
            "is_static_pose": self._to_bool(parsed.get("is_static_pose", False)),
            "should_keep_full_frame": self._to_bool(parsed.get("should_keep_full_frame", False)),
            "should_avoid_subject_crop": self._to_bool(parsed.get("should_avoid_subject_crop", False)),
            "reason": str(parsed.get("reason", "")).strip(),
            "confidence": confidence,
        }

    def test_connection(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
            return True, f"Connected successfully. {len(models)} model(s) available."
        except Exception as exc:
            return False, f"Connection failed: {exc}"