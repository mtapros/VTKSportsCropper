from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib import error, request

from PIL import Image, ImageOps


class LMStudioClient:
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

    def _image_content_item(
        self,
        image_path: str | Path,
        max_side: int = 1024,
        jpeg_quality: int = 90,
    ) -> dict:
        return {
            "type": "image_url",
            "image_url": {
                "url": self._image_file_to_data_url(
                    image_path,
                    max_side=max_side,
                    jpeg_quality=jpeg_quality,
                ),
            },
        }

    def vision_chat(
        self,
        model: str,
        user_content: str | list[dict],
        system_prompt: str = "You are a concise sports photography vision assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> dict:
        if isinstance(user_content, str):
            content = [{"type": "text", "text": user_content}]
        else:
            content = user_content

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
        return self._http_post_json("/chat/completions", payload)

    def vision_chat_text(
        self,
        model: str,
        image_path: str | Path,
        user_prompt: str,
        system_prompt: str = "You are a concise sports photography vision assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        data = self.vision_chat(
            model=model,
            user_content=[
                {
                    "type": "text",
                    "text": user_prompt,
                },
                self._image_content_item(image_path),
            ],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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

    def select_burst_best_frame(
        self,
        model: str,
        frames: list[dict],
        rubric_name: str = "generic",
        temperature: float = 0.1,
        max_tokens: int = 700,
    ) -> dict:
        if not frames:
            raise ValueError("No burst frames were provided.")

        emphasis = (
            "Prioritize the cleanest dance recital delivery frame: sharp subject, flattering pose, usable face angle, "
            "full-body visibility, and uncropped feet/hands when possible."
            if str(rubric_name).strip().lower() == "dance"
            else "Prioritize the strongest keeper frame: sharpness, expression, moment, subject visibility, and clean composition."
        )

        system_prompt = (
            "You are comparing near-duplicate frames from the same sports photo burst.\n"
            "These images are already grouped heuristically as a burst, so do not regroup them.\n"
            "Choose the single best keeper frame from the provided options.\n"
            "Return only valid JSON.\n"
            "Do not use markdown.\n"
            "Do not include code fences.\n"
            "Do not include any text before or after the JSON.\n\n"
            "Use exactly these keys:\n"
            "- best_frame\n"
            "- alternates\n"
            "- rejects\n"
            "- confidence\n"
            "- reason\n\n"
            "Rules:\n"
            "- best_frame must be one of the provided frame_id values.\n"
            "- alternates must be an array of zero or more remaining frame_id values, ordered best to worst.\n"
            "- rejects must be an array of zero or more remaining frame_id values.\n"
            "- confidence must be a number between 0 and 1.\n"
            "- reason must be one short sentence.\n"
            "- Use the heuristic metadata only as a hint when the visual differences are subtle.\n"
        )

        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Compare these burst frames and choose the best single keeper.\n"
                    f"{emphasis}\n"
                    "If there are strong backups, list them in alternates.\n"
                    "If a frame is clearly weaker, list it in rejects.\n"
                    "Return only valid JSON."
                ),
            }
        ]

        for frame in frames:
            frame_id = str(frame.get("frame_id", "")).strip()
            if not frame_id:
                raise ValueError("Each burst frame must include a frame_id.")
            image_path = frame.get("image_path")
            if not image_path:
                raise ValueError(f"Burst frame {frame_id!r} is missing image_path.")
            filename = str(frame.get("filename", "")).strip() or Path(image_path).name
            decision = str(frame.get("decision", "")).strip() or "Unknown"
            score = float(frame.get("heuristic_score", 0.0))
            focus_score = float(frame.get("focus_score", 0.0))
            face_visible = bool(frame.get("face_visible", False))
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"{frame_id}: {filename}\n"
                        f"- heuristic_decision: {decision}\n"
                        f"- heuristic_score: {score:.1f}\n"
                        f"- focus_score: {focus_score:.1f}\n"
                        f"- face_visible: {'yes' if face_visible else 'no'}"
                    ),
                }
            )
            content.append(self._image_content_item(image_path, max_side=768, jpeg_quality=85))

        data = self.vision_chat(
            model=model,
            user_content=content,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Empty burst selection response")
        message = choices[0].get("message", {})
        content_text = message.get("content", "")
        if isinstance(content_text, list):
            text_parts = []
            for item in content_text:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content_text = "\n".join(part for part in text_parts if part)
        return self._extract_json_object(str(content_text))

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

    def test_connection(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
            return True, f"Connected successfully. {len(models)} model(s) available."
        except Exception as exc:
            return False, f"Connection failed: {exc}"