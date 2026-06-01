from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib import error, request

from PIL import Image


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
        max_side: int = 1536,
        jpeg_quality: int = 90,
    ) -> str:
        path = Path(image_path)

        with Image.open(path) as img:
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

    def test_connection(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
            return True, f"Connected successfully. {len(models)} model(s) available."
        except Exception as exc:
            return False, f"Connection failed: {exc}"