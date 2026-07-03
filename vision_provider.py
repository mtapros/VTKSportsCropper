"""Provider-agnostic vision model clients.

Supports local models served by LM Studio as well as frontier models
reached over HTTPS APIs. All clients expose the same high-level interface
as ``LMStudioClient`` (``chat``, ``vision_chat_text``, ``dance_cull_rubric``,
``burst_select_frames``, ``classify_scene_type`` …) by translating the
OpenAI-style message format used internally into each provider's wire format.

Providers:
- ``lmstudio``            local LM Studio server (OpenAI-compatible, no key)
- ``openai``              OpenAI API (api.openai.com)
- ``openai_compatible``   any OpenAI-compatible endpoint (OpenRouter, xAI,
                          Mistral, vLLM, …) — supply base URL + key
- ``anthropic``           Anthropic Messages API
- ``gemini``              Google Gemini generateContent API

API keys may be given literally, as ``env:VAR_NAME`` to read an environment
variable, or left blank to fall back to the provider's default environment
variable (e.g. ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import base64
import os

from lmstudio_client import LMStudioClient

PROVIDERS: dict[str, dict] = {
    "lmstudio": {
        "label": "LM Studio (local)",
        "default_base_url": "http://127.0.0.1:1234/v1",
        "key_env": "",
        "requires_key": False,
    },
    "openai": {
        "label": "OpenAI API",
        "default_base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "requires_key": True,
    },
    "openai_compatible": {
        "label": "OpenAI-compatible API (OpenRouter, xAI, …)",
        "default_base_url": "",
        "key_env": "OPENAI_API_KEY",
        "requires_key": False,
    },
    "anthropic": {
        "label": "Anthropic API",
        "default_base_url": "https://api.anthropic.com",
        "key_env": "ANTHROPIC_API_KEY",
        "requires_key": True,
    },
    "gemini": {
        "label": "Google Gemini API",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "key_env": "GEMINI_API_KEY",
        "requires_key": True,
    },
}

DEFAULT_PROVIDER = "lmstudio"


def normalize_provider(provider: str) -> str:
    value = str(provider or "").strip().lower()
    return value if value in PROVIDERS else DEFAULT_PROVIDER


def resolve_api_key(provider: str, raw_api_key: str = "") -> str:
    """Resolve an API key value.

    ``raw_api_key`` may be a literal key, ``env:VAR_NAME`` to read from the
    environment, or blank to fall back to the provider's default env var.
    """
    provider = normalize_provider(provider)
    raw = str(raw_api_key or "").strip()
    if raw.lower().startswith("env:"):
        return os.environ.get(raw[4:].strip(), "").strip()
    if raw:
        return raw
    key_env = PROVIDERS[provider].get("key_env", "")
    if key_env:
        return os.environ.get(key_env, "").strip()
    return ""


def _split_system_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    remainder: list[dict] = []
    for message in messages or []:
        if message.get("role") == "system":
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            if content:
                system_parts.append(str(content))
        else:
            remainder.append(message)
    return "\n\n".join(system_parts), remainder


def _parse_data_url(url: str) -> tuple[str, str]:
    """Return (media_type, base64_data) from a data URL."""
    if not url.startswith("data:"):
        raise ValueError("Only base64 data URLs are supported for API providers")
    header, _, data = url.partition(",")
    media_type = header[5:].split(";", 1)[0] or "image/jpeg"
    if ";base64" not in header:
        data = base64.b64encode(data.encode("utf-8")).decode("utf-8")
    return media_type, data


class AnthropicClient(LMStudioClient):
    ANTHROPIC_VERSION = "2023-06-01"

    def _auth_headers(self) -> dict:
        headers = {"anthropic-version": self.ANTHROPIC_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    @staticmethod
    def _convert_content(content) -> list[dict] | str:
        if isinstance(content, str):
            return content
        converted: list[dict] = []
        for item in content or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                media_type, data = _parse_data_url(url)
                converted.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
        return converted

    def chat(self, model: str, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> dict:
        system_text, remainder = _split_system_messages(messages)
        payload: dict = {
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "messages": [
                {
                    "role": message.get("role", "user"),
                    "content": self._convert_content(message.get("content", "")),
                }
                for message in remainder
            ],
        }
        if system_text:
            payload["system"] = system_text

        data = self._http_post_json("/v1/messages", payload)
        text_parts = [
            block.get("text", "")
            for block in data.get("content", []) or []
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return {"choices": [{"message": {"content": "\n".join(p for p in text_parts if p)}}]}

    def list_models(self) -> list[str]:
        data = self._http_get_json("/v1/models")
        return [item.get("id", "") for item in data.get("data", []) if item.get("id")]


class GeminiClient(LMStudioClient):
    def _auth_headers(self) -> dict:
        if self.api_key:
            return {"x-goog-api-key": self.api_key}
        return {}

    @staticmethod
    def _convert_content(content) -> list[dict]:
        if isinstance(content, str):
            return [{"text": content}] if content else []
        parts: list[dict] = []
        for item in content or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                media_type, data = _parse_data_url(url)
                parts.append({"inline_data": {"mime_type": media_type, "data": data}})
        return parts

    def chat(self, model: str, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> dict:
        system_text, remainder = _split_system_messages(messages)
        contents = []
        for message in remainder:
            role = "model" if message.get("role") == "assistant" else "user"
            parts = self._convert_content(message.get("content", ""))
            if parts:
                contents.append({"role": role, "parts": parts})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        model_id = str(model or "").strip()
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        data = self._http_post_json(f"/models/{model_id}:generateContent", payload)

        text_parts: list[str] = []
        for candidate in data.get("candidates", []) or []:
            for part in (candidate.get("content", {}) or {}).get("parts", []) or []:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(part["text"])
            break
        return {"choices": [{"message": {"content": "\n".join(text_parts)}}]}

    def list_models(self) -> list[str]:
        data = self._http_get_json("/models")
        names = []
        for item in data.get("models", []) or []:
            name = str(item.get("name", ""))
            if name.startswith("models/"):
                name = name[len("models/"):]
            if name:
                names.append(name)
        return names


_CLIENT_CLASSES = {
    "lmstudio": LMStudioClient,
    "openai": LMStudioClient,
    "openai_compatible": LMStudioClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


def create_vision_client(
    provider: str = DEFAULT_PROVIDER,
    base_url: str = "",
    timeout: float = 60.0,
    api_key: str = "",
) -> LMStudioClient:
    """Create a vision client for the given provider.

    ``base_url`` falls back to the provider default when blank.
    ``api_key`` accepts literal keys, ``env:VAR_NAME``, or blank (default env var).
    """
    provider = normalize_provider(provider)
    spec = PROVIDERS[provider]
    url = str(base_url or "").strip() or spec["default_base_url"]
    if not url:
        raise RuntimeError(f"Base URL is required for provider '{provider}'.")
    key = resolve_api_key(provider, api_key)
    if spec["requires_key"] and not key:
        env_hint = spec.get("key_env") or "an API key"
        raise RuntimeError(
            f"Provider '{provider}' requires an API key. "
            f"Enter one in the model settings or set {env_hint}."
        )
    client_cls = _CLIENT_CLASSES[provider]
    return client_cls(base_url=url, timeout=timeout, api_key=key)
