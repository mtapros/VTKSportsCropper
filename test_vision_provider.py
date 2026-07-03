import base64
import os
import unittest
from unittest import mock

from lmstudio_client import LMStudioClient
from vision_provider import (
    AnthropicClient,
    GeminiClient,
    create_vision_client,
    normalize_provider,
    resolve_api_key,
)


DATA_URL = "data:image/jpeg;base64," + base64.b64encode(b"fake-image").decode("utf-8")


def vision_messages() -> list[dict]:
    return [
        {"role": "system", "content": "system text"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "user text"},
                {"type": "image_url", "image_url": {"url": DATA_URL}},
            ],
        },
    ]


class TestProviderRegistry(unittest.TestCase):
    def test_normalize_provider_defaults_unknown_to_lmstudio(self):
        self.assertEqual(normalize_provider("anthropic"), "anthropic")
        self.assertEqual(normalize_provider("bogus"), "lmstudio")
        self.assertEqual(normalize_provider(""), "lmstudio")

    def test_resolve_api_key_literal(self):
        self.assertEqual(resolve_api_key("openai", "sk-test"), "sk-test")

    def test_resolve_api_key_env_syntax(self):
        with mock.patch.dict(os.environ, {"MY_CUSTOM_KEY": "from-env"}):
            self.assertEqual(resolve_api_key("openai", "env:MY_CUSTOM_KEY"), "from-env")

    def test_resolve_api_key_default_env_var(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-env"}):
            self.assertEqual(resolve_api_key("anthropic", ""), "anthropic-env")


class TestFactory(unittest.TestCase):
    def test_lmstudio_default(self):
        client = create_vision_client()
        self.assertIsInstance(client, LMStudioClient)
        self.assertNotIsInstance(client, (AnthropicClient, GeminiClient))
        self.assertEqual(client.base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(client.api_key, "")

    def test_openai_requires_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                create_vision_client(provider="openai")

    def test_openai_with_key(self):
        client = create_vision_client(provider="openai", api_key="sk-abc")
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.api_key, "sk-abc")
        auth = client._auth_headers().get("Authorization", "")
        self.assertTrue(auth.startswith("Bearer"))
        self.assertTrue(auth.endswith("sk-abc"))

    def test_openai_compatible_custom_url(self):
        client = create_vision_client(
            provider="openai_compatible",
            base_url="https://openrouter.ai/api/v1",
            api_key="or-key",
        )
        self.assertEqual(client.base_url, "https://openrouter.ai/api/v1")

    def test_openai_compatible_requires_url(self):
        with self.assertRaises(RuntimeError):
            create_vision_client(provider="openai_compatible", base_url="")

    def test_anthropic_and_gemini_classes(self):
        self.assertIsInstance(create_vision_client(provider="anthropic", api_key="k"), AnthropicClient)
        self.assertIsInstance(create_vision_client(provider="gemini", api_key="k"), GeminiClient)


class TestAnthropicTranslation(unittest.TestCase):
    def test_chat_payload_and_response(self):
        client = AnthropicClient(base_url="https://api.anthropic.com", api_key="k")
        captured = {}

        def fake_post(path, payload, headers=None):
            captured["path"] = path
            captured["payload"] = payload
            return {"content": [{"type": "text", "text": "hello"}]}

        with mock.patch.object(client, "_http_post_json", side_effect=fake_post):
            data = client.chat("claude-x", vision_messages(), temperature=0.3, max_tokens=99)

        self.assertEqual(captured["path"], "/v1/messages")
        payload = captured["payload"]
        self.assertEqual(payload["system"], "system text")
        self.assertEqual(payload["model"], "claude-x")
        self.assertEqual(payload["max_tokens"], 99)
        self.assertEqual(len(payload["messages"]), 1)
        blocks = payload["messages"][0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "user text"})
        self.assertEqual(blocks[1]["type"], "image")
        self.assertEqual(blocks[1]["source"]["media_type"], "image/jpeg")
        self.assertEqual(
            blocks[1]["source"]["data"],
            base64.b64encode(b"fake-image").decode("utf-8"),
        )
        self.assertEqual(LMStudioClient._chat_response_text(data), "hello")

    def test_auth_headers(self):
        client = AnthropicClient(base_url="https://api.anthropic.com", api_key="k")
        headers = client._auth_headers()
        self.assertEqual(headers.get("x-api-key"), "k")
        self.assertIn("anthropic-version", headers)


class TestGeminiTranslation(unittest.TestCase):
    def test_chat_payload_and_response(self):
        client = GeminiClient(
            base_url="https://generativelanguage.googleapis.com/v1beta", api_key="k"
        )
        captured = {}

        def fake_post(path, payload, headers=None):
            captured["path"] = path
            captured["payload"] = payload
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "gemini says"}]}}
                ]
            }

        with mock.patch.object(client, "_http_post_json", side_effect=fake_post):
            data = client.chat("models/gemini-x", vision_messages(), temperature=0.3, max_tokens=99)

        self.assertEqual(captured["path"], "/models/gemini-x:generateContent")
        payload = captured["payload"]
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "system text"}]})
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 99)
        parts = payload["contents"][0]["parts"]
        self.assertEqual(parts[0], {"text": "user text"})
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/jpeg")
        self.assertEqual(LMStudioClient._chat_response_text(data), "gemini says")

    def test_auth_headers(self):
        client = GeminiClient(base_url="https://example.com", api_key="k")
        self.assertEqual(client._auth_headers(), {"x-goog-api-key": "k"})


class TestLMStudioClientVisionMessages(unittest.TestCase):
    def test_vision_chat_text_routes_through_chat(self):
        client = LMStudioClient(base_url="http://127.0.0.1:1234/v1")
        captured = {}

        def fake_chat(model, messages, temperature=0.2, max_tokens=512):
            captured["model"] = model
            captured["messages"] = messages
            return {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.object(client, "chat", side_effect=fake_chat):
            with mock.patch.object(client, "_image_file_to_data_url", return_value=DATA_URL):
                text = client.vision_chat_text("m", "img.jpg", "prompt")

        self.assertEqual(text, "ok")
        self.assertEqual(captured["model"], "m")
        user_content = captured["messages"][1]["content"]
        self.assertEqual(user_content[1]["image_url"]["url"], DATA_URL)


if __name__ == "__main__":
    unittest.main()
