"""Unit and integration tests for thepipe.provider (MiniMax support)."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from thepipe.provider import (
    MINIMAX_PRESET,
    OPENAI_PRESET,
    PROVIDER_PRESETS,
    ProviderPreset,
    clamp_temperature,
    create_provider_client,
    detect_provider,
    get_provider_preset,
    strip_think_tags,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestProviderPreset(unittest.TestCase):
    """Tests for the ProviderPreset dataclass."""

    def test_openai_preset_exists(self):
        self.assertIn("openai", PROVIDER_PRESETS)
        self.assertEqual(OPENAI_PRESET.name, "openai")
        self.assertEqual(OPENAI_PRESET.base_url, "https://api.openai.com/v1")

    def test_minimax_preset_exists(self):
        self.assertIn("minimax", PROVIDER_PRESETS)
        self.assertEqual(MINIMAX_PRESET.name, "minimax")
        self.assertEqual(MINIMAX_PRESET.base_url, "https://api.minimax.io/v1")
        self.assertEqual(MINIMAX_PRESET.default_model, "MiniMax-M2.7")
        self.assertEqual(MINIMAX_PRESET.api_key_env, "MINIMAX_API_KEY")

    def test_minimax_models(self):
        models = MINIMAX_PRESET.models
        self.assertIn("MiniMax-M2.7", models)
        self.assertIn("MiniMax-M2.7-highspeed", models)
        self.assertIn("MiniMax-M2.5", models)
        self.assertIn("MiniMax-M2.5-highspeed", models)

    def test_minimax_temperature_range(self):
        self.assertEqual(MINIMAX_PRESET.temperature_min, 0.0)
        self.assertEqual(MINIMAX_PRESET.temperature_max, 1.0)


class TestGetProviderPreset(unittest.TestCase):
    """Tests for get_provider_preset()."""

    def test_known_provider_openai(self):
        preset = get_provider_preset("openai")
        self.assertEqual(preset.name, "openai")

    def test_known_provider_minimax(self):
        preset = get_provider_preset("minimax")
        self.assertEqual(preset.name, "minimax")

    def test_case_insensitive(self):
        preset = get_provider_preset("MiniMax")
        self.assertEqual(preset.name, "minimax")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_provider_preset("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("Available providers", str(ctx.exception))


class TestDetectProvider(unittest.TestCase):
    """Tests for detect_provider()."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False)
    def test_detects_minimax_when_only_minimax_key(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove OPENAI_API_KEY if present
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                os.environ["MINIMAX_API_KEY"] = "test-key"
                self.assertEqual(detect_provider(), "minimax")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_defaults_to_openai(self):
        self.assertEqual(detect_provider(), "openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_openai_when_no_keys(self):
        self.assertEqual(detect_provider(), "openai")


class TestClampTemperature(unittest.TestCase):
    """Tests for clamp_temperature()."""

    def test_none_returns_none(self):
        self.assertIsNone(clamp_temperature(None, MINIMAX_PRESET))

    def test_within_range(self):
        self.assertEqual(clamp_temperature(0.5, MINIMAX_PRESET), 0.5)

    def test_above_max_clamped(self):
        self.assertEqual(clamp_temperature(1.5, MINIMAX_PRESET), 1.0)

    def test_below_min_clamped(self):
        self.assertEqual(clamp_temperature(-0.1, MINIMAX_PRESET), 0.0)

    def test_zero_accepted(self):
        self.assertEqual(clamp_temperature(0.0, MINIMAX_PRESET), 0.0)

    def test_openai_wider_range(self):
        self.assertEqual(clamp_temperature(1.5, OPENAI_PRESET), 1.5)
        self.assertEqual(clamp_temperature(2.5, OPENAI_PRESET), 2.0)


class TestStripThinkTags(unittest.TestCase):
    """Tests for strip_think_tags()."""

    def test_no_think_tags(self):
        text = "Hello world"
        self.assertEqual(strip_think_tags(text), "Hello world")

    def test_single_think_tag(self):
        text = "<think>reasoning here</think>The answer is 42."
        self.assertEqual(strip_think_tags(text), "The answer is 42.")

    def test_multiline_think_tag(self):
        text = "<think>\nStep 1: analyze\nStep 2: compute\n</think>\nResult: done"
        self.assertEqual(strip_think_tags(text), "Result: done")

    def test_multiple_think_tags(self):
        text = "<think>a</think>Hello <think>b</think>world"
        self.assertEqual(strip_think_tags(text), "Hello world")

    def test_empty_think_tag(self):
        text = "<think></think>Just the output"
        self.assertEqual(strip_think_tags(text), "Just the output")


class TestCreateProviderClient(unittest.TestCase):
    """Tests for create_provider_client()."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"}, clear=False)
    def test_creates_minimax_client(self):
        client, preset = create_provider_client("minimax")
        self.assertEqual(preset.name, "minimax")
        self.assertEqual(preset.default_model, "MiniMax-M2.7")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=False)
    def test_creates_openai_client(self):
        client, preset = create_provider_client("openai")
        self.assertEqual(preset.name, "openai")

    def test_explicit_api_key(self):
        client, preset = create_provider_client("minimax", api_key="explicit-key")
        self.assertEqual(preset.name, "minimax")

    @patch.dict(os.environ, {}, clear=True)
    def test_no_api_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_provider_client("minimax")
        self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False)
    def test_custom_base_url(self):
        client, preset = create_provider_client(
            "minimax", base_url="https://custom.api.example.com/v1"
        )
        self.assertEqual(preset.name, "minimax")


class TestChunkerAgenticFallback(unittest.TestCase):
    """Tests for chunk_agentic() MiniMax fallback."""

    def test_agentic_json_fallback(self):
        """Verify chunk_agentic() falls back to json_object mode when
        .beta.chat.completions.parse() is unavailable."""
        from thepipe.chunker import chunk_agentic
        from thepipe.core import Chunk

        chunks = [Chunk(path="test.txt", text="Line one\nLine two\nLine three")]

        # Mock an OpenAI client whose .beta.chat.completions.parse() fails
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = Exception(
            "Not supported"
        )

        # Mock fallback .chat.completions.create() to return valid JSON
        sections_json = json.dumps(
            {
                "sections": [
                    {"title": "All", "start_line": 1, "end_line": 3},
                ]
            }
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = sections_json
        mock_client.chat.completions.create.return_value = mock_response

        result = chunk_agentic(chunks, openai_client=mock_client, model="MiniMax-M2.7")

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("Line one", result[0].text)

    def test_agentic_fallback_with_think_tags(self):
        """Verify think tags are stripped in agentic chunking fallback."""
        from thepipe.chunker import chunk_agentic
        from thepipe.core import Chunk

        chunks = [Chunk(path="test.txt", text="Hello world")]

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = Exception(
            "Not supported"
        )

        sections_json = (
            '<think>reasoning about sections</think>'
            + json.dumps(
                {"sections": [{"title": "Intro", "start_line": 1, "end_line": 1}]}
            )
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = sections_json
        mock_client.chat.completions.create.return_value = mock_response

        result = chunk_agentic(chunks, openai_client=mock_client, model="MiniMax-M2.5")

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


class TestProviderPresetImmutability(unittest.TestCase):
    """Verify presets are frozen dataclasses."""

    def test_preset_is_frozen(self):
        with self.assertRaises(AttributeError):
            MINIMAX_PRESET.name = "changed"  # type: ignore[misc]

    def test_preset_has_correct_fields(self):
        self.assertIsInstance(MINIMAX_PRESET, ProviderPreset)
        self.assertIsInstance(MINIMAX_PRESET.models, dict)


# ---------------------------------------------------------------------------
# Integration tests (require MINIMAX_API_KEY)
# ---------------------------------------------------------------------------

HAS_MINIMAX_KEY = bool(os.getenv("MINIMAX_API_KEY"))


@unittest.skipUnless(HAS_MINIMAX_KEY, "MINIMAX_API_KEY not set")
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that hit the real MiniMax API."""

    def setUp(self):
        self.client, self.preset = create_provider_client("minimax")
        self.files_directory = os.path.join(os.path.dirname(__file__), "files")

    def test_minimax_chat_completion(self):
        """Verify basic chat completion works with MiniMax."""
        response = self.client.chat.completions.create(
            model=self.preset.default_model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        content = response.choices[0].message.content
        self.assertIsNotNone(content)
        self.assertGreater(len(content), 0)

    def test_minimax_json_mode(self):
        """Verify json_object response_format works with MiniMax."""
        response = self.client.chat.completions.create(
            model=self.preset.default_model,
            messages=[
                {
                    "role": "user",
                    "content": 'Return a JSON object with key "status" and value "ok".',
                }
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        self.assertIsNotNone(content)
        data = json.loads(strip_think_tags(content))
        self.assertIn("status", data)

    def test_minimax_scrape_pdf_text_only(self):
        """Verify MiniMax can be used for PDF scraping."""
        from thepipe.scraper import scrape_file

        pdf_path = os.path.join(self.files_directory, "example.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest("example.pdf not found in test files")

        chunks = scrape_file(
            filepath=pdf_path,
            openai_client=self.client,
            model=self.preset.default_model,
            include_input_images=False,
        )
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(any(c.text for c in chunks))


if __name__ == "__main__":
    unittest.main()
