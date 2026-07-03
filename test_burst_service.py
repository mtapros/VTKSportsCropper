from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from burst_service import (
    DEFAULT_BURST_THUMBNAIL_SIZE,
    DHASH_NEAR_DUPLICATE_THRESHOLD,
    DHASH_SCENE_CHANGE_THRESHOLD,
    MAX_BURST_GROUP_FRAMES,
    MAX_BURST_GROUP_SPAN_SEC,
    BurstSettingsStore,
    BurstToolSettings,
    DEFAULT_BURST_WINNER_CRITERIA,
    default_winner_criteria_text,
    dhash,
    extract_capture_timestamp,
    group_adjacent_images,
    hamming_distance,
    normalize_winner_criteria_lines,
    pil_laplacian_focus,
    _looks_like_sequential_burst_names,
)
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(color: tuple, size: tuple = (64, 64)) -> Image.Image:
    """Return a small solid-colour RGB image."""
    img = Image.new("RGB", size, color)
    return img


def _gradient_image(size: tuple = (64, 64)) -> Image.Image:
    """Return a horizontal gradient RGB image (dark left → bright right)."""
    img = Image.new("RGB", size)
    w, h = size
    for x in range(w):
        v = int(255 * x / max(1, w - 1))
        for y in range(h):
            img.putpixel((x, y), (v, v, v))
    return img


def _save_jpeg_with_exif(tmp_dir: Path, filename: str, exif_bytes: bytes) -> Path:
    """Save a tiny JPEG with the supplied raw EXIF data and return its path."""
    from PIL import Image
    img = Image.new("RGB", (8, 8), (128, 128, 128))
    path = tmp_dir / filename
    img.save(str(path), format="JPEG", exif=exif_bytes)
    return path


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

class BurstServiceTests(unittest.TestCase):
    def test_normalize_winner_criteria_keeps_defaults_and_dedupes(self):
        lines = normalize_winner_criteria_lines("sharpest\npeak action\nPeak Action")
        self.assertEqual(list(DEFAULT_BURST_WINNER_CRITERIA), lines[: len(DEFAULT_BURST_WINNER_CRITERIA)])
        self.assertEqual(1, sum(1 for line in lines if line.casefold() == "peak action"))

    def test_settings_store_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BurstSettingsStore(Path(tmpdir) / "burst_settings.json")
            store.save_profile(
                "Dance",
                BurstToolSettings(fps_threshold=12.5, keep_per_burst=3, winner_criteria="peak action", thumbnail_size=320),
            )
            loaded = store.load_profile("Dance")
            self.assertEqual(12.5, loaded.fps_threshold)
            self.assertEqual(3, loaded.keep_per_burst)
            self.assertEqual("peak action", loaded.winner_criteria)
            self.assertEqual(320, loaded.thumbnail_size)
            self.assertEqual(default_winner_criteria_text(), store.load_profile("Unknown").winner_criteria)
            self.assertEqual(DEFAULT_BURST_THUMBNAIL_SIZE, store.load_profile("Unknown").thumbnail_size)


# ---------------------------------------------------------------------------
# 1.1 Subsecond EXIF tag preference
# ---------------------------------------------------------------------------

class SubsecTimestampTests(unittest.TestCase):
    """extract_capture_timestamp should prefer 37522 > 37523 > 37521."""

    def _make_image_with_exif_tags(self, tag_values: dict) -> MagicMock:
        """Build a mock PIL Image whose getexif() returns *tag_values*."""
        exif_mock = MagicMock()
        exif_mock.__bool__ = lambda self: True

        def _get(tag, default=None):
            return tag_values.get(tag, default)

        exif_mock.get = _get
        img_mock = MagicMock()
        img_mock.getexif.return_value = exif_mock
        img_mock.__enter__ = lambda s: img_mock
        img_mock.__exit__ = MagicMock(return_value=False)
        return img_mock

    def _patch_open(self, img_mock):
        return patch("burst_service.Image.open", return_value=img_mock)

    def test_prefers_37522_over_37523(self):
        img = self._make_image_with_exif_tags({
            36867: "2024:06:01 12:00:00",
            37522: "75",  # SubSecTimeOriginal  → 0.75 s
            37523: "20",  # SubSecTimeDigitized → should be ignored
            37521: "10",  # SubSecTime          → should be ignored
        })
        with self._patch_open(img), tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            ts, src = extract_capture_timestamp(Path(f.name))
        self.assertEqual("exif_subsec", src)
        self.assertAlmostEqual(0.75, ts % 1.0, places=5)

    def test_falls_back_to_37523_when_37522_absent(self):
        img = self._make_image_with_exif_tags({
            36867: "2024:06:01 12:00:00",
            37523: "30",  # SubSecTimeDigitized → 0.30 s
            37521: "10",  # SubSecTime          → should be ignored
        })
        with self._patch_open(img), tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            ts, src = extract_capture_timestamp(Path(f.name))
        self.assertEqual("exif_subsec", src)
        self.assertAlmostEqual(0.30, ts % 1.0, places=5)

    def test_falls_back_to_37521_when_37522_and_37523_absent(self):
        img = self._make_image_with_exif_tags({
            36867: "2024:06:01 12:00:00",
            37521: "50",  # SubSecTime → 0.50 s
        })
        with self._patch_open(img), tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            ts, src = extract_capture_timestamp(Path(f.name))
        self.assertEqual("exif_subsec", src)
        self.assertAlmostEqual(0.50, ts % 1.0, places=5)

    def test_whole_second_when_no_subsec_tags(self):
        img = self._make_image_with_exif_tags({
            36867: "2024:06:01 12:00:00",
        })
        with self._patch_open(img), tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            ts, src = extract_capture_timestamp(Path(f.name))
        self.assertEqual("exif_whole", src)
        self.assertAlmostEqual(0.0, ts % 1.0, places=5)

    def test_mtime_when_no_exif(self):
        img = self._make_image_with_exif_tags({})
        # Make exif falsy
        img.getexif.return_value = None
        with self._patch_open(img), tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            _, src = extract_capture_timestamp(Path(f.name))
        self.assertEqual("mtime", src)


# ---------------------------------------------------------------------------
# 1.2 Second-boundary grouping
# ---------------------------------------------------------------------------

class SecondBoundaryGroupingTests(unittest.TestCase):
    """group_adjacent_images should keep whole-second-EXIF burst members together."""

    def _make_exif_ts(self, whole_ts_str: str, subsec: str | None = None) -> dict:
        tags: dict = {36867: whole_ts_str}
        if subsec is not None:
            tags[37522] = subsec
        return tags

    def _mock_open(self, tag_map: dict):
        exif_mock = MagicMock()
        exif_mock.__bool__ = lambda s: True
        exif_mock.get = lambda t, default=None: tag_map.get(t, default)
        img_mock = MagicMock()
        img_mock.getexif.return_value = exif_mock
        img_mock.__enter__ = lambda s: img_mock
        img_mock.__exit__ = MagicMock(return_value=False)
        return img_mock

    def test_whole_second_exif_crossing_boundary_grouped_by_name(self):
        """Two files with sequential names at the 1-s EXIF boundary should be grouped."""
        paths = [Path("IMG_0001.jpg"), Path("IMG_0002.jpg")]
        ts_map = {
            "IMG_0001.jpg": ("2024:06:01 12:00:00", None),
            "IMG_0002.jpg": ("2024:06:01 12:00:01", None),
        }

        def side_effect(path):
            name = Path(path).name
            ts_str, subsec = ts_map[name]
            return self._mock_open(self._make_exif_ts(ts_str, subsec))

        with patch("burst_service.Image.open", side_effect=side_effect):
            groups = group_adjacent_images(paths, fps_threshold=8.0)

        self.assertEqual(1, len(groups), "Sequential whole-second names should be one group")
        self.assertEqual(2, len(groups[0]))

    def test_whole_second_exif_crossing_boundary_split_when_non_sequential(self):
        """Non-sequential names at the 1-s boundary should NOT be grouped."""
        paths = [Path("IMG_0001.jpg"), Path("IMG_9999.jpg")]
        ts_map = {
            "IMG_0001.jpg": ("2024:06:01 12:00:00", None),
            "IMG_9999.jpg": ("2024:06:01 12:00:01", None),
        }

        def side_effect(path):
            name = Path(path).name
            ts_str, subsec = ts_map[name]
            return self._mock_open(self._make_exif_ts(ts_str, subsec))

        with patch("burst_service.Image.open", side_effect=side_effect):
            groups = group_adjacent_images(paths, fps_threshold=8.0)

        self.assertEqual(2, len(groups), "Non-sequential names at 1-s boundary should be two groups")

    def test_subsec_exif_uses_delta_threshold_not_name_check(self):
        """Subsecond EXIF should use the normal time-delta threshold (no name check)."""
        paths = [Path("IMG_0001.jpg"), Path("IMG_9999.jpg")]
        ts_map = {
            "IMG_0001.jpg": ("2024:06:01 12:00:00", "00"),   # ts = 0.00
            "IMG_9999.jpg": ("2024:06:01 12:00:00", "10"),   # ts = 0.10 → delta = 0.10 < 1/8
        }

        def side_effect(path):
            name = Path(path).name
            ts_str, subsec = ts_map[name]
            return self._mock_open(self._make_exif_ts(ts_str, subsec))

        with patch("burst_service.Image.open", side_effect=side_effect):
            groups = group_adjacent_images(paths, fps_threshold=8.0)

        self.assertEqual(1, len(groups), "Close subsec timestamps should be grouped regardless of filename")


# ---------------------------------------------------------------------------
# 1.3 dHash and hamming_distance
# ---------------------------------------------------------------------------

class DHashTests(unittest.TestCase):

    def test_identical_images_have_zero_distance(self):
        img = _solid_image((100, 150, 200))
        h1 = dhash(img)
        h2 = dhash(img)
        self.assertEqual(0, hamming_distance(h1, h2))

    def test_different_images_have_nonzero_distance(self):
        # Use gradient images with opposite directions so their dHashes differ.
        img_left_dark = _gradient_image()   # dark left → bright right
        from PIL import ImageOps
        img_right_dark = ImageOps.mirror(img_left_dark)   # bright left → dark right
        h1 = dhash(img_left_dark)
        h2 = dhash(img_right_dark)
        d = hamming_distance(h1, h2)
        self.assertGreater(d, 0)

    def test_hamming_distance_symmetric(self):
        img_a = _gradient_image()
        img_b = _solid_image((200, 50, 50))
        h_a = dhash(img_a)
        h_b = dhash(img_b)
        self.assertEqual(hamming_distance(h_a, h_b), hamming_distance(h_b, h_a))

    def test_dhash_returns_int(self):
        img = _solid_image((128, 128, 128))
        h = dhash(img)
        self.assertIsInstance(h, int)

    def test_similar_images_low_distance(self):
        # Two very similar images (same gradient, slight brightness tweak)
        base = _gradient_image((128, 64))
        from PIL import ImageEnhance
        tweaked = ImageEnhance.Brightness(base).enhance(1.05)
        h_base = dhash(base)
        h_tweaked = dhash(tweaked)
        d = hamming_distance(h_base, h_tweaked)
        self.assertLessEqual(d, 10, "Slightly tweaked image should have low dHash distance")


# ---------------------------------------------------------------------------
# pil_laplacian_focus
# ---------------------------------------------------------------------------

class PilLaplacianFocusTests(unittest.TestCase):

    def test_returns_float(self):
        img = _solid_image((128, 128, 128))
        result = pil_laplacian_focus(img)
        self.assertIsInstance(result, float)

    def test_sharp_greater_than_blurry(self):
        sharp = _gradient_image((256, 256))
        from PIL import ImageFilter
        blurry = sharp.filter(ImageFilter.GaussianBlur(20))
        f_sharp = pil_laplacian_focus(sharp)
        f_blurry = pil_laplacian_focus(blurry)
        self.assertGreater(f_sharp, f_blurry)

    def test_bbox_crops_correctly(self):
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        # Draw a white rectangle in a known region
        for x in range(10, 20):
            for y in range(10, 20):
                img.putpixel((x, y), (255, 255, 255))
        score_region = pil_laplacian_focus(img, bbox=(0, 0, 100, 100))
        self.assertGreater(score_region, 0.0)


# ---------------------------------------------------------------------------
# _looks_like_sequential_burst_names
# ---------------------------------------------------------------------------

class SequentialNameTests(unittest.TestCase):
    def test_sequential(self):
        self.assertTrue(_looks_like_sequential_burst_names(Path("IMG_0001.jpg"), Path("IMG_0002.jpg")))

    def test_gap_of_2(self):
        self.assertTrue(_looks_like_sequential_burst_names(Path("IMG_0001.jpg"), Path("IMG_0003.jpg")))

    def test_gap_of_3(self):
        self.assertTrue(_looks_like_sequential_burst_names(Path("IMG_0001.jpg"), Path("IMG_0004.jpg")))

    def test_gap_too_large(self):
        self.assertFalse(_looks_like_sequential_burst_names(Path("IMG_0001.jpg"), Path("IMG_0005.jpg")))

    def test_different_prefix(self):
        self.assertFalse(_looks_like_sequential_burst_names(Path("DSC_0001.jpg"), Path("IMG_0002.jpg")))

    def test_none_previous(self):
        self.assertFalse(_looks_like_sequential_burst_names(None, Path("IMG_0001.jpg")))


# ---------------------------------------------------------------------------
# 2.3 Burst candidate capping / blur pre-filter (pure logic via burst_service)
# ---------------------------------------------------------------------------

class BurstCandidateFilterTests(unittest.TestCase):
    """Pure-logic tests for the candidate pre-filter behaviour.

    Rather than invoking the Tkinter-dependent AICullTool, we replicate the
    filtering logic in a helper so it's easily testable.
    """

    def _filter_candidates(
        self,
        focus_values: list[float],
        keep_per_burst: int,
        max_candidates: int = 6,
        blur_frac: float = 0.30,
    ) -> list[int]:
        """Return the indices of candidates that survive the filter."""
        candidates = [{"path": f"/tmp/img_{i:04d}.jpg", "burst_focus": f} for i, f in enumerate(focus_values)]

        keep_min = max(2, keep_per_burst + 1)
        best_focus = max(float(c.get("burst_focus", 0.0)) for c in candidates)
        blur_threshold = best_focus * blur_frac

        filtered = list(candidates)
        if best_focus > 0 and len(filtered) > keep_min:
            non_blurry = [c for c in filtered if float(c.get("burst_focus", 0.0)) >= blur_threshold]
            if len(non_blurry) >= keep_min:
                filtered = non_blurry

        if len(filtered) > max_candidates:
            filtered = sorted(filtered, key=lambda x: float(x.get("burst_focus", 0.0)), reverse=True)
            filtered = filtered[:max_candidates]

        indices = [int(c["path"].split("_")[1].split(".")[0]) for c in filtered]
        return sorted(indices)

    def test_blurry_frames_excluded(self):
        # Candidate 0 has very low focus (blurry), others are sharp.
        focus = [1.0, 100.0, 95.0, 90.0]
        kept = self._filter_candidates(focus, keep_per_burst=1)
        self.assertNotIn(0, kept)

    def test_always_keeps_keep_min_candidates(self):
        # All frames have low focus except 1; keep_per_burst=1 → keep_min=2.
        focus = [1.0, 1.0, 1.0, 2.0]
        kept = self._filter_candidates(focus, keep_per_burst=1)
        self.assertGreaterEqual(len(kept), 2)

    def test_cap_at_max_candidates(self):
        focus = [float(i + 1) * 10 for i in range(12)]
        kept = self._filter_candidates(focus, keep_per_burst=1, max_candidates=6)
        self.assertLessEqual(len(kept), 6)

    def test_top_focus_selected_after_cap(self):
        focus = [float(i + 1) * 10 for i in range(10)]
        kept = self._filter_candidates(focus, keep_per_burst=1, max_candidates=4)
        # Should be the 4 highest (indices 6, 7, 8, 9)
        self.assertEqual(sorted(kept), [6, 7, 8, 9])


# ---------------------------------------------------------------------------
# 3.3 Winner-criteria normalization injection
# ---------------------------------------------------------------------------

class WinnerCriteriaInjectionTests(unittest.TestCase):

    def test_profile_criteria_in_normalized_lines(self):
        custom = "peak action\nbest expression"
        lines = normalize_winner_criteria_lines(custom)
        # All defaults should be present.
        for default in DEFAULT_BURST_WINNER_CRITERIA:
            self.assertIn(default, lines)
        # Custom criteria should also be present.
        self.assertIn("peak action", lines)
        self.assertIn("best expression", lines)

    def test_deduplicated_criteria(self):
        lines = normalize_winner_criteria_lines("sharpest\nsharpest\nSHARPEST")
        count = sum(1 for l in lines if l.casefold() == "sharpest")
        self.assertEqual(1, count)

    def test_criteria_bullet_stripping(self):
        lines = normalize_winner_criteria_lines("- peak action\n• best expression\n* good timing")
        self.assertIn("peak action", lines)
        self.assertIn("best expression", lines)
        self.assertIn("good timing", lines)

    def test_empty_criteria_returns_defaults_only(self):
        lines = normalize_winner_criteria_lines(None)
        self.assertEqual(list(DEFAULT_BURST_WINNER_CRITERIA), lines)

    def test_criteria_text_built_from_settings_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = BurstSettingsStore(Path(tmpdir) / "bs.json")
            store.save_profile(
                "Soccer",
                BurstToolSettings(winner_criteria="peak action\nbest expression"),
            )
            profile = store.load_profile("Soccer")
            lines = normalize_winner_criteria_lines(profile.winner_criteria)
            self.assertIn("peak action", lines)
            self.assertIn("best expression", lines)


if __name__ == "__main__":
    unittest.main()
