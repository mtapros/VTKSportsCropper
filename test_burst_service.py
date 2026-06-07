from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from burst_service import (
    DEFAULT_BURST_THUMBNAIL_SIZE,
    BurstSettingsStore,
    BurstToolSettings,
    DEFAULT_BURST_WINNER_CRITERIA,
    default_winner_criteria_text,
    normalize_winner_criteria_lines,
)


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


if __name__ == "__main__":
    unittest.main()
