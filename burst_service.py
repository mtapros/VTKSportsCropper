from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_BURST_WINNER_CRITERIA = (
    "sharpest",
    "cleanest",
    "strongest timing",
    "clear subject visibility",
    "minimal motion blur",
    "most reliably usable if unsure",
)


@dataclass
class BurstAnalysis:
    ordered_paths: list[Path]
    all_groups: list[list[Path]]
    burst_groups: list[list[Path]]

    @property
    def total_images(self) -> int:
        return len(self.ordered_paths)

    @property
    def burst_images(self) -> int:
        return sum(len(group) for group in self.burst_groups)

    @property
    def non_burst_images(self) -> int:
        return max(0, self.total_images - self.burst_images)


@dataclass
class BurstToolSettings:
    fps_threshold: float = 8.0
    keep_per_burst: int = 1
    winner_criteria: str = ""


class BurstSettingsStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load_profile(self, profile_name: str) -> BurstToolSettings:
        profile_key = str(profile_name or "").strip() or "Generic Sport"
        raw = self._load_all()
        data = raw.get(profile_key, {}) if isinstance(raw, dict) else {}
        return BurstToolSettings(
            fps_threshold=max(0.1, float(data.get("fps_threshold", 8.0))),
            keep_per_burst=max(1, int(data.get("keep_per_burst", 1))),
            winner_criteria=str(data.get("winner_criteria", default_winner_criteria_text())).strip()
            or default_winner_criteria_text(),
        )

    def save_profile(self, profile_name: str, settings: BurstToolSettings) -> None:
        profile_key = str(profile_name or "").strip() or "Generic Sport"
        raw = self._load_all()
        raw[profile_key] = {
            "fps_threshold": max(0.1, float(settings.fps_threshold)),
            "keep_per_burst": max(1, int(settings.keep_per_burst)),
            "winner_criteria": str(settings.winner_criteria or "").strip() or default_winner_criteria_text(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def _load_all(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}


def default_winner_criteria_text() -> str:
    return "\n".join(DEFAULT_BURST_WINNER_CRITERIA)


def normalize_winner_criteria_lines(text: str | None, include_defaults: bool = True) -> list[str]:
    raw_lines: list[str] = []
    if include_defaults:
        raw_lines.extend(DEFAULT_BURST_WINNER_CRITERIA)
    if text:
        raw_lines.extend(str(text).replace("\r", "\n").split("\n"))

    normalized: list[str] = []
    seen: set[str] = set()
    for line in raw_lines:
        clean = str(line or "").strip().lstrip("-•*").strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(clean)
    return normalized


def list_supported_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return []
    paths = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    return sorted(paths, key=lambda p: p.name.lower())


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(Path(image_path)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img.copy()


def build_square_thumbnail(image_path: Path, side: int, background: tuple[int, int, int] = (20, 20, 20)) -> Image.Image:
    side = max(1, int(side))
    img = load_rgb_image(Path(image_path))
    img.thumbnail((side, side), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), background)
    offset = ((side - img.width) // 2, (side - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def extract_capture_timestamp(image_path: Path) -> tuple[float, str]:
    path = Path(image_path)
    try:
        with Image.open(path) as img:
            exif = img.getexif()
        if exif:
            dt_value = exif.get(36867) or exif.get(36868) or exif.get(306)
            if dt_value:
                dt = datetime.strptime(str(dt_value).strip(), "%Y:%m:%d %H:%M:%S")
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
        return float(path.stat().st_mtime), "mtime"
    except Exception:
        return 0.0, "none"


def group_adjacent_images(ordered_paths: list[Path], fps_threshold: float) -> list[list[Path]]:
    if not ordered_paths:
        return []

    threshold_sec = 1.0 / max(0.1, float(fps_threshold))
    groups: list[list[Path]] = []
    current: list[Path] = []
    previous_ts: float | None = None

    for path in ordered_paths:
        ts, _ = extract_capture_timestamp(Path(path))
        if not current:
            current = [Path(path)]
            previous_ts = ts
            continue

        delta = ts - float(previous_ts or 0.0)
        if 0.0 <= delta <= threshold_sec:
            current.append(Path(path))
        else:
            groups.append(current)
            current = [Path(path)]
        previous_ts = ts

    if current:
        groups.append(current)
    return groups


def analyze_bursts(folder: Path, fps_threshold: float) -> BurstAnalysis:
    ordered_paths = list_supported_images(Path(folder))
    all_groups = group_adjacent_images(ordered_paths, fps_threshold)
    burst_groups = [group for group in all_groups if len(group) > 1]
    return BurstAnalysis(
        ordered_paths=ordered_paths,
        all_groups=all_groups,
        burst_groups=burst_groups,
    )
