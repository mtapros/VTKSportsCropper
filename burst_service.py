from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def list_supported_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return []
    paths = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    return sorted(paths, key=lambda p: p.name.lower())


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
