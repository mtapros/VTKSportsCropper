from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_BURST_WINNER_CRITERIA = (
    "sharpest",
    "cleanest",
    "strongest timing",
    "clear subject visibility",
    "minimal motion blur",
    "most reliably usable if unsure",
)
DEFAULT_BURST_THUMBNAIL_SIZE = 288
MIN_BURST_THUMBNAIL_SIZE = 120
MAX_BURST_THUMBNAIL_SIZE = 768

# Burst grouping / splitting constants
MAX_BURST_GROUP_FRAMES: int = 20
MAX_BURST_GROUP_SPAN_SEC: float = 5.0

# dHash thresholds (64-bit hash, max distance = 64)
DHASH_SCENE_CHANGE_THRESHOLD: int = 24   # above → likely scene change → force split
DHASH_NEAR_DUPLICATE_THRESHOLD: int = 8  # below → near duplicate
DHASH_GAP_MERGE_THRESHOLD: int = 10      # very similar → can merge across small gap


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
    thumbnail_size: int = DEFAULT_BURST_THUMBNAIL_SIZE


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
            thumbnail_size=max(
                MIN_BURST_THUMBNAIL_SIZE,
                min(MAX_BURST_THUMBNAIL_SIZE, int(data.get("thumbnail_size", DEFAULT_BURST_THUMBNAIL_SIZE))),
            ),
        )

    def save_profile(self, profile_name: str, settings: BurstToolSettings) -> None:
        profile_key = str(profile_name or "").strip() or "Generic Sport"
        raw = self._load_all()
        raw[profile_key] = {
            "fps_threshold": max(0.1, float(settings.fps_threshold)),
            "keep_per_burst": max(1, int(settings.keep_per_burst)),
            "winner_criteria": str(settings.winner_criteria or "").strip() or default_winner_criteria_text(),
            "thumbnail_size": max(
                MIN_BURST_THUMBNAIL_SIZE,
                min(MAX_BURST_THUMBNAIL_SIZE, int(settings.thumbnail_size)),
            ),
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


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute a 64-bit difference hash (dHash) for *image* using PIL only.

    Resize the image to (hash_size+1) × hash_size grayscale pixels and compare
    adjacent horizontal pixels to produce a *hash_size²*-bit integer.
    """
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Return the number of differing bits between two integer hashes."""
    xor = hash_a ^ hash_b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def pil_laplacian_focus(image: Image.Image, bbox: tuple[int, int, int, int] | None = None) -> float:
    """Estimate image sharpness via Laplacian edge-energy using PIL only.

    Crops to *bbox* (x1, y1, x2, y2) when provided, then downsamples to at
    most 256 px on the long edge before computing edge variance.  Returns a
    non-negative float; higher = sharper.
    """
    try:
        if bbox is not None:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            if x2 <= x1 or y2 <= y1:
                return 0.0
            image = image.crop((x1, y1, x2, y2))

        # Downsample so variance computation is fast
        max_edge = 256
        w, h = image.size
        if max(w, h) > max_edge:
            scale = max_edge / float(max(w, h))
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        pixels = list(edges.getdata())
        n = len(pixels)
        if n == 0:
            return 0.0
        mean = sum(pixels) / n
        variance = sum((p - mean) * (p - mean) for p in pixels) / n
        return float(variance)
    except Exception:
        return 0.0


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
    """Return (unix_timestamp, precision_source) for *image_path*.

    precision_source values:
      "exif_subsec"  – EXIF datetime + subsecond fraction (37522 / 37523 / 37521)
      "exif_whole"   – EXIF datetime only, 1-second resolution
      "mtime"        – filesystem modification time
      "none"         – could not determine
    """
    path = Path(image_path)
    try:
        with Image.open(path) as img:
            exif = img.getexif()
        if exif:
            dt_value = exif.get(36867) or exif.get(36868) or exif.get(306)
            if dt_value:
                dt = datetime.strptime(str(dt_value).strip(), "%Y:%m:%d %H:%M:%S")
                frac = 0.0
                # Prefer SubSecTimeOriginal (37522), then SubSecTimeDigitized (37523),
                # then SubSecTime (37521) as last resort.
                subsec = exif.get(37522) or exif.get(37523) or exif.get(37521)
                if subsec is not None:
                    digits = "".join(ch for ch in str(subsec) if ch.isdigit())
                    if digits:
                        frac = float(f"0.{digits}")
                        return dt.timestamp() + frac, "exif_subsec"
                return dt.timestamp(), "exif_whole"
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
    previous_source: str | None = None
    previous_path: Path | None = None

    for path in ordered_paths:
        path = Path(path)
        ts, source = extract_capture_timestamp(path)
        if not current:
            current = [path]
            previous_ts = ts
            previous_source = source
            previous_path = path
            continue

        delta = ts - float(previous_ts or 0.0)
        same_burst = False
        if delta >= 0:
            if source == "mtime" and previous_source == "mtime" and delta == 0:
                # Identical mtimes — use filename sequencing to avoid false merges.
                same_burst = _looks_like_sequential_burst_names(previous_path, path)
            elif source == "exif_whole" and previous_source == "exif_whole" and 0 <= delta <= 1.0:
                # Whole-second EXIF: a 1-second boundary gap is ambiguous — corroborate with names.
                same_burst = _looks_like_sequential_burst_names(previous_path, path)
            else:
                same_burst = delta <= threshold_sec

        if same_burst:
            current.append(path)
        else:
            groups.append(current)
            current = [path]

        previous_ts = ts
        previous_source = source
        previous_path = path

    if current:
        groups.append(current)
    return groups


def _looks_like_sequential_burst_names(previous_path: Path | None, current_path: Path) -> bool:
    """Return True when *current_path* appears to be a sequential continuation of *previous_path*."""
    if previous_path is None:
        return False
    prev_stem = previous_path.stem
    curr_stem = current_path.stem

    prev_digits = ""
    i = len(prev_stem) - 1
    while i >= 0 and prev_stem[i].isdigit():
        prev_digits = prev_stem[i] + prev_digits
        i -= 1

    curr_digits = ""
    j = len(curr_stem) - 1
    while j >= 0 and curr_stem[j].isdigit():
        curr_digits = curr_stem[j] + curr_digits
        j -= 1

    if not prev_digits or not curr_digits:
        return False

    prev_prefix = prev_stem[: len(prev_stem) - len(prev_digits)]
    curr_prefix = curr_stem[: len(curr_stem) - len(curr_digits)]
    if prev_prefix != curr_prefix:
        return False

    try:
        delta = int(curr_digits) - int(prev_digits)
    except Exception:
        return False
    return 0 < delta <= 3


def analyze_bursts(folder: Path, fps_threshold: float) -> BurstAnalysis:
    ordered_paths = list_supported_images(Path(folder))
    all_groups = group_adjacent_images(ordered_paths, fps_threshold)
    burst_groups = [group for group in all_groups if len(group) > 1]
    return BurstAnalysis(
        ordered_paths=ordered_paths,
        all_groups=all_groups,
        burst_groups=burst_groups,
    )
