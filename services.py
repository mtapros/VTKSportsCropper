from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageOps


SUPPORTED_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


class ImageService:
    def list_images(self, folder: Path) -> list[Path]:
        found: list[Path] = []
        for pattern in SUPPORTED_PATTERNS:
            found.extend(folder.glob(pattern))
        return sorted(set(found))

    def load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB").copy()

    def build_thumbnail(self, path: Path, size=(140, 105)) -> Image.Image:
        image = self.load_image(path)
        image.thumbnail(size)
        return image

    def save_crop(self, image: Image.Image, bbox, output_path: Path) -> None:
        cropped = image.crop(bbox.as_tuple())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, quality=95)