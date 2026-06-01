from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass
class CropBox:
    name: str
    bbox: BoundingBox
    color: str = "#00FF00"


@dataclass
class Detection:
    id: int
    label: str
    bbox: BoundingBox
    color: str = "#00BFFF"
    source: str = "unknown"


@dataclass
class SportProfile:
    name: str
    prompts: list[str] = field(default_factory=lambda: ["athlete", "player", "ball", "uniform"])
    focus_min: float = 5.0
    focus_relative: float = 60.0
    edge_margin: int = 10
    margin_buffer: float = 12.0
    main_ratio: str = "4:5"
    auto_rotate: bool = True
    join_descriptors: bool = True
    safe_ratios: dict[str, bool] = field(default_factory=lambda: {"2:3": True, "5:7": True, "1:1": False})

    sport_type: str = "generic"

    dance_prefer_full_body: bool = True
    dance_penalize_cropped_feet: bool = True
    dance_favor_symmetry: bool = False
    dance_favor_peak_action: bool = True
    dance_prefer_clean_pose: bool = True
    dance_prefer_single_subject: bool = False


@dataclass
class AppState:
    input_folder: Optional[Path] = None
    output_folder: Optional[Path] = None
    image_paths: list[Path] = field(default_factory=list)
    current_index: int = 0
    active_tool_id: str = "ai_crop"
    current_image_path: Optional[Path] = None