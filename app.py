from __future__ import annotations

from pathlib import Path
import tkinter as tk

from ai_crop_tool import AICropTool
from ai_cull_tool import AICullTool
from cached_crop_tool import CachedCropTool
from lmstudio_tool import LMStudioTool
from pipeline_tool import PipelineTool
from core import ImageRepository, get_af_points_and_boxes
from manual_crop_tool import ManualCropTool
from models import AppState, CropBox, SportProfile
from profiles import ProfileStore
from ui import MainWindow


class SportsToolkitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.base_dir = Path(__file__).resolve().parent

        self.state = AppState()
        self.image_repo = ImageRepository()
        self.profile_store = ProfileStore(self.base_dir / "sports_profiles.json")
        self.profiles = self.profile_store.load()

        self.current_image = None
        self.current_overlay_boxes: list[CropBox] = []
        self.current_manual_boxes = []
        self.current_af_points = []
        self.current_af_boxes = []

        self.profile_var = tk.StringVar(value=next(iter(self.profiles.keys())))

        self.tools: dict[str, object] = {}
        self.tools_by_id = self.tools
        self.tool_display_to_id: dict[str, str] = {}
        self.ai_detection_cache: dict[tuple, list] = {}
        self.af_cache: dict[str, tuple[list, list]] = {}

        self._register_tools()
        self.ui = MainWindow(root, self)
        self.ui.set_profiles(self.get_profile_names())
        self.set_active_tool("ai_crop")
        self.apply_profile_to_active_tool(self.get_current_profile())
        self._refresh_header_info()
        self.log("Toolkit ready.")

    def _register_tools(self):
        ai_crop_tool = AICropTool(self)
        manual_tool = ManualCropTool(self)
        ai_cull_tool = AICullTool(self)
        cached_crop_tool = CachedCropTool(self)
        pipeline_tool = PipelineTool(self)
        lmstudio_tool = LMStudioTool(self)

        for tool in [
            ai_crop_tool,
            manual_tool,
            ai_cull_tool,
            cached_crop_tool,
            pipeline_tool,
            lmstudio_tool,
        ]:
            self.tools[tool.tool_id] = tool
            self.tool_display_to_id[tool.display_name] = tool.tool_id

    def log(self, message: str):
        self.ui.append_log(message)

    def get_profile_names(self) -> list[str]:
        return list(self.profiles.keys())

    def get_selected_profile_name(self) -> str:
        return self.profile_var.get()

    def get_current_profile(self) -> SportProfile:
        profile_name = self.profile_var.get()
        return self.profiles.get(profile_name, next(iter(self.profiles.values())))

    def get_current_profile_for_active_tool(self) -> SportProfile:
        active = self.tools[self.state.active_tool_id]
        if hasattr(active, "get_profile_data"):
            return active.get_profile_data()
        return self.get_current_profile()

    def apply_profile_to_active_tool(self, profile: SportProfile):
        active = self.tools[self.state.active_tool_id]
        if hasattr(active, "apply_profile"):
            active.apply_profile(profile)

    def save_profile(self, profile: SportProfile, new_name: str | None = None):
        name = (new_name or profile.name).strip()
        if not name:
            self.log("Profile name cannot be empty.")
            return

        profile.name = name
        self.profiles[name] = profile
        self.profile_store.save(self.profiles)
        self.profile_var.set(name)
        self.ui.set_profiles(self.get_profile_names())
        self.ui.set_selected_profile(name)
        self.log(f"Profile saved: {name}")

    def save_current_profile_from_active_tool(self, new_name: str | None = None):
        active = self.tools[self.state.active_tool_id]
        if not hasattr(active, "get_profile_data"):
            self.log("Active tool does not support profile saving.")
            return
        profile = active.get_profile_data()
        self.save_profile(profile, new_name=new_name)

    def set_input_folder(self, folder: str):
        self.state.input_folder = Path(folder)
        self.ai_detection_cache.clear()
        self.af_cache.clear()
        self.log(f"Input folder set: {self.state.input_folder}")
        self._auto_load_input_folder()

    def _auto_load_input_folder(self):
        if not self.state.input_folder:
            self.state.image_paths = []
            self.state.current_index = 0
            self.state.current_image_path = None
            self.current_image = None
            self.current_overlay_boxes = []
            self.current_manual_boxes = []
            self.current_af_points = []
            self.current_af_boxes = []
            self.ui.set_thumbnail_paths([])
            self.ui.clear_image()
            self.ui.clear_debug_views()
            self._refresh_header_info()
            return

        paths = self.image_repo.list_images(self.state.input_folder)
        self.state.image_paths = paths
        self.state.current_index = 0

        if not paths:
            self.current_image = None
            self.state.current_image_path = None
            self.current_overlay_boxes = []
            self.current_manual_boxes = []
            self.current_af_points = []
            self.current_af_boxes = []
            self.ui.set_thumbnail_paths([])
            self.ui.clear_image()
            self.ui.clear_debug_views()
            self._refresh_header_info()
            self.log("No images found in selected input folder.")
            return

        self.ui.set_thumbnail_paths(paths)
        self._refresh_header_info()
        self.log(f"Loaded {len(paths)} image(s) from input folder.")
        self.load_current_image()

    def set_output_folder(self, folder: str):
        self.state.output_folder = Path(folder)
        self._refresh_header_info()
        self.log(f"Output folder set: {self.state.output_folder}")

    def start_batch(self):
        self._auto_load_input_folder()

    def load_image(self, path: Path):
        self.state.current_image_path = Path(path)
        self.current_image = self.image_repo.load_image(self.state.current_image_path)

        cache_key = str(self.state.current_image_path)
        if cache_key in self.af_cache:
            self.current_af_points, self.current_af_boxes = self.af_cache[cache_key]
        else:
            self.current_af_points, self.current_af_boxes = get_af_points_and_boxes(self.state.current_image_path)
            self.af_cache[cache_key] = (self.current_af_points, self.current_af_boxes)

        return self.current_image

    def load_current_image(self):
        if not self.state.image_paths:
            self.current_image = None
            self.state.current_image_path = None
            self.ui.clear_image()
            self.ui.clear_debug_views()
            self._refresh_header_info()
            return

        path = self.state.image_paths[self.state.current_index]

        try:
            self.load_image(path)
        except Exception as exc:
            self.current_image = None
            self.ui.clear_image()
            self.ui.clear_debug_views()
            self._refresh_header_info()
            self.log(f"Failed to load image: {path.name} ({exc})")
            return

        self.current_overlay_boxes = []
        self.current_manual_boxes = []
        self.ui.set_manual_boxes([])
        self.ui.set_manual_selected_ids(set())
        self.ui.set_overlay_boxes([])
        self.ui.clear_debug_views()
        self.ui.show_image(self.current_image)
        self.ui.highlight_thumbnail_index(self.state.current_index)
        self._refresh_header_info()

        if self.current_af_boxes and self.state.active_tool_id != "cached_crop":
            self.log(f"Loaded {len(self.current_af_boxes)} AF box(es).")
        elif self.current_af_points and self.state.active_tool_id != "cached_crop":
            self.log(f"Loaded {len(self.current_af_points)} AF point(s).")

        self.log(f"Loaded: {path.name}")
        tool = self.tools.get(self.state.active_tool_id)
        if tool and hasattr(tool, "on_image_changed"):
            tool.on_image_changed()

    def set_debug_views(self, views):
        self.ui.set_debug_views(views)

    def clear_debug_views(self):
        self.ui.clear_debug_views()

    def _refresh_header_info(self):
        input_name = self.state.input_folder.name if self.state.input_folder else "—"
        output_name = self.state.output_folder.name if self.state.output_folder else "—"
        filename = self.state.current_image_path.name if self.state.current_image_path else "No image loaded"

        total = len(self.state.image_paths)
        if total > 0:
            index_text = f"{self.state.current_index + 1} / {total}"
        else:
            index_text = "0 / 0"

        self.ui.set_header_info(
            input_name=input_name,
            output_name=output_name,
            filename=filename,
            index_text=index_text,
        )

    def set_active_tool(self, tool_id: str):
        self.state.active_tool_id = tool_id
        tool = self.tools[tool_id]

        self.ui.set_tool_panel(tool.build_panel)
        self.ui.set_active_tool_label(tool.display_name)

        current_profile = self.get_current_profile()
        if hasattr(tool, "apply_profile"):
            tool.apply_profile(current_profile)

        if self.current_image is not None and hasattr(tool, "on_image_changed"):
            tool.on_image_changed()

        self.log(f"Active tool: {tool.display_name}")

    def set_active_tool_by_display_name(self, display_name: str):
        tool_id = self.tool_display_to_id.get(display_name)
        if tool_id:
            self.set_active_tool(tool_id)

    def on_profile_changed(self, profile_name: str):
        if profile_name in self.profiles:
            self.profile_var.set(profile_name)
            self.ui.set_selected_profile(profile_name)
            tool = self.tools[self.state.active_tool_id]
            if hasattr(tool, "apply_profile"):
                tool.apply_profile(self.profiles[profile_name])
            self.log(f"Profile selected: {profile_name}")
            if self.current_image is not None and hasattr(tool, "on_image_changed"):
                tool.on_image_changed()

    def get_tool_display_names(self) -> list[str]:
        return list(self.tool_display_to_id.keys())

    def set_overlays(self, boxes: list[CropBox]):
        if self.state.active_tool_id == "cached_crop":
            self.current_overlay_boxes = list(boxes)
        else:
            af_overlays = [
                CropBox(name=f"AF_{i+1}", bbox=box, color="#FF00FF")
                for i, box in enumerate(self.current_af_boxes)
            ]
            self.current_overlay_boxes = af_overlays + boxes
        self.ui.set_overlay_boxes(self.current_overlay_boxes)

    def set_manual_boxes(self, boxes):
        self.current_manual_boxes = boxes
        self.ui.set_manual_boxes(boxes)

    def set_manual_selected_ids(self, selected_ids):
        self.ui.set_manual_selected_ids(selected_ids)

    def save_current_overlays(self, prefer_name: str | None = None):
        if self.current_image is None or self.state.current_image_path is None:
            return
        if not self.current_overlay_boxes:
            self.log("No overlay crops to save.")
            return

        output_dir = self.state.output_folder or (self.state.current_image_path.parent / "Output")
        base_name = self.state.current_image_path.stem

        saved = 0
        for crop in self.current_overlay_boxes:
            if crop.name.startswith("AF_"):
                continue
            if prefer_name is not None and crop.name != prefer_name:
                continue
            out_path = output_dir / f"{base_name}_{crop.name}.jpg"
            self.image_repo.save_crop(self.current_image, crop.bbox, out_path)
            saved += 1

        if saved == 0 and prefer_name is not None:
            self.log(f'No overlay named "{prefer_name}" found to save.')
        else:
            self.log(f"Saved {saved} crop(s).")

    def approve(self):
        tool = self.tools[self.state.active_tool_id]
        if hasattr(tool, "approve"):
            tool.approve()

    def next_image(self):
        if not self.state.image_paths:
            return
        if self.state.current_index < len(self.state.image_paths) - 1:
            self.state.current_index += 1
            self.load_current_image()
        else:
            self.log("Reached last image.")

    def previous_image(self):
        if not self.state.image_paths:
            return
        if self.state.current_index > 0:
            self.state.current_index -= 1
            self.load_current_image()
        else:
            self.log("Already at first image.")

    def select_image_index(self, index: int):
        if 0 <= index < len(self.state.image_paths):
            self.state.current_index = index
            self.load_current_image()

    def on_manual_detection_clicked(self, detection_id: int):
        tool = self.tools.get("manual_crop")
        if tool and self.state.active_tool_id == "manual_crop":
            tool.toggle_detection(detection_id)


def main():
    root = tk.Tk()
    SportsToolkitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()