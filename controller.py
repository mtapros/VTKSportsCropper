from __future__ import annotations

from pathlib import Path
from typing import Optional

from engine import build_demo_crops
from models import SessionState, SportProfile
from profiles import ProfileStore
from services import ImageService


class AppController:
    def __init__(self, profile_store: ProfileStore, image_service: ImageService):
        self.profile_store = profile_store
        self.image_service = image_service

        self.session = SessionState()
        self.profiles = self.profile_store.load_profiles()
        self.current_profile = self.profiles[next(iter(self.profiles.keys()))]

        self.current_image = None
        self.current_crops = []
        self.ui = None

    def attach_ui(self, ui) -> None:
        self.ui = ui
        self.ui.set_profiles(list(self.profiles.keys()))
        self.ui.apply_profile_to_form(self.current_profile)
        self.ui.set_workflow(self.session.workflow)
        self.log("Ready.")

    def log(self, message: str) -> None:
        if self.ui:
            self.ui.append_log(message)

    def set_input_folder(self, folder: str) -> None:
        if not folder:
            return
        self.session.input_folder = Path(folder)
        self.log(f"Input folder set: {self.session.input_folder}")

    def set_output_folder(self, folder: str) -> None:
        if not folder:
            return
        self.session.output_folder = Path(folder)
        self.log(f"Output folder set: {self.session.output_folder}")

    def set_workflow(self, workflow: str) -> None:
        self.session.workflow = workflow
        self.log(f"Workflow set to: {workflow.upper()}")

    def set_profile(self, profile_name: str) -> None:
        profile = self.profiles.get(profile_name)
        if not profile:
            return
        self.current_profile = profile
        if self.ui:
            self.ui.apply_profile_to_form(profile)
        self.log(f"Profile selected: {profile_name}")

    def save_current_profile_from_form(self, form_data: dict) -> None:
        name = form_data["name"]
        profile = SportProfile(
            name=name,
            prompts=form_data["prompts"],
            focus_min=float(form_data["focus_min"]),
            focus_relative=float(form_data["focus_relative"]),
            edge_margin=int(form_data["edge_margin"]),
            margin_buffer=float(form_data["margin_buffer"]),
            main_ratio=form_data["main_ratio"],
            auto_rotate=bool(form_data["auto_rotate"]),
            join_descriptors=bool(form_data["join_descriptors"]),
            safe_ratios=form_data["safe_ratios"],
        )
        self.profiles[name] = profile
        self.current_profile = profile
        self.profile_store.save_profiles(self.profiles)
        if self.ui:
            self.ui.set_profiles(list(self.profiles.keys()))
        self.log(f"Profile saved: {name}")

    def start_batch(self) -> None:
        if not self.session.input_folder:
            self.log("Please select an input folder.")
            return

        image_paths = self.image_service.list_images(self.session.input_folder)
        self.session.image_paths = image_paths
        self.session.current_index = 0

        if not image_paths:
            self.log("No images found.")
            if self.ui:
                self.ui.set_thumbnail_paths([])
                self.ui.clear_image()
            return

        if self.session.output_folder is None:
            self.session.output_folder = self.session.input_folder / "Output"
            self.log(f"Output folder defaulted to: {self.session.output_folder}")

        if self.ui:
            self.ui.set_thumbnail_paths(image_paths)

        self.log(f"Found {len(image_paths)} images.")
        self.load_current_image()

    def load_current_image(self) -> None:
        if not self.session.image_paths:
            return

        path = self.session.image_paths[self.session.current_index]
        self.current_image = self.image_service.load_image(path)

        form = self.ui.get_form_data() if self.ui else None
        ratio = form["main_ratio"] if form else self.current_profile.main_ratio
        margin = float(form["margin_buffer"]) if form else self.current_profile.margin_buffer

        self.current_crops = build_demo_crops(
            self.current_image.width,
            self.current_image.height,
            ratio,
            margin,
        )

        if self.ui:
            self.ui.show_image(self.current_image)
            self.ui.set_crop_boxes(self.current_crops)
            self.ui.highlight_thumbnail_index(self.session.current_index)

        self.log(f"Loaded: {path.name}")

    def approve_and_next(self) -> None:
        if self.current_image is None or not self.session.image_paths:
            return

        current_path = self.session.image_paths[self.session.current_index]
        output_dir = self.session.output_folder or (current_path.parent / "Output")
        base_name = current_path.stem

        for crop in self.current_crops:
            output_path = output_dir / f"{base_name}_{crop.name}.jpg"
            self.image_service.save_crop(self.current_image, crop.bbox, output_path)

        self.log(f"Saved {len(self.current_crops)} crop(s).")
        self.next_image()

    def skip_and_next(self) -> None:
        self.log("Skipped image.")
        self.next_image()

    def next_image(self) -> None:
        if not self.session.image_paths:
            return
        if self.session.current_index < len(self.session.image_paths) - 1:
            self.session.current_index += 1
            self.load_current_image()
        else:
            self.log("Reached last image.")

    def previous_image(self) -> None:
        if not self.session.image_paths:
            return
        if self.session.current_index > 0:
            self.session.current_index -= 1
            self.load_current_image()
        else:
            self.log("Already at first image.")

    def on_thumbnail_selected(self, index: int) -> None:
        if 0 <= index < len(self.session.image_paths):
            self.session.current_index = index
            self.load_current_image()