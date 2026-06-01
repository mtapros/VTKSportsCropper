from __future__ import annotations

import json
from pathlib import Path
from models import SportProfile


DEFAULT_PROFILES: dict[str, SportProfile] = {
    "Generic Sport": SportProfile(
        name="Generic Sport",
        prompts=["athlete", "player", "ball", "uniform"],
        focus_min=5.0,
        focus_relative=60.0,
        edge_margin=10,
        margin_buffer=12.0,
        main_ratio="4:5",
        auto_rotate=True,
        join_descriptors=True,
        safe_ratios={"2:3": True, "5:7": True, "1:1": False},
        sport_type="generic",
    ),
    "Soccer": SportProfile(
        name="Soccer",
        prompts=["player", "soccer ball", "goalkeeper", "goal"],
        focus_min=5.0,
        focus_relative=60.0,
        edge_margin=10,
        margin_buffer=12.0,
        main_ratio="4:5",
        auto_rotate=True,
        join_descriptors=True,
        safe_ratios={"2:3": True, "5:7": True, "1:1": False},
        sport_type="soccer",
    ),
    "Dance": SportProfile(
        name="Dance",
        prompts=["face", "person", "dancer", ""],
        focus_min=0.0,
        focus_relative=0.0,
        edge_margin=0,
        margin_buffer=5.0,
        main_ratio="4:5",
        auto_rotate=False,
        join_descriptors=False,
        safe_ratios={"2:3": True, "5:7": True, "1:1": False},
        sport_type="dance",
        dance_prefer_full_body=True,
        dance_penalize_cropped_feet=True,
        dance_favor_symmetry=False,
        dance_favor_peak_action=True,
        dance_prefer_clean_pose=True,
        dance_prefer_single_subject=False,
    ),
}


class ProfileStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, SportProfile]:
        if not self.path.exists():
            self.save(DEFAULT_PROFILES)
            return DEFAULT_PROFILES.copy()

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            profiles: dict[str, SportProfile] = {}
            for name, data in raw.items():
                prompts = list(data.get("prompts", ["athlete", "player", "ball", "uniform"]))[:4]
                while len(prompts) < 4:
                    prompts.append("")

                sport_type = str(data.get("sport_type", "")).strip().lower()
                if not sport_type:
                    if "dance" in name.lower():
                        sport_type = "dance"
                    elif "soccer" in name.lower():
                        sport_type = "soccer"
                    else:
                        sport_type = "generic"

                profiles[name] = SportProfile(
                    name=name,
                    prompts=prompts,
                    focus_min=float(data.get("focus_min", 5.0)),
                    focus_relative=float(data.get("focus_relative", 60.0)),
                    edge_margin=int(data.get("edge_margin", 10)),
                    margin_buffer=float(data.get("margin_buffer", 12.0)),
                    main_ratio=str(data.get("main_ratio", "4:5")),
                    auto_rotate=bool(data.get("auto_rotate", True)),
                    join_descriptors=bool(data.get("join_descriptors", True)),
                    safe_ratios=dict(data.get("safe_ratios", {"2:3": True, "5:7": True, "1:1": False})),
                    sport_type=sport_type,
                    dance_prefer_full_body=bool(data.get("dance_prefer_full_body", sport_type == "dance")),
                    dance_penalize_cropped_feet=bool(data.get("dance_penalize_cropped_feet", sport_type == "dance")),
                    dance_favor_symmetry=bool(data.get("dance_favor_symmetry", False)),
                    dance_favor_peak_action=bool(data.get("dance_favor_peak_action", sport_type == "dance")),
                    dance_prefer_clean_pose=bool(data.get("dance_prefer_clean_pose", sport_type == "dance")),
                    dance_prefer_single_subject=bool(data.get("dance_prefer_single_subject", False)),
                )
            return profiles or DEFAULT_PROFILES.copy()
        except Exception:
            self.save(DEFAULT_PROFILES)
            return DEFAULT_PROFILES.copy()

    def save(self, profiles: dict[str, SportProfile]) -> None:
        data = {}
        for name, profile in profiles.items():
            data[name] = {
                "prompts": profile.prompts[:4],
                "focus_min": profile.focus_min,
                "focus_relative": profile.focus_relative,
                "edge_margin": profile.edge_margin,
                "margin_buffer": profile.margin_buffer,
                "main_ratio": profile.main_ratio,
                "auto_rotate": profile.auto_rotate,
                "join_descriptors": profile.join_descriptors,
                "safe_ratios": profile.safe_ratios,
                "sport_type": profile.sport_type,
                "dance_prefer_full_body": profile.dance_prefer_full_body,
                "dance_penalize_cropped_feet": profile.dance_penalize_cropped_feet,
                "dance_favor_symmetry": profile.dance_favor_symmetry,
                "dance_favor_peak_action": profile.dance_favor_peak_action,
                "dance_prefer_clean_pose": profile.dance_prefer_clean_pose,
                "dance_prefer_single_subject": profile.dance_prefer_single_subject,
            }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")