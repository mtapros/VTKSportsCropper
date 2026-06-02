from __future__ import annotations

import json
from pathlib import Path

from models import SportProfile


DEFAULT_PROFILES: dict[str, SportProfile] = {
    "Generic Sport": SportProfile(
        name="Generic Sport",
        prompts=["person", "ball", "", ""],
        focus_min=0.0,
        focus_relative=0.0,
        edge_margin=0,
        margin_buffer=20.0,
        main_ratio="4:5",
        auto_rotate=True,
        join_descriptors=False,
        safe_ratios={"2:3": True, "5:7": True, "1:1": False},
        sport_type="generic",
        vl_rubric_name="generic",
        prefer_full_body=False,
        penalize_cropped_feet=False,
        favor_symmetry=False,
        favor_peak_action=False,
        prefer_clean_pose=False,
        prefer_single_subject=False,
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
        vl_rubric_name="generic",
        prefer_full_body=True,
        penalize_cropped_feet=True,
        favor_symmetry=False,
        favor_peak_action=True,
        prefer_clean_pose=True,
        prefer_single_subject=False,
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
                    else:
                        sport_type = "generic"

                prefer_full_body = data.get(
                    "prefer_full_body",
                    data.get("dance_prefer_full_body", sport_type == "dance"),
                )
                penalize_cropped_feet = data.get(
                    "penalize_cropped_feet",
                    data.get("dance_penalize_cropped_feet", sport_type == "dance"),
                )
                favor_symmetry = data.get(
                    "favor_symmetry",
                    data.get("dance_favor_symmetry", False),
                )
                favor_peak_action = data.get(
                    "favor_peak_action",
                    data.get("dance_favor_peak_action", sport_type == "dance"),
                )
                prefer_clean_pose = data.get(
                    "prefer_clean_pose",
                    data.get("dance_prefer_clean_pose", sport_type == "dance"),
                )
                prefer_single_subject = data.get(
                    "prefer_single_subject",
                    data.get("dance_prefer_single_subject", False),
                )

                vl_rubric_name = str(data.get("vl_rubric_name", "generic")).strip() or "generic"

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
                    vl_rubric_name=vl_rubric_name,
                    prefer_full_body=bool(prefer_full_body),
                    penalize_cropped_feet=bool(penalize_cropped_feet),
                    favor_symmetry=bool(favor_symmetry),
                    favor_peak_action=bool(favor_peak_action),
                    prefer_clean_pose=bool(prefer_clean_pose),
                    prefer_single_subject=bool(prefer_single_subject),
                )

            if not profiles:
                self.save(DEFAULT_PROFILES)
                return DEFAULT_PROFILES.copy()

            return profiles

        except Exception:
            self.save(DEFAULT_PROFILES)
            return DEFAULT_PROFILES.copy()

    def save(self, profiles: dict[str, SportProfile]):
        serializable: dict[str, dict] = {}
        for name, profile in profiles.items():
            serializable[name] = {
                "prompts": list(profile.prompts[:4]),
                "focus_min": profile.focus_min,
                "focus_relative": profile.focus_relative,
                "edge_margin": profile.edge_margin,
                "margin_buffer": profile.margin_buffer,
                "main_ratio": profile.main_ratio,
                "auto_rotate": profile.auto_rotate,
                "join_descriptors": profile.join_descriptors,
                "safe_ratios": dict(profile.safe_ratios),
                "sport_type": profile.sport_type,
                "vl_rubric_name": profile.vl_rubric_name,
                "prefer_full_body": profile.prefer_full_body,
                "penalize_cropped_feet": profile.penalize_cropped_feet,
                "favor_symmetry": profile.favor_symmetry,
                "favor_peak_action": profile.favor_peak_action,
                "prefer_clean_pose": profile.prefer_clean_pose,
                "prefer_single_subject": profile.prefer_single_subject,
            }

        self.path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")