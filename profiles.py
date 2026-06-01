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
            }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")