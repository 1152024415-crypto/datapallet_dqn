"""
aod_env_meeting.py

Environment wrapper to evaluate on meeting-style sensor logs (demo/meeting.json).
The input format is a list of events with timestamps and sensor ids.
We bucket events into fixed step windows (default 10s) and build TruthStep sequences.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import json
import os

import numpy as np

from dqn_engine.constants import (
    LOCATIONS,
    SCENES,
    ACTIVITIES,
    SOUND_LEVELS,
    LIGHT_LEVELS,
    PRIORITY,
    PROBE_ACTIONS,
    PROBE_COST,
    RECOMMEND_ACTIONS,
    ALL_ACTIONS,
    ACTION_TO_ID,
    ID_TO_ACTION,
    Gate,
    TruthStep,
)
from dqn_engine.aod_env_v3 import AODRecommendationEnv


def _fmt_hhmmss(seconds: int) -> str:
    total = int(seconds) % 86400
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_iso_ts(ts: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _compute_durations(values: List[int]) -> List[int]:
    durations: List[int] = []
    prev = None
    dur = 0
    for v in values:
        if prev is None or v != prev:
            dur = 0
        else:
            dur += 1
        durations.append(dur)
        prev = v
    return durations


def _attach_recent_history(steps: List[Any], history_len: int = 3) -> List[Any]:
    if not steps:
        return steps

    pad: Tuple[int, int] = (0, 0)

    def _hist_for(attr_id: str, attr_dur: str) -> List[List[Tuple[int, int]]]:
        out: List[List[Tuple[int, int]]] = []
        ended: List[Tuple[int, int]] = []
        cur_id: Optional[int] = None
        cur_last_dur: int = 0

        for step in steps:
            sid = int(getattr(step, attr_id, 0) or 0)
            sdur = int(getattr(step, attr_dur, 0) or 0)

            if cur_id is None:
                cur_id = sid
                cur_last_dur = sdur
            elif sid != cur_id:
                ended.append((int(cur_id), int(cur_last_dur) + 1))
                cur_id = sid
                cur_last_dur = sdur
            else:
                cur_last_dur = sdur

            hist: List[Tuple[int, int]] = [(int(cur_id), int(cur_last_dur))]
            for seg in reversed(ended[-(history_len - 1):]):
                hist.append(seg)

            if len(hist) < history_len:
                hist.extend([pad] * (history_len - len(hist)))

            out.append(hist)

        return out

    act_hist = _hist_for("act", "act_dur")
    loc_hist = _hist_for("loc", "loc_dur")
    scene_hist = _hist_for("scene", "scene_dur")
    light_hist = _hist_for("light", "light_dur")

    for i, step in enumerate(steps):
        step.activities = act_hist[i]
        step.locations = loc_hist[i]
        step.scenes = scene_hist[i]
        step.lights = light_hist[i]

    return steps


def _normalize(x: Any) -> str:
    return str(x or "").strip().lower()


def _map_activity(raw: Any) -> str:
    mapping = {
        "stationary": "stationary",
        "slow walk": "slow walk",
        "fast walk": "fast walk",
        "fast run": "fast run",
        "running": "fast run",
        "elevator": "elevator",
    }
    return mapping.get(_normalize(raw), "unknown")


def _map_location(raw: Any) -> str:
    mapping = {
        "work": "Work",
        "street": "Street",
        "subway_station": "Subway Station",
        "park": "Park",
    }
    return mapping.get(_normalize(raw), "other")


def _map_scene(raw: Any) -> str:
    mapping = {
        "walking_outdoor": "street",
        "street": "street",
        "subway_platform": "subway_platform",
        "office_work": "office",
        "office": "office",
        "meeting": "conference_room",
        "conference_room": "conference_room",
        "dining": "dining_room",
        "cafeteria": "cafeteria",
        "food_court": "food_court",
        "park": "park",
        "other": "other",
    }
    return mapping.get(_normalize(raw), "other")


def _map_light(raw: Any) -> str:
    mapping = {
        "moderate": "moderate",
        "bright": "bright",
        "dim": "dim",
        "dark": "extremely_dark",
        "extremely_dark": "extremely_dark",
        "harsh_light": "harsh",
        "harsh": "harsh",
    }
    return mapping.get(_normalize(raw), "unknown")


def _map_sound(raw: Any) -> str:
    mapping = {
        "very_quiet": "very_quiet",
        "quiet": "soft",
        "soft": "soft",
        "normal": "normal",
        "noisy": "noisy",
        "very_noisy": "very_noisy",
    }
    return mapping.get(_normalize(raw), "unknown")


def generate_demo_steps(
    meeting_path: str,
    history_len: int = 0,
    step_seconds: int = 10,
    bucket_seconds: int = 10,
) -> List[TruthStep]:
    if not os.path.exists(meeting_path):
        raise FileNotFoundError(meeting_path)

    with open(meeting_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    # Bucket events into fixed windows
    buckets: Dict[int, Dict[str, Any]] = {}
    for ev in events:
        ts = _parse_iso_ts(ev.get("timestamp"))
        if ts is None:
            continue
        tod_s = ts.hour * 3600 + ts.minute * 60 + ts.second
        bucket = (tod_s // bucket_seconds) * bucket_seconds
        bucket_map = buckets.setdefault(bucket, {})
        bucket_map[ev.get("id")] = ev.get("value")

    if not buckets:
        raise ValueError(f"No valid events found in {meeting_path}")

    bucket_keys = sorted(buckets.keys())
    start, end = bucket_keys[0], bucket_keys[-1]
    all_buckets = list(range(start, end + 1, bucket_seconds))

    last_vals: Dict[str, Any] = {}
    act_ids: List[int] = []
    loc_ids: List[int] = []
    scene_ids: List[int] = []
    light_ids: List[int] = []
    sound_ids: List[int] = []
    time_secs: List[int] = []

    act_id_map = {name: i for i, name in enumerate(ACTIVITIES)}
    loc_id_map = {name: i for i, name in enumerate(LOCATIONS)}
    scene_id_map = {name: i for i, name in enumerate(SCENES)}
    light_id_map = {name: i for i, name in enumerate(LIGHT_LEVELS)}
    sound_id_map = {name: i for i, name in enumerate(SOUND_LEVELS)}

    for b in all_buckets:
        bucket_map = buckets.get(b, {})
        for k, v in bucket_map.items():
            last_vals[k] = v

        raw_act = last_vals.get("activity_mode")
        raw_loc = last_vals.get("Location")
        raw_light = last_vals.get("Light_Intensity")
        raw_sound = last_vals.get("Sound_Intensity")
        raw_scene = last_vals.get("Scence")
        if isinstance(raw_scene, dict):
            raw_scene = raw_scene.get("scene_type_str") or raw_scene.get("scene_type")

        act = act_id_map.get(_map_activity(raw_act), act_id_map.get("unknown", 0))
        loc = loc_id_map.get(_map_location(raw_loc), loc_id_map.get("unknown", 0))
        scene = scene_id_map.get(_map_scene(raw_scene), scene_id_map.get("unknown", 0))
        light = light_id_map.get(_map_light(raw_light), light_id_map.get("unknown", 0))
        sound = sound_id_map.get(_map_sound(raw_sound), sound_id_map.get("unknown", 0))

        act_ids.append(act)
        loc_ids.append(loc)
        scene_ids.append(scene)
        light_ids.append(light)
        sound_ids.append(sound)
        time_secs.append(b)

    act_durs = _compute_durations(act_ids)
    loc_durs = _compute_durations(loc_ids)
    scene_durs = _compute_durations(scene_ids)
    light_durs = _compute_durations(light_ids)

    steps: List[TruthStep] = []
    for i, tod_s in enumerate(time_secs):
        step = TruthStep(t=int(tod_s), activities=[], locations=[], scenes=[], lights=[])
        step.step_idx = i
        step.time_str = _fmt_hhmmss(tod_s)
        step.tod_s = int(tod_s)
        step.act = act_ids[i]
        step.loc = loc_ids[i]
        step.scene = scene_ids[i]
        step.light = light_ids[i]
        step.sound = sound_ids[i]
        step.act_dur = act_durs[i]
        step.loc_dur = loc_durs[i]
        step.scene_dur = scene_durs[i]
        step.light_dur = light_durs[i]

        step.bt = 0
        step.low_density = 0
        step.arrival_subway = 0
        step.arrival_train = 0
        step.arrival_airport = 0
        step.important_station = 0
        step.moment_event = 0
        step.gaze_event = 0
        step.semi_acquainted = 0
        step.idea_event = 0
        step.earphones_event = 0
        step.meeting_entry = 0
        step.driving_entry = 0
        step.gt_id = "NONE"

        steps.append(step)

    steps.sort(key=lambda s: int(getattr(s, "tod_s", getattr(s, "t", 0)) or 0))
    steps = _attach_recent_history(steps, history_len=history_len + 1)
    return steps


class AODDemoEnv(AODRecommendationEnv):
    """
    AODRecommendationEnv variant that reads meeting.json instead of jsonl trajectories.
    """

    def __init__(
        self,
        meeting_path: str,
        step_seconds: int = 10,
        bucket_seconds: int = 10,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.meeting_path = str(meeting_path)
        self.step_seconds = int(step_seconds)
        self.bucket_seconds = int(bucket_seconds)

    def reset(
        self,
        seed: Optional[int] = None,
        profile_name: Optional[str] = None,
        day_id: int = 0,
        logger=None,
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        profile = profile_name or Path(self.meeting_path).stem
        self._profile = SimpleNamespace(name=profile)

        self._day = generate_demo_steps(
            meeting_path=self.meeting_path,
            history_len=self.history_len,
            step_seconds=self.step_seconds,
            bucket_seconds=self.bucket_seconds,
        )
        self._t = 0
        self.episode_steps = min(self.episode_steps, len(self._day))

        # Reset observation state
        if self.loc_always_available:
            self._loc_obs = self._day[self._t].loc
        else:
            self._loc_obs = self.unknown_loc
        self._scene_obs = self.unknown_scene
        self._sound_obs = self._day[self._t].sound
        self._light_obs = self._day[self._t].light
        self._age_loc = self._age_scene = self._age_sound = self._age_light = 999
        if self.loc_always_available:
            self._age_loc = 0
        self._age_sound = 0
        self._age_light = 0
        self._loc_hist = [(0, 0)] * (self.history_len + 1)
        self._scene_hist = [(0, 0)] * (self.history_len + 1)
        self._last_loc_value = None
        self._last_scene_value = None
        self._loc_run_dur = 0
        self._scene_run_dur = 0
        self._walk_run_secs = 0
        self._stationary_secs = 0
        self._last_tod_s = None
        self._update_observed_history()

        for k in self.gates:
            self.gates[k] = Gate(active=False, fired=False, off_counter=999)

        self._recent_probes = []
        self.stats = {
            "return": 0.0,
            "sensor_cost": 0.0,
            "wrong": 0.0,
            "miss": 0.0,
            "success": 0.0,
            "redundant": 0.0,
            "delay_pen": 0.0,
            "steps": 0.0,
        }
        for k in PRIORITY:
            self.stats[f"succ_{k}"] = 0.0
            self.stats[f"miss_{k}"] = 0.0

        obs = self._make_obs(self._day[self._t])
        return obs, {"profile": self._profile.name}
