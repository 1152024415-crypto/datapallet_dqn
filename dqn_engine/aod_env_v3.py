"""
aod_env.py

A dummy, profile-driven, minute-step RL environment for always-on-display (AOD) recommendations.

Key properties:
- Time step: 1 minute
- Episode: 24h (1440 steps) by default (configurable)
- Partial observability:
    * Always visible (free): time, activity, activity-duration, BT state
    * On-demand (costed): location and scene via probe actions
    * Always visible (free): sound intensity and light intensity
- Scenario "gates" (hidden truth -> oracle label):
    * Each gate can be rewarded once per activation window
    * Reset hysteresis prevents spamming repeated triggers
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from types import SimpleNamespace
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
import csv
import os

import numpy as np

# from generator import Segment, build_base_skeleton, normalize_segments

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
    _fmt_hhmm,
    TRAIN_EPISODES,
    EVAL_EPISODES,
)


def _compute_durations(values: List[int]) -> List[int]:
    """Compute consecutive duration counters for a sequence of values."""
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
    """
    Transition-based history for act/loc/scene/light.

    For each attribute:
      - Track segments (runs where id constant).
      - For current segment at time i: duration = step.<dur> (from file)  [0,1,2,...]
      - For *ended* segments: duration = final length of that segment (fixed)

    History order: [current_segment, prev_segment, prev2, prev3]
    If fewer segments exist: pad with (0, 0).
    """
    if not steps:
        return steps

    PAD: Tuple[int, int] = (0, 0)

    def _hist_for(attr_id: str, attr_dur: str) -> List[List[Tuple[int, int]]]:
        out: List[List[Tuple[int, int]]] = []

        # ended segments in chronological order: [(id, final_len), ...]
        ended: List[Tuple[int, int]] = []

        cur_id: Optional[int] = None
        cur_last_dur: int = 0

        for i, step in enumerate(steps):
            sid = int(getattr(step, attr_id, 0) or 0)
            sdur = int(getattr(step, attr_dur, 0) or 0)

            if cur_id is None:
                cur_id = sid
                cur_last_dur = sdur
            elif sid != cur_id:
                # finalize previous segment (0-based dur => length = last_dur + 1)
                ended.append((int(cur_id), int(cur_last_dur) + 1))
                cur_id = sid
                cur_last_dur = sdur
            else:
                cur_last_dur = sdur

            hist: List[Tuple[int, int]] = [(int(cur_id), int(cur_last_dur))]
            # append last ended segments (most recent first)
            for seg in reversed(ended[-(history_len - 1):]):
                hist.append(seg)

            if len(hist) < history_len:
                hist.extend([PAD] * (history_len - len(hist)))

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


def log_step_with_history(step, idx: int | None = None, logger: Logger = None):
    prefix = f"[step {idx}] " if idx is not None else ""
    logger.info(prefix + f"t={step.t}")
    logger.info(f"  act   : id={step.act}, dur={step.act_dur}, hist={step.activities}")
    logger.info(f"  loc   : id={step.loc}, dur={step.loc_dur}, hist={step.locations}")
    logger.info(f"  scene : id={step.scene}, dur={step.scene_dur}, hist={step.scenes}")
    logger.info(f"  light : id={step.light}, dur={step.light_dur}, hist={step.lights}")
    logger.info(f"  gt    : {step.gt_id}")


def _parse_hhmmss_to_seconds(s: Any) -> int:
    """
    "08:00:10" -> 28810
    Returns 0 if missing/bad.
    """
    try:
        hh, mm, ss = str(s).strip().split(":")
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        return 0


def generate_day(
        rng: Optional[np.random.Generator] = None,
        day_id: int = 0,
        logger=None,
        history_len: int = 0,
        jsonl_dir: str = "trajectories",
        # jsonl_dir: str = "/home/mohan/SFM/SFM_HQ_demo/10k_samples_segment_balanced_trajectories"
) -> List["TruthStep"]:
    """
    Loads one jsonl file (day_id selects file in directory) with format:

      {"trajectory_id": ..., "timestep": 0, "time": "08:00:00", ...}
      {"trajectory_id": ..., "timestep": 1, "time": "08:00:10", ...}
      ...

    Builds TruthStep list, preserving:
      - timestep (as step.step_idx)
      - time string (optional)
      - time-of-day seconds (as step.tod_s)  <-- used for time encoding
    """

    # jsonl_list = sorted([f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl") or True])
    # if day_id < 0 or day_id >= len(jsonl_list):
    #     raise IndexError(f"day_id={day_id} out of range, #files={len(jsonl_list)}")

    chosen_jsonl_path = os.path.join(jsonl_dir, f"U0{day_id}.jsonl")

    # -------- load jsonl --------
    records: List[Dict[str, Any]] = []
    with open(chosen_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records in {chosen_jsonl_path}")

    # -------- group by trajectory_id (keep your behavior: smallest id) --------
    traj_map: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        traj_id = int(rec.get("trajectory_id", 0) or 0)
        traj_map.setdefault(traj_id, []).append(rec)

    chosen_traj_id = sorted(traj_map.keys())[0]
    step_recs = traj_map[chosen_traj_id]
    step_recs.sort(key=lambda r: int(r.get("timestep", 0) or 0))

    # -------- id dictionaries --------
    act_id = {name: i for i, name in enumerate(ACTIVITIES)}
    loc_id = {name: i for i, name in enumerate(LOCATIONS)}
    scene_id = {name: i for i, name in enumerate(SCENES)}
    light_id = {name: i for i, name in enumerate(LIGHT_LEVELS)}

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
        return mapping.get(_normalize(raw), "stationary")

    def _map_location(raw: Any) -> str:
        mapping = {
            "work": "Work",
            "street": "Street",
            "subway_station": "Subway Station",
            "subway station": "Subway Station",
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
            "dining": "dining_room",
            "cafeteria": "cafeteria",
            "food_court": "food_court",
            "park": "park",
        }
        return mapping.get(_normalize(raw), "other")

    def _map_light(raw: Any) -> str:
        mapping = {
            "moderate_brightness": "moderate",
            "bright": "bright",
            "dim": "dim",
            "dark": "extremely_dark",
            "extremely_dark": "extremely_dark",
            "harsh_light": "harsh",
            "harsh": "harsh",
        }
        return mapping.get(_normalize(raw), "moderate")

    def _map_gt_action(raw: Any) -> str:
        mapping = {
            "none": "NONE",
            "aod_steps": "step_count",
            "transit_qr": "transit_QR_code",
            "silent_mode": "silent_DND",
            "open_music": "Play Music/news",
            "open_news": "Play Music/news",
            "relaxation": "relax",
        }
        return mapping.get(_normalize(raw), "NONE")

    normal_sound = SOUND_LEVELS.index("normal") if "normal" in SOUND_LEVELS else 0

    steps: List[TruthStep] = []

    for rec in step_recs:
        step_idx = int(rec.get("timestep", 0) or 0)

        time_str = rec.get("time", "00:00:00")
        tod_s = _parse_hhmmss_to_seconds(time_str)

        act = act_id.get(_map_activity(rec.get("activity")), act_id.get("stationary", 0))
        loc = loc_id.get(_map_location(rec.get("location")), loc_id.get("other", 0))
        scene = scene_id.get(_map_scene(rec.get("scene")), scene_id.get("other", 0))
        light = light_id.get(_map_light(rec.get("light")), light_id.get("moderate", 0))
        gt_action = _map_gt_action(rec.get("gt_action", "none"))

        step = TruthStep(t=tod_s, activities=[], locations=[], scenes=[], lights=[])
        # If your TruthStep supports extra attrs, store them:
        step.step_idx = step_idx  # original timestep index
        step.time_str = time_str  # "08:00:10"
        step.tod_s = tod_s  # seconds since midnight (used as t)

        step.act = act
        step.loc = loc
        step.scene = scene
        step.light = light

        # durations from jsonl (already correct per-step)
        step.act_dur = int(rec.get("activity_dur", 0) or 0)
        step.loc_dur = int(rec.get("location_dur", 0) or 0)
        step.scene_dur = int(rec.get("scene_dur", 0) or 0)
        step.light_dur = int(rec.get("light_dur", 0) or 0)

        step.sound = normal_sound

        # flags
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

        step.gt_id = gt_action
        steps.append(step)

    # Sort by actual time-of-day seconds (or by step_idx if you prefer)
    steps.sort(key=lambda s: int(getattr(s, "tod_s", getattr(s, "t", 0)) or 0))

    # Attach transition history for truth activities/lights (loc/scene handled by env observation)
    steps = _attach_recent_history(steps, history_len=history_len + 1)

    return steps


# ============================================================================
# Main Environment Class
# ============================================================================

class AODRecommendationEnv:
    """
    Always-On-Display Recommendation Environment.

    A minute-step RL environment where the agent must decide when to probe
    for information and when to recommend actions based on partial observations.
    """

    def __init__(
            self,
            episode_steps: int = 24 * 60,
            reliability: float = 1,
            gate_reset_hysteresis_min: int = 3,
            seed: int = 0,
            r_success: float = 1.0,
            r_wrong: float = -2.0,
            r_miss: float = -1.0,
            r_delay: float = 0,  # Increased from -0.01 to encourage action
            r_redundant: float = 0,
            history_len: int = 0,
    ):
        self.episode_steps = int(episode_steps)

        # Reliability and gate settings
        self.reliability = float(reliability)
        self.reset_hyst = int(gate_reset_hysteresis_min)

        # Reward settings
        self.r_success = float(r_success)
        self.r_wrong = float(r_wrong)
        self.r_miss = float(r_miss)
        self.r_delay = float(r_delay)
        self.r_redundant = float(r_redundant)

        # Random number generator
        self._rng = np.random.default_rng(seed)

        # Action mappings
        self.probe_ids = [ACTION_TO_ID[a] for a in PROBE_ACTIONS]
        self.none_id = ACTION_TO_ID["NONE"]
        self.reco_ids = [ACTION_TO_ID[a] for a in RECOMMEND_ACTIONS if a != "NONE"]

        # Observation dimensions
        self.n_act = len(ACTIVITIES)
        self.n_loc = len(LOCATIONS)
        self.n_scene = len(SCENES)
        self.n_sound = len(SOUND_LEVELS)
        self.n_light = len(LIGHT_LEVELS)

        # self.unknown_loc = self.n_loc
        # self.unknown_scene = self.n_scene
        # self.unknown_sound = self.n_sound
        # self.unknown_light = self.n_light

        self.unknown_loc = 0
        self.unknown_scene = 0
        self.unknown_sound = 0
        self.unknown_light = 0

        # Observation vector (must match `_make_obs` exactly):
        # - time: 2
        # - activity history (t..t-3): 4 * (n_act + 1)
        # - location history (t..t-3): 4 * (n_loc + 1)
        # - light history (t..t-3): 4 * (n_light + 1)
        # - scene history (t..t-3): 4 * (n_scene + 1)

        self.history_len = history_len

        self.obs_dim = (
                2
                + (self.history_len + 1) * (self.n_act + 1)
                + (self.history_len + 1) * (self.n_loc + 1)
                + (self.history_len + 1) * (self.n_light + 1)
                + (self.history_len + 1) * (self.n_scene + 1)
        )
        # self.obs_dim = (
        #     2
        #     + 1 * (self.n_act + 1)
        #     + 1 * (self.n_loc + 1)
        #     + 1 * (self.n_light + 1)
        #     + 1 * (self.n_scene + 1)
        # )

        self.action_n = len(ALL_ACTIONS)

        # Episode state
        self._day: List[TruthStep] = []
        self._t = 0
        self._profile: Optional[Any] = None

        # Observation state (partial observability)
        self._loc_obs = self.unknown_loc
        self._scene_obs = self.unknown_scene
        self._sound_obs = self.unknown_sound
        self._light_obs = self.unknown_light
        self._age_loc = 999
        self._age_scene = 999
        self._age_sound = 999
        self._age_light = 999
        self._loc_hist: List[Tuple[int, int]] = [(0, 0)] * (self.history_len + 1)
        self._scene_hist: List[Tuple[int, int]] = [(0, 0)] * (self.history_len + 1)
        self._last_loc_value: Optional[int] = None
        self._last_scene_value: Optional[int] = None
        self._loc_run_dur = 0
        self._scene_run_dur = 0

        # Gate state
        self.gates: Dict[str, Gate] = {k: Gate() for k in PRIORITY}
        self.stats: Dict[str, float] = {}

        # Track recent probes to reward probe-then-recommend sequences
        self._recent_probes: List[Tuple[int, str]] = []  # List of (steps_ago, probe_type)
        self._probe_history_window: int = 5  # Consider probes within last 5 steps

    # ========================================================================
    # Public API: Reset and Step
    # ========================================================================

    def reset(
            self,
            seed: Optional[int] = None,
            profile_name: Optional[str] = None,
            day_id: int = 0,
            logger: Logger = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment and start a new episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._profile = SimpleNamespace(name=profile_name or "default")

        # Generate day
        self._day = generate_day(rng=self._rng, day_id=day_id, logger=logger, history_len=self.history_len)
        self._t = 0
        self.episode_steps = min(self.episode_steps, len(self._day))

        # # Export day for debugging
        # Reset observation state
        self._loc_obs = self.unknown_loc
        self._scene_obs = self.unknown_scene
        # Sound/light are always observable (no probe needed)
        self._sound_obs = self._day[self._t].sound
        self._light_obs = self._day[self._t].light
        self._age_loc = self._age_scene = self._age_sound = self._age_light = 999
        self._age_sound = 0
        self._age_light = 0
        self._loc_hist = [(0, 0)] * (self.history_len + 1)
        self._scene_hist = [(0, 0)] * (self.history_len + 1)
        self._last_loc_value = None
        self._last_scene_value = None
        self._loc_run_dur = 0
        self._scene_run_dur = 0
        self._update_observed_history()

        # Reset gates
        for k in self.gates:
            self.gates[k] = Gate(active=False, fired=False, off_counter=999)

        # Reset probe tracking
        self._recent_probes = []

        # Reset stats
        self.stats = {
            "return": 0.0, "sensor_cost": 0.0, "wrong": 0.0, "miss": 0.0,
            "success": 0.0, "redundant": 0.0, "delay_pen": 0.0, "steps": 0.0
        }
        for k in PRIORITY:
            self.stats[f"succ_{k}"] = 0.0
            self.stats[f"miss_{k}"] = 0.0

        obs = self._make_obs(self._day[self._t])
        return obs, {"profile": self._profile.name}

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, bool, Dict, Optional[TruthStep]]:
        """Execute one step in the environment."""
        if action_id < 0 or action_id >= self.action_n:
            raise ValueError(f"Invalid action_id={action_id}")

        action = ID_TO_ACTION[action_id]
        truth = self._day[self._t]

        # Update gates and get oracle action
        oracle_action, oracle_gate = self._update_gates_and_oracle(truth)

        # Calculate reward
        reward = 0.0

        # Apply action
        if action in PROBE_ACTIONS:
            # Probe action: cost and update observation
            c = PROBE_COST[action]
            reward -= c
            self.stats["sensor_cost"] += c

            self._apply_probe(action, truth)

            # Track this probe for potential future reward
            self._recent_probes.append((0, action))

            reward += self.r_miss
            self.stats["miss"] += 1.0
        else:
            # Recommend action: check against oracle
            if oracle_action == "NONE":
                if action != "NONE":
                    reward += self.r_wrong
                    self.stats["wrong"] += 1.0
                # else:
                #     reward += self.r_success
            else:
                if action == oracle_action:
                    reward += self.r_success
                    self.gates[oracle_gate].fired = True
                    self.stats["success"] += 1.0
                    self.stats[f"succ_{oracle_gate}"] += 1.0
                elif action == "NONE":
                    reward += self.r_miss
                    self.stats["miss"] += 1.0
                else:
                    reward += self.r_wrong
                    self.stats["wrong"] += 1.0

        # Advance time
        self._t += 1
        terminated = (self._t >= self.episode_steps)
        truncated = False

        # Age observations
        self._age_loc = min(self._age_loc + 1, 999)
        self._age_scene = min(self._age_scene + 1, 999)
        # Sound/light are always observable, so keep them fresh
        self._age_sound = 0
        self._age_light = 0

        # Age probe history (increment steps_ago for all probes)
        self._recent_probes = [(steps_ago + 1, probe_type)
                               for steps_ago, probe_type in self._recent_probes
                               if steps_ago + 1 < self._probe_history_window * 2]

        # Update stats
        self.stats["return"] += reward
        self.stats["steps"] += 1.0

        # Get next observation
        next_truth = self._day[min(self._t, len(self._day) - 1)]
        # Keep sound/light always observed (no probe needed)
        self._sound_obs = next_truth.sound
        self._light_obs = next_truth.light
        self._update_observed_history()
        obs = self._make_obs(next_truth)

        return obs, float(reward), terminated, truncated, {"profile": self._profile.name, "oracle": oracle_action}

    # ========================================================================
    # Observation Construction
    # ========================================================================

    # def _time_features(self, t: int) -> Tuple[float, float]:
    #     """Convert time to sin/cos features for circular encoding."""
    #     theta = 2.0 * np.pi * (t / (24.0*60.0))
    #     return float(np.sin(theta)), float(np.cos(theta))

    def _time_features(self, t_seconds: int) -> Tuple[float, float]:
        """t_seconds = seconds since midnight (0..86399+)"""
        theta = 2.0 * np.pi * (float(t_seconds) / 86400.0)
        return float(np.sin(theta)), float(np.cos(theta))

    def _one_hot(self, idx: int, size: int) -> np.ndarray:
        """Create one-hot encoding vector."""
        v = np.zeros((size,), dtype=np.float32)
        if 0 <= idx < size:
            v[idx] = 1.0
        return v

    def _make_obs(self, truth: TruthStep) -> np.ndarray:
        """Construct observation vector from truth step."""

        def _encode_history(pairs: List[Tuple[int, int]], size: int, max_dur: int) -> List[float]:
            encoded: List[float] = []
            for idx, dur in pairs[:self.history_len + 1]:
                encoded.extend(self._one_hot(idx, size).tolist())
                encoded.append(float(min(dur, max_dur)) / float(max_dur))
                # encoded.append(float(dur))
            return encoded

        time_sin, time_cos = self._time_features(truth.t)

        if self.history_len > 0:
            act_hist = _encode_history(truth.activities, self.n_act, 180 * 6)
            light_hist = _encode_history(truth.lights, self.n_light, 180 * 6)
            loc_hist = _encode_history(self._loc_hist, self.n_loc, 180 * 6)
            scene_hist = _encode_history(self._scene_hist, self.n_scene, 180 * 6)

            obs = np.concatenate([
                np.array([time_sin, time_cos], dtype=np.float32),
                np.array(act_hist, dtype=np.float32),
                np.array(loc_hist, dtype=np.float32),
                np.array(light_hist, dtype=np.float32),
                np.array(scene_hist, dtype=np.float32),
            ]).astype(np.float32)

        else:
            act_hist = self._one_hot(truth.act, self.n_act)
            act_dur = min(truth.act_dur, 180 * 6) / (180.0 * 6)
            light_hist = self._one_hot(truth.light, self.n_light)
            light_dur = min(truth.light_dur, 180 * 6) / (180.0 * 6)

            loc_hist = self._one_hot(self._loc_obs, self.n_loc)
            loc_dur = min(self._age_loc, 180 * 6) / (180.0 * 6)
            scene_hist = self._one_hot(self._scene_obs, self.n_scene)
            scene_dur = min(self._age_scene, 180 * 6) / (180.0 * 6)

            obs = np.concatenate([
                np.array([time_sin, time_cos], dtype=np.float32),
                np.array(act_hist, dtype=np.float32),
                np.array([act_dur], dtype=np.float32),
                np.array(loc_hist, dtype=np.float32),
                np.array([loc_dur], dtype=np.float32),
                np.array(light_hist, dtype=np.float32),
                np.array([light_dur], dtype=np.float32),
                np.array(scene_hist, dtype=np.float32),
                np.array([scene_dur], dtype=np.float32),
            ]).astype(np.float32)

        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"obs_dim mismatch: {obs.shape[0]} vs {self.obs_dim}")

        return obs

    def _update_observed_history(self) -> None:
        # unknown is 0 by your vocab definition
        loc_value = 0 if self._loc_obs == self.unknown_loc else int(self._loc_obs)
        scene_value = 0 if self._scene_obs == self.unknown_scene else int(self._scene_obs)

        target_len = self.history_len + 1

        # ---------- LOC ----------
        if loc_value == 0:
            # unknown: reset run tracking, don't push transition
            self._last_loc_value = None
            self._loc_run_dur = 0
        else:
            if self._last_loc_value is None:
                # first known observation => start new segment
                self._last_loc_value = loc_value
                self._loc_run_dur = 0
                self._loc_hist = [(loc_value, 0)] + self._loc_hist[: self.history_len]
            elif self._last_loc_value == loc_value:
                # same segment => increment run duration AND update head in-place
                self._loc_run_dur += 1
                self._loc_hist[0] = (loc_value, self._loc_run_dur)
            else:
                # transition => start new segment at head
                self._last_loc_value = loc_value
                self._loc_run_dur = 0
                self._loc_hist = [(loc_value, 0)] + self._loc_hist[: self.history_len]

        # ---------- SCENE ----------
        if scene_value == 0:
            self._last_scene_value = None
            self._scene_run_dur = 0
        else:
            if self._last_scene_value is None:
                self._last_scene_value = scene_value
                self._scene_run_dur = 0
                self._scene_hist = [(scene_value, 0)] + self._scene_hist[: self.history_len]
            elif self._last_scene_value == scene_value:
                self._scene_run_dur += 1
                self._scene_hist[0] = (scene_value, self._scene_run_dur)
            else:
                self._last_scene_value = scene_value
                self._scene_run_dur = 0
                self._scene_hist = [(scene_value, 0)] + self._scene_hist[: self.history_len]

        # pad/truncate to fixed length
        if len(self._loc_hist) < target_len:
            self._loc_hist += [(0, 0)] * (target_len - len(self._loc_hist))
        else:
            self._loc_hist = self._loc_hist[:target_len]

        if len(self._scene_hist) < target_len:
            self._scene_hist += [(0, 0)] * (target_len - len(self._scene_hist))
        else:
            self._scene_hist = self._scene_hist[:target_len]

    def _apply_probe(self, probe_action: str, truth: TruthStep) -> None:
        """Apply probe action and update observation state."""
        # ok = (self._rng.random() < self.reliability)
        ok = 1
        if probe_action == "QUERY_LOC_GPS":
            self._loc_obs = truth.loc if ok else self.unknown_loc
            self._age_loc = 0
        elif probe_action == "QUERY_VISUAL":
            self._scene_obs = truth.scene if ok else self.unknown_scene
            self._age_scene = 0
        else:
            raise ValueError(probe_action)

    # ========================================================================
    # Scenario Predicates and Gates
    # ========================================================================

    def _predicates(self, truth: TruthStep) -> Dict[str, bool]:
        """Evaluate all scenario predicates for a truth step."""
        # act_idx = {a: i for i, a in enumerate(ACTIVITIES)}
        # scene_idx = {s: i for i, s in enumerate(SCENES)}
        # loc_idx = {l: i for i, l in enumerate(LOCATIONS)}

        # hour = truth.t // 60
        # is_sleep = (hour >= 23 or hour < 7)

        # walk_types = {
        #     act_idx[name]
        #     for name in ("slow walk", "fast walk", "fast run")
        #     if name in act_idx
        # }
        # is_walk_type = truth.act in walk_types
        # is_stationary = truth.act == act_idx.get("stationary")

        # r_step = is_walk_type and (2 <= truth.act_dur)
        # r_relax = is_stationary and (30 <= truth.act_dur) and (not is_sleep)

        # subway_loc = loc_idx.get("Subway Station")
        # subway_scene = scene_idx.get("subway_platform")
        # r_transit = (
        #     (subway_loc is not None and truth.loc == subway_loc) or
        #     (subway_scene is not None and truth.scene == subway_scene)
        # )

        # silent_scenes = {
        #     scene_idx[name]
        #     for name in ("conference_room", "conference_center", "lecture_room")
        #     if name in scene_idx
        # }
        # r_silent = truth.scene in silent_scenes

        # play_music_scenes = {
        #     scene_idx[name]
        #     for name in (
        #         "dining_room",
        #         "cafeteria",
        #         "food_court",
        #         "dining_hall",
        #         "booth_indoor",
        #         "park",
        #         "street",
        #     )
        #     if name in scene_idx
        # }
        # r_play = truth.scene in play_music_scenes

        r_transit = truth.gt_id == "transit_QR_code"
        r_silent = truth.gt_id == "silent_DND"
        r_step = truth.gt_id == "step_count"
        r_relax = truth.gt_id == "relax"
        r_play = truth.gt_id == "Play Music/news"

        return {
            "TRANSIT_QR_CODE": r_transit,
            "SILENT_DND": r_silent,
            "STEP_COUNT": r_step,
            "RELAX": r_relax,
            "PLAY_MUSIC/NEWS": r_play,
        }

    def _gate_to_action(self, gate_key: str) -> str:
        """Map gate key to corresponding action name."""
        mapping = {
            "TRANSIT_QR_CODE": "transit_QR_code",
            "SILENT_DND": "silent_DND",
            "STEP_COUNT": "step_count",
            "RELAX": "relax",
            "PLAY_MUSIC/NEWS": "Play Music/news",
        }
        return mapping[gate_key]

    def _update_gates_and_oracle(self, truth: TruthStep) -> Tuple[str, Optional[str]]:
        """Update all gates based on predicates and return oracle action."""
        pred = self._predicates(truth)

        # Update gate states
        for k in PRIORITY:
            is_active = bool(pred.get(k, False))
            g = self.gates[k]
            g.active = is_active
            if is_active:
                g.off_counter = 0
            else:
                g.off_counter = min(g.off_counter + 1, 999)
                if g.off_counter >= self.reset_hyst:
                    g.fired = False
            self.gates[k] = g

        # Find highest priority active unfired gate
        for k in PRIORITY:
            # if self.gates[k].active and (not self.gates[k].fired):
            if self.gates[k].active:
                return self._gate_to_action(k), k
        return "NONE", None

    # ========================================================================
    # Action Sampling (for testing/debugging)
    # ========================================================================

    def sample_biased_action(self, p_none=0.75, p_probe=0.20, p_reco=0.05) -> int:
        """
        Sample action with biased probabilities.

        Notes:
        - If p_none + p_probe + p_reco != 1, we normalize by sampling u in [0, sum_p).
        """
        sum_p = float(p_none + p_probe + p_reco)
        if sum_p <= 0:
            return self.none_id

        u = float(self._rng.random()) * sum_p
        if u < p_none:
            return self.none_id
        if u < p_none + p_probe:
            return int(self._rng.choice(self.probe_ids))
        return int(self._rng.choice(self.reco_ids))

    def sample_random_action(self) -> int:
        """Sample uniformly random action."""
        return int(self._rng.integers(0, self.action_n))

    # ========================================================================
    # Export and Debugging
    # ========================================================================

    def export_day(self, filepath: str, make_timeline_txt: bool = True, compress: bool = True) -> None:
        """
        Export day data to CSV and optionally TXT timeline.

        Writes:
        1) CSV: one row per minute (detailed truth + oracle)
        2) TXT: human-readable timeline (optional), compressed into segments

        Side-effect safe: restores env state after.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # Snapshot state so we don't affect training
        t_snap = self._t
        gates_snap = deepcopy(self.gates)
        age_loc_snap = self._age_loc
        age_scene_snap = self._age_scene
        age_sound_snap = self._age_sound
        age_light_snap = self._age_light
        loc_obs_snap = self._loc_obs
        scene_obs_snap = self._scene_obs
        sound_obs_snap = self._sound_obs
        light_obs_snap = self._light_obs

        try:
            # Fresh gates so oracle computation starts clean
            self.gates = {k: Gate(active=False, fired=False, off_counter=999) for k in PRIORITY}

            # Scenario keys
            scenario_keys = list(PRIORITY)

            headers = [
                          "t_min", "time", "act", "act_dur", "loc", "scene",
                          "bt", "sound", "light", "low_density",
                          "arrival_subway", "arrival_train", "arrival_airport",
                          "important_station", "moment_event", "gaze_event",
                          "semi_acquainted", "idea_event", "earphones_event",
                          "meeting_entry", "driving_entry",
                          "oracle_action", "oracle_gate"
                      ] + scenario_keys

            # Build CSV rows
            rows = []
            oracle_counts = Counter()
            act_counts = Counter()
            loc_counts = Counter()
            scene_counts = Counter()

            for step in self._day:
                oracle_action, oracle_gate = self._update_gates_and_oracle(step)
                scenario_predicates = self._predicates(step)

                act_name = ACTIVITIES[step.act]
                loc_name = LOCATIONS[step.loc]
                scene_name = SCENES[step.scene]
                sound_name = SOUND_LEVELS[step.sound] if step.sound < len(SOUND_LEVELS) else step.sound
                light_name = LIGHT_LEVELS[step.light] if step.light < len(LIGHT_LEVELS) else step.light

                oracle_counts[oracle_action] += 1
                act_counts[act_name] += 1
                loc_counts[loc_name] += 1
                scene_counts[scene_name] += 1

                row = [
                    step.t, _fmt_hhmm(step.t),
                    act_name, step.act_dur,
                    loc_name, scene_name,
                    int(step.bt), sound_name, light_name, int(step.low_density),
                    int(step.arrival_subway), int(step.arrival_train), int(step.arrival_airport),
                    int(step.important_station), int(step.moment_event), int(step.gaze_event),
                    int(step.semi_acquainted), int(step.idea_event), int(step.earphones_event),
                    int(step.meeting_entry), int(step.driving_entry),
                    oracle_action, (oracle_gate or "")
                ]
                row.extend([1 if scenario_predicates.get(key, False) else 0 for key in scenario_keys])
                rows.append(row)

            # Write CSV
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["profile", self._profile.name])
                writer.writerow(["generated_at", datetime.now().isoformat()])
                writer.writerow(headers)
                writer.writerows(rows)

            # Write TXT timeline if requested
            if make_timeline_txt:
                self._write_timeline_txt(filepath, oracle_counts, act_counts, loc_counts, scene_counts, scenario_keys,
                                         compress)

        finally:
            # Restore state
            self._t = t_snap
            self.gates = gates_snap
            self._age_loc = age_loc_snap
            self._age_scene = age_scene_snap
            self._age_sound = age_sound_snap
            self._age_light = age_light_snap
            self._loc_obs = loc_obs_snap
            self._scene_obs = scene_obs_snap
            self._sound_obs = sound_obs_snap
            self._light_obs = light_obs_snap

    def _write_timeline_txt(
            self,
            csv_path: str,
            oracle_counts: Counter,
            act_counts: Counter,
            loc_counts: Counter,
            scene_counts: Counter,
            scenario_keys: List[str],
            compress: bool
    ) -> None:
        """Write human-readable timeline text file."""
        txt_path = os.path.splitext(csv_path)[0] + ".txt"

        def flags_str(step):
            flags = []
            if step.arrival_subway: flags.append("arrival_subway")
            if step.arrival_train: flags.append("arrival_train")
            if step.arrival_airport: flags.append("arrival_airport")
            if step.important_station: flags.append("important_station")
            if step.meeting_entry: flags.append("meeting_entry")
            if step.driving_entry: flags.append("driving_entry")
            if step.moment_event: flags.append("moment_event")
            if step.gaze_event: flags.append("gaze_event")
            if step.idea_event: flags.append("idea_event")
            if step.earphones_event: flags.append("earphones_event")
            if step.low_density: flags.append("low_density")
            if step.bt: flags.append("bt=1")
            return ",".join(flags) if flags else "-"

        def scenario_flags_str(step):
            """Get scenario predicate flags as string"""
            preds = self._predicates(step)
            active_scenarios = [key for key in scenario_keys if preds.get(key, False)]
            return ",".join(active_scenarios) if active_scenarios else "-"

        timeline_lines: List[str] = []
        if compress and len(self._day) > 0:
            # Compress consecutive steps with same properties
            self.gates = {k: Gate(active=False, fired=False, off_counter=999) for k in PRIORITY}

            tuples = []
            for step in self._day:
                oracle_action, _oracle_gate = self._update_gates_and_oracle(step)
                tuples.append((
                    ACTIVITIES[step.act],
                    LOCATIONS[step.loc],
                    SCENES[step.scene],
                    oracle_action,
                    flags_str(step),
                    scenario_flags_str(step)
                ))

            def flush_segment(si, ei, tup):
                t0 = self._day[si].t
                t1 = self._day[ei].t
                dur = ei - si + 1
                act, loc, scene, oracle_action, flags, scenario_flags = tup
                timeline_lines.append(
                    f"{_fmt_hhmm(t0)}-{_fmt_hhmm(t1)} ({dur:4d}m) | {act:<14} | {loc:<15} | {scene:<16} | flags:{flags:<35} | scenarios:{scenario_flags:<50} | oracle:{oracle_action}"
                )

            prev = tuples[0]
            start_i = 0
            for i in range(1, len(tuples)):
                if tuples[i] != prev:
                    flush_segment(start_i, i - 1, prev)
                    start_i = i
                    prev = tuples[i]
            flush_segment(start_i, len(tuples) - 1, prev)

        else:
            # Non-compressed: 1 line per minute
            self.gates = {k: Gate(active=False, fired=False, off_counter=999) for k in PRIORITY}
            for step in self._day:
                oracle_action, _oracle_gate = self._update_gates_and_oracle(step)
                timeline_lines.append(
                    f"{_fmt_hhmm(step.t)} | {ACTIVITIES[step.act]}({step.act_dur:3d}) | "
                    f"{LOCATIONS[step.loc]} | {SCENES[step.scene]} | flags:{flags_str(step)} | scenarios:{scenario_flags_str(step):<50} | oracle:{oracle_action}"
                )

        # Write file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"profile: {self._profile.name}\n")
            f.write(f"generated_at: {datetime.now().isoformat()}\n")
            f.write(f"T: {len(self._day)} steps\n\n")

            f.write("Top oracle actions (by steps):\n")
            for k, v in oracle_counts.most_common(10):
                f.write(f"  {k}: {v}\n")
            f.write("\nTop activities (by steps):\n")
            for k, v in act_counts.most_common(10):
                f.write(f"  {k}: {v}\n")
            f.write("\nTop locations (by steps):\n")
            for k, v in loc_counts.most_common(10):
                f.write(f"  {k}: {v}\n")
            f.write("\nTop scenes (by steps):\n")
            for k, v in scene_counts.most_common(10):
                f.write(f"  {k}: {v}\n")

            # Count scenario activations
            f.write("\nScenario activations (steps with predicate TRUE):\n")
            scenario_activations = {key: 0 for key in scenario_keys}
            self.gates = {k: Gate(active=False, fired=False, off_counter=999) for k in PRIORITY}
            for step in self._day:
                preds = self._predicates(step)
                for key in scenario_keys:
                    if preds.get(key, False):
                        scenario_activations[key] += 1
            for key in scenario_keys:
                f.write(f"  {key}: {scenario_activations[key]} steps\n")

            f.write("\n--- Timeline ---\n")
            f.write("\n".join(timeline_lines))
            f.write("\n")