"""
aod_env.py

A dummy, profile-driven, minute-step RL environment for always-on-display (AOD) recommendations.

Key properties:
• Time step: 1 minute

• Episode: 24h (1440 steps) by default (configurable)

• Partial observability:

    ◦ Always visible (free): time, activity, activity-duration, BT state

    ◦ On-demand (costed): location, scene, sound intensity, light intensity via probe actions

• Scenario "gates" (hidden truth -> oracle label):

    ◦ Each gate can be rewarded once per activation window

    ◦ Reset hysteresis prevents spamming repeated triggers

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

LOCATIONS = [
    "Home", "Work", "Bus Station", "Subway Station", "Train Station", "Airport",
    "Accommodation", "Residential", "Commercial", "School", "Health", "Government",
    "Entertainment", "Dining", "Shopping", "Sport", "Attraction", "Park", "Street",
]

SCENES = [
    "null", "other", "one_person", "group_of_people", "workspace", "meeting",
    "service_counter", "payment_counter", "elevator", "parking_lot", "driving"
]

ACTIVITIES = [
    "still", "walking", "brisk_walking", "running",
    "bus", "subway", "train", "car"
]

SOUND_LEVELS = ["very_quiet", "soft", "normal", "noisy", "very_noisy"]
LIGHT_LEVELS = ["extremely_dark", "dim", "moderate", "bright", "harsh"]

PROBE_ACTIONS = [
    "QUERY_LOC_NET",
    "QUERY_LOC_GPS",
    "QUERY_VISUAL",
    "QUERY_SOUND_INTENSITY",
    "QUERY_LIGHT_INTENSITY",
]

PROBE_COST = {
    "QUERY_LOC_NET": 0.10,
    "QUERY_LOC_GPS": 0.25,
    "QUERY_VISUAL": 1.00,
    "QUERY_SOUND_INTENSITY": 0.10,
    "QUERY_LIGHT_INTENSITY": 0.10,
}

RECOMMEND_ACTIONS = [
    "NONE",
    "step_count_and_map",
    "transit_QR_code",
    "train_information",
    "flight_information",
    "payment_QR_code",
    "preferred_APP",
    "glasses_snapshot",
    "identify_person",
    "silent_DND",
    "navigation",
    "audio_record",
    "relax",
    "arrived",
]

ALL_ACTIONS = PROBE_ACTIONS + RECOMMEND_ACTIONS
ACTION_TO_ID = {a: i for i, a in enumerate(ALL_ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}


@dataclass
class Gate:
    active: bool = False
    fired: bool = False
    off_counter: int = 999


PRIORITY = [
    "S14_PAYMENT",
    "S4_AIRPORT",
    "S3_TRAIN_STATION",
    "S2_SUBWAY_STATION",
    "S15_MEETING_ENTRY",
    "S16_DRIVING_APP",
    "S22_EARPHONES_APP",
    "S9_MOMENT_SNAPSHOT",
    "S10_GAZE_ID",
    "S1_STEPS",
    "S12_IDEA_RECORD",
    "S7_RELAX",
    "S8_IMPORTANT_STATION",
    "S20_LOW_DENSITY_NAV",
]


@dataclass
class UserProfile:
    name: str
    commute_mode: str
    work_start: int
    work_end: int
    meetings_per_day: float
    shopping_prob_evening: float
    earphones_events_per_day: float
    idea_events_per_day: float
    moment_events_per_day: float
    gaze_events_per_day: float
    low_density_prob: float
    traveler_prob: float


DEFAULT_PROFILES = [
    UserProfile("Office_Subway", "subway", 9 * 60, 18 * 60, 2.5, 0.35, 2.0, 1.0, 0.3, 0.4, 0.15, 0.05),
    UserProfile("Office_Driver", "car", 9 * 60, 18 * 60, 2.0, 0.30, 1.5, 1.2, 0.25, 0.35, 0.10, 0.05),
    UserProfile("Business_Traveler", "car", 10 * 60, 19 * 60, 3.5, 0.25, 1.8, 1.0, 0.20, 0.55, 0.10, 0.35),
    UserProfile("Student_Campus", "walk", 10 * 60, 16 * 60, 1.0, 0.20, 3.0, 1.5, 0.35, 0.25, 0.20, 0.02),
    UserProfile("Homebody", "walk", 12 * 60, 14 * 60, 0.4, 0.15, 1.0, 0.6, 0.10, 0.10, 0.25, 0.01),
]


def _clip_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


@dataclass
class TruthStep:
    t: int
    act: int
    act_dur: int
    loc: int
    scene: int
    bt: int
    sound: int
    light: int
    low_density: int
    arrival_subway: int = 0
    arrival_train: int = 0
    arrival_airport: int = 0
    important_station: int = 0
    moment_event: int = 0
    gaze_event: int = 0
    semi_acquainted: int = 0
    idea_event: int = 0
    earphones_event: int = 0
    meeting_entry: int = 0
    driving_entry: int = 0


def generate_day(profile: UserProfile, rng: np.random.Generator, T: int = 24 * 60) -> List[TruthStep]:
    loc_id = {name: i for i, name in enumerate(LOCATIONS)}
    scene_id = {name: i for i, name in enumerate(SCENES)}
    act_id = {name: i for i, name in enumerate(ACTIVITIES)}

    is_travel_day = (rng.random() < profile.traveler_prob)

    commute_min = int(rng.integers(10, 60)) if profile.commute_mode in ("subway", "bus", "car") else int(
        rng.integers(5, 25))
    commute_min = _clip_int(commute_min, 5, 60)

    n_meetings = int(rng.poisson(profile.meetings_per_day))
    meeting_starts, meeting_durs = [], []
    for _ in range(n_meetings):
        start = int(rng.integers(profile.work_start + 30, max(profile.work_start + 60, profile.work_end - 30)))
        dur = int(rng.integers(5, 121))
        meeting_starts.append(start)
        meeting_durs.append(dur)

    do_shopping = (rng.random() < profile.shopping_prob_evening)
    shopping_start = int(rng.integers(18 * 60, 21 * 60)) if do_shopping else None
    shopping_dur = int(rng.integers(15, 90)) if do_shopping else 0

    checkout_start, checkout_len = None, 0
    if do_shopping and shopping_start is not None:
        latest = shopping_start + shopping_dur - 1
        earliest = max(shopping_start + 5, latest - 8)
        if earliest < latest:
            checkout_start = int(rng.integers(earliest, latest))
            checkout_len = int(rng.integers(1, 4))

    ear_times = sorted(
        [int(rng.integers(7 * 60, 23 * 60)) for _ in range(int(rng.poisson(profile.earphones_events_per_day)))])
    idea_times = sorted(
        [int(rng.integers(8 * 60, 22 * 60)) for _ in range(int(rng.poisson(profile.idea_events_per_day)))])
    moment_times = sorted(
        [int(rng.integers(10 * 60, 22 * 60)) for _ in range(int(rng.poisson(profile.moment_events_per_day)))])
    gaze_times = sorted([int(rng.integers(profile.work_start, profile.work_end)) for _ in
                         range(int(rng.poisson(profile.gaze_events_per_day)))])

    travel_arrival_time = int(rng.integers(9 * 60, 15 * 60)) if is_travel_day else None
    travel_mode = "airport" if (is_travel_day and rng.random() < 0.6) else ("train" if is_travel_day else None)

    commute_out_start = _clip_int(profile.work_start - commute_min - int(rng.integers(5, 20)), 6 * 60, 10 * 60)
    commute_back_start = _clip_int(profile.work_end + int(rng.integers(5, 40)), 15 * 60, 21 * 60)

    important_station_times = set()
    if profile.commute_mode == "subway":
        important_station_times.add(commute_out_start + commute_min // 2)
        important_station_times.add(commute_back_start + commute_min // 2)

    bt = 0
    act_dur = 0
    prev_act = act_id["still"]

    ear_i = idea_i = moment_i = gaze_i = 0
    steps: List[TruthStep] = []

    for t in range(T):
        hour = t // 60

        ear_event = 0
        if ear_i < len(ear_times) and t == ear_times[ear_i]:
            ear_event = 1
            bt = 1
            ear_i += 1
        if bt == 1 and rng.random() < 0.005:
            bt = 0

        if 22 <= hour or hour < 6:
            light = int(rng.integers(0, 2))
        elif 6 <= hour < 9:
            light = int(rng.integers(1, 3))
        elif 9 <= hour < 17:
            light = int(rng.integers(2, 4))
        else:
            light = int(rng.integers(1, 4))

        sound = 2

        loc = loc_id["Home"]
        scene = scene_id["other"]
        act = act_id["still"]
        arrival_sub = arrival_train = arrival_air = 0
        meeting_entry = 0
        driving_entry = 0
        low_density = 0

        if t < 7 * 60 or t >= 23 * 60:
            loc = loc_id["Home"]
            scene = scene_id["other"]
            act = act_id["still"]
            sound = int(rng.integers(0, 2))
        elif 7 * 60 <= t < commute_out_start:
            loc = loc_id["Home"]
            scene = scene_id["other"]
            act = act_id["still"] if rng.random() < 0.7 else act_id["walking"]
            sound = int(rng.integers(0, 3))
        elif commute_out_start <= t < commute_out_start + commute_min:
            if profile.commute_mode == "subway":
                loc = loc_id["Subway Station"]
                act = act_id["subway"]
                scene = scene_id["other"]
                sound = int(rng.integers(2, 5))
                if t == commute_out_start:
                    arrival_sub = 1
            elif profile.commute_mode == "bus":
                loc = loc_id["Bus Station"]
                act = act_id["bus"]
                scene = scene_id["other"]
                sound = int(rng.integers(2, 5))
            elif profile.commute_mode == "car":
                loc = loc_id["Street"]
                act = act_id["car"]
                scene = scene_id["driving"]
                sound = int(rng.integers(2, 4))
                if t == commute_out_start:
                    driving_entry = 1
            else:
                loc = loc_id["Street"]
                act = act_id["walking"]
                scene = scene_id["other"]
                sound = int(rng.integers(1, 3))
        elif profile.work_start <= t < profile.work_end:
            loc = loc_id["Work"]
            act = act_id["still"]
            scene = scene_id["workspace"]
            sound = int(rng.integers(1, 3))
        elif commute_back_start <= t < commute_back_start + commute_min:
            if profile.commute_mode == "subway":
                loc = loc_id["Subway Station"]
                act = act_id["subway"]
                scene = scene_id["other"]
                sound = int(rng.integers(2, 5))
                if t == commute_back_start:
                    arrival_sub = 1
            elif profile.commute_mode == "bus":
                loc = loc_id["Bus Station"]
                act = act_id["bus"]
                scene = scene_id["other"]
                sound = int(rng.integers(2, 5))
            elif profile.commute_mode == "car":
                loc = loc_id["Street"]
                act = act_id["car"]
                scene = scene_id["driving"]
                sound = int(rng.integers(2, 4))
                if t == commute_back_start:
                    driving_entry = 1
            else:
                loc = loc_id["Street"]
                act = act_id["walking"]
                scene = scene_id["other"]
                sound = int(rng.integers(1, 3))
        else:
            if do_shopping and shopping_start is not None and (shopping_start <= t < shopping_start + shopping_dur):
                loc = loc_id["Shopping"]
                act = act_id["walking"] if rng.random() < 0.6 else act_id["still"]
                scene = scene_id["other"]
                sound = int(rng.integers(2, 5))
            else:
                r = rng.random()
                if r < 0.45:
                    loc = loc_id["Residential"]
                    act = act_id["walking"] if rng.random() < 0.4 else act_id["still"]
                    scene = scene_id["other"]
                    sound = int(rng.integers(0, 3))
                elif r < 0.70:
                    loc = loc_id["Dining"]
                    act = act_id["still"]
                    scene = scene_id["group_of_people"] if rng.random() < 0.6 else scene_id["other"]
                    sound = int(rng.integers(2, 5))
                else:
                    loc = loc_id["Park"]
                    act = act_id["walking"] if rng.random() < 0.8 else act_id["still"]
                    scene = scene_id["group_of_people"] if rng.random() < 0.5 else scene_id["other"]
                    sound = int(rng.integers(1, 4))

            if loc in (loc_id["Street"], loc_id["Park"]):
                low_density = 1 if (rng.random() < profile.low_density_prob) else 0

        if is_travel_day and travel_arrival_time is not None:
            if travel_arrival_time <= t < travel_arrival_time + int(rng.integers(30, 120)):
                if travel_mode == "airport":
                    loc = loc_id["Airport"]
                    act = act_id["walking"] if rng.random() < 0.7 else act_id["still"]
                    scene = scene_id["other"]
                    sound = int(rng.integers(2, 5))
                    if t == travel_arrival_time:
                        arrival_air = 1
                elif travel_mode == "train":
                    loc = loc_id["Train Station"]
                    act = act_id["train"]
                    scene = scene_id["other"]
                    sound = int(rng.integers(2, 5))
                    if t == travel_arrival_time:
                        arrival_train = 1

        for ms, md in zip(meeting_starts, meeting_durs):
            if ms <= t < ms + md:
                loc = loc_id["Work"]
                act = act_id["still"]
                if t == ms:
                    meeting_entry = 1
                scene = scene_id["meeting"]
                sound = int(rng.integers(1, 4))

        if checkout_start is not None and checkout_start <= t < checkout_start + checkout_len:
            loc = loc_id["Shopping"]
            scene = scene_id["payment_counter"]
            act = act_id["still"] if rng.random() < 0.6 else act_id["walking"]
            sound = int(rng.integers(2, 5))

        if act == prev_act:
            act_dur = act_dur + 1
        else:
            act_dur = 0
        prev_act = act

        important_station = 1 if (t in important_station_times) else 0

        moment_event = 0
        if moment_i < len(moment_times) and t == moment_times[moment_i]:
            moment_event = 1
            moment_i += 1

        gaze_event = 0
        semi_acq = 0
        if gaze_i < len(gaze_times) and t == gaze_times[gaze_i]:
            if scene in (scene_id["meeting"], scene_id["group_of_people"], scene_id["one_person"]):
                gaze_event = 1
                semi_acq = 1 if rng.random() < 0.7 else 0
            gaze_i += 1

        idea_event = 0
        if idea_i < len(idea_times) and t == idea_times[idea_i]:
            if scene == scene_id["driving"] or act == act_id["car"]:
                idea_event = 1
            elif rng.random() < 0.3:
                idea_event = 1
            idea_i += 1

        steps.append(TruthStep(
            t=t,
            act=act,
            act_dur=act_dur,
            loc=loc,
            scene=scene,
            bt=1 if ear_event == 1 else bt,
            sound=_clip_int(sound, 0, len(SOUND_LEVELS) - 1),
            light=_clip_int(light, 0, len(LIGHT_LEVELS) - 1),
            low_density=low_density,
            arrival_subway=arrival_sub,
            arrival_train=arrival_train,
            arrival_airport=arrival_air,
            important_station=important_station,
            moment_event=moment_event,
            gaze_event=gaze_event,
            semi_acquainted=semi_acq,
            idea_event=idea_event,
            earphones_event=ear_event,
            meeting_entry=meeting_entry,
            driving_entry=driving_entry,
        ))
    return steps


class AODRecommendationEnv:
    def __init__(
            self,
            profiles: Optional[List[UserProfile]] = None,
            episode_minutes: int = 24 * 60,
            reliability: float = 0.97,
            gate_reset_hysteresis_min: int = 3,
            seed: int = 0,
            r_success: float = 20.0,
            r_wrong: float = -6.0,
            r_miss: float = -20.0,
            r_delay: float = -0.01,
            r_redundant: float = -1.0,
    ):
        self.profiles = profiles if profiles is not None else DEFAULT_PROFILES
        self.episode_minutes = int(episode_minutes)
        self.reliability = float(reliability)
        self.reset_hyst = int(gate_reset_hysteresis_min)

        self.r_success = float(r_success)
        self.r_wrong = float(r_wrong)
        self.r_miss = float(r_miss)
        self.r_delay = float(r_delay)
        self.r_redundant = float(r_redundant)

        self._rng = np.random.default_rng(seed)

        self.probe_ids = [ACTION_TO_ID[a] for a in PROBE_ACTIONS]
        self.none_id = ACTION_TO_ID["NONE"]
        self.reco_ids = [ACTION_TO_ID[a] for a in RECOMMEND_ACTIONS if a != "NONE"]

        self.n_act = len(ACTIVITIES)
        self.n_loc = len(LOCATIONS)
        self.n_scene = len(SCENES)
        self.n_sound = len(SOUND_LEVELS)
        self.n_light = len(LIGHT_LEVELS)

        self.unknown_loc = self.n_loc
        self.unknown_scene = self.n_scene
        self.unknown_sound = self.n_sound
        self.unknown_light = self.n_light

        self.obs_dim = 2 + self.n_act + 2 + (self.n_loc + 1) + (self.n_scene + 1) + (self.n_sound + 1) + (
                    self.n_light + 1) + 4
        self.action_n = len(ALL_ACTIONS)

        self._day: List[TruthStep] = []
        self._t = 0
        self._profile: Optional[UserProfile] = None

        self._loc_obs = self.unknown_loc
        self._scene_obs = self.unknown_scene
        self._sound_obs = self.unknown_sound
        self._light_obs = self.unknown_light
        self._age_loc = 999
        self._age_scene = 999
        self._age_sound = 999
        self._age_light = 999

        self.gates: Dict[str, Gate] = {k: Gate() for k in PRIORITY}
        self.stats: Dict[str, float] = {}

    def sample_biased_action(self, p_none=0.75, p_probe=0.20, p_reco=0.05) -> int:
        u = float(self._rng.random())
        if u < p_none:
            return self.none_id
        if u < p_none + p_probe:
            return int(self._rng.choice(self.probe_ids))
        return int(self._rng.choice(self.reco_ids))

    def reset(self, seed: Optional[int] = None, profile_name: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        if profile_name is None:
            self._profile = self.profiles[int(self._rng.integers(0, len(self.profiles)))]
        else:
            matches = [p for p in self.profiles if p.name == profile_name]
            if not matches:
                raise ValueError(f"Unknown profile_name={profile_name}")
            self._profile = matches[0]

        self._day = generate_day(self._profile, self._rng, T=max(self.episode_minutes, 24 * 60))
        self._t = 0

        self._loc_obs = self.unknown_loc
        self._scene_obs = self.unknown_scene
        self._sound_obs = self.unknown_sound
        self._light_obs = self.unknown_light
        self._age_loc = self._age_scene = self._age_sound = self._age_light = 999

        for k in self.gates:
            self.gates[k] = Gate(active=False, fired=False, off_counter=999)

        self.stats = {"return": 0.0, "sensor_cost": 0.0, "wrong": 0.0, "miss": 0.0, "success": 0.0, "redundant": 0.0,
                      "delay_pen": 0.0, "steps": 0.0}
        for k in PRIORITY:
            self.stats[f"succ_{k}"] = 0.0
            self.stats[f"miss_{k}"] = 0.0

        obs = self._make_obs(self._day[self._t])
        return obs, {"profile": self._profile.name}

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if action_id < 0 or action_id >= self.action_n:
            raise ValueError(f"Invalid action_id={action_id}")
        action = ID_TO_ACTION[action_id]
        truth = self._day[self._t]

        prev_active = {k: self.gates[k].active for k in PRIORITY}
        prev_fired = {k: self.gates[k].fired for k in PRIORITY}

        oracle_action, oracle_gate = self._update_gates_and_oracle(truth)

        reward = 0.0
        if self._any_active_unfired():
            reward += self.r_delay
            self.stats["delay_pen"] += -self.r_delay

        # after update:
        for k in PRIORITY:
            if prev_active[k] and (not self.gates[k].active) and (not prev_fired[k]):
                reward += self.r_miss
                self.stats["miss"] += 1.0
                self.stats[f"miss_{k}"] += 1.0

        if action in PROBE_ACTIONS:
            c = PROBE_COST[action]
            reward -= c
            self.stats["sensor_cost"] += c
            self._apply_probe(action, truth)
        else:
            if oracle_action == "NONE":
                if action != "NONE":
                    reward += self.r_wrong
                    self.stats["wrong"] += 1.0
            else:
                if action == oracle_action:
                    if oracle_gate is not None and (not self.gates[oracle_gate].fired):
                        reward += self.r_success
                        self.gates[oracle_gate].fired = True
                        self.stats["success"] += 1.0
                        self.stats[f"succ_{oracle_gate}"] += 1.0
                    else:
                        reward += self.r_redundant
                        self.stats["redundant"] += 1.0
                else:
                    if action != "NONE":
                        reward += self.r_wrong
                        self.stats["wrong"] += 1.0

        self._t += 1
        terminated = (self._t >= self.episode_minutes)
        truncated = False

        self._age_loc = min(self._age_loc + 1, 999)
        self._age_scene = min(self._age_scene + 1, 999)
        self._age_sound = min(self._age_sound + 1, 999)
        self._age_light = min(self._age_light + 1, 999)

        if terminated:
            reward += self._apply_terminal_misses()

        self.stats["return"] += reward
        self.stats["steps"] += 1.0

        obs = self._make_obs(self._day[min(self._t, len(self._day) - 1)])
        return obs, float(reward), terminated, truncated, {"profile": self._profile.name, "oracle": oracle_action}

    def _time_features(self, t: int) -> Tuple[float, float]:
        theta = 2.0 * np.pi * (t / (24.0 * 60.0))
        return float(np.sin(theta)), float(np.cos(theta))

    def _one_hot(self, idx: int, size: int) -> np.ndarray:
        v = np.zeros((size,), dtype=np.float32)
        if 0 <= idx < size:
            v[idx] = 1.0
        return v

    def _make_obs(self, truth: TruthStep) -> np.ndarray:
        time_sin, time_cos = self._time_features(truth.t)
        act_oh = self._one_hot(truth.act, self.n_act)
        bt = float(truth.bt)
        act_dur = float(min(truth.act_dur, 180)) / 180.0

        loc_oh = self._one_hot(self._loc_obs, self.n_loc + 1)
        scene_oh = self._one_hot(self._scene_obs, self.n_scene + 1)
        sound_oh = self._one_hot(self._sound_obs, self.n_sound + 1)
        light_oh = self._one_hot(self._light_obs, self.n_light + 1)

        age_loc = float(min(self._age_loc, 120)) / 120.0
        age_scene = float(min(self._age_scene, 120)) / 120.0
        age_sound = float(min(self._age_sound, 120)) / 120.0
        age_light = float(min(self._age_light, 120)) / 120.0

        obs = np.concatenate([
            np.array([time_sin, time_cos], dtype=np.float32),
            act_oh,
            np.array([act_dur, bt], dtype=np.float32),
            loc_oh, scene_oh, sound_oh, light_oh,
            np.array([age_loc, age_scene, age_sound, age_light], dtype=np.float32),
        ]).astype(np.float32)
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"obs_dim mismatch: {obs.shape[0]} vs {self.obs_dim}")
        return obs

    def _apply_probe(self, probe_action: str, truth: TruthStep) -> None:
        ok = (self._rng.random() < self.reliability)
        if probe_action in ("QUERY_LOC_NET", "QUERY_LOC_GPS"):
            self._loc_obs = truth.loc if ok else self.unknown_loc
            self._age_loc = 0
        elif probe_action == "QUERY_VISUAL":
            self._scene_obs = truth.scene if ok else self.unknown_scene
            self._age_scene = 0
        elif probe_action == "QUERY_SOUND_INTENSITY":
            self._sound_obs = truth.sound if ok else self.unknown_sound
            self._age_sound = 0
        elif probe_action == "QUERY_LIGHT_INTENSITY":
            self._light_obs = truth.light if ok else self.unknown_light
            self._age_light = 0
        else:
            raise ValueError(probe_action)

    def _predicates(self, truth: TruthStep) -> Dict[str, bool]:
        act_idx = {a: i for i, a in enumerate(ACTIVITIES)}
        scene_idx = {s: i for i, s in enumerate(SCENES)}
        loc_idx = {l: i for i, l in enumerate(LOCATIONS)}

        is_walk_type = truth.act in (act_idx["walking"], act_idx["brisk_walking"], act_idx["running"])
        s1 = is_walk_type and (truth.act_dur >= 1)

        s2 = (truth.arrival_subway == 1)
        s3 = (truth.arrival_train == 1)
        s4 = (truth.arrival_airport == 1)

        s7 = (truth.act == act_idx["still"]) and (truth.act_dur >= 30)
        s8 = (truth.important_station == 1)
        s9 = (truth.moment_event == 1)
        s10 = (truth.gaze_event == 1 and truth.semi_acquainted == 1)
        s12 = (truth.idea_event == 1)
        s14 = (truth.scene == scene_idx["payment_counter"])
        s15 = (truth.meeting_entry == 1)
        s16 = (truth.driving_entry == 1)
        s20 = (truth.low_density == 1 and truth.loc in (loc_idx["Street"], loc_idx["Park"]))
        s22 = (truth.earphones_event == 1)

        return {
            "S1_STEPS": s1,
            "S2_SUBWAY_STATION": s2,
            "S3_TRAIN_STATION": s3,
            "S4_AIRPORT": s4,
            "S7_RELAX": s7,
            "S8_IMPORTANT_STATION": s8,
            "S9_MOMENT_SNAPSHOT": s9,
            "S10_GAZE_ID": s10,
            "S12_IDEA_RECORD": s12,
            "S14_PAYMENT": s14,
            "S15_MEETING_ENTRY": s15,
            "S16_DRIVING_APP": s16,
            "S20_LOW_DENSITY_NAV": s20,
            "S22_EARPHONES_APP": s22,
        }

    def _gate_to_action(self, gate_key: str) -> str:
        mapping = {
            "S1_STEPS": "step_count_and_map",
            "S2_SUBWAY_STATION": "transit_QR_code",
            "S3_TRAIN_STATION": "train_information",
            "S4_AIRPORT": "flight_information",
            "S7_RELAX": "relax",
            "S8_IMPORTANT_STATION": "arrived",
            "S9_MOMENT_SNAPSHOT": "glasses_snapshot",
            "S10_GAZE_ID": "identify_person",
            "S12_IDEA_RECORD": "audio_record",
            "S14_PAYMENT": "payment_QR_code",
            "S15_MEETING_ENTRY": "silent_DND",
            "S16_DRIVING_APP": "preferred_APP",
            "S20_LOW_DENSITY_NAV": "navigation",
            "S22_EARPHONES_APP": "preferred_APP",
        }
        return mapping[gate_key]

    def _update_gates_and_oracle(self, truth: TruthStep) -> Tuple[str, Optional[str]]:
        pred = self._predicates(truth)
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

        for k in PRIORITY:
            if self.gates[k].active and (not self.gates[k].fired):
                return self._gate_to_action(k), k
        return "NONE", None

    def _any_active_unfired(self) -> bool:
        for k in PRIORITY:
            if self.gates[k].active and (not self.gates[k].fired):
                return True
        return False

    def _apply_terminal_misses(self) -> float:
        r = 0.0
        for k in PRIORITY:
            if self.gates[k].active and (not self.gates[k].fired):
                r += self.r_miss
                self.stats["miss"] += 1.0
                self.stats[f"miss_{k}"] += 1.0
        return r

    def sample_random_action(self) -> int:
        return int(self._rng.integers(0, self.action_n))