import torch
import numpy as np
import time
from datetime import datetime
from collections import deque
from typing import Dict, Any, Tuple, List

from dqn_engine.constants import (
    ACTIVITIES, LOCATIONS, SCENES, SOUND_LEVELS, LIGHT_LEVELS, ALL_ACTIONS
)
from dqn_engine.deploy_dqn import load_checkpoint, select_action_greedy, obs_dim_from_history

from datapallet.enums import (
    ActivityMode, LocationType, SceneType, LightIntensity, SoundIntensity
)
from datapallet.datapallet import DataPallet

MAX_DUR_STEPS = 1080


class DQNEngineAdapter:
    def __init__(self, ckpt_path: str, history_len: int = 0, device: str = "cpu"):
        self.device = torch.device(device)
        self.history_len = history_len

        obs_dim = obs_dim_from_history(history_len)
        print(f"[DQN Adapter] Loading model from {ckpt_path}, obs_dim={obs_dim}, device={device}")
        self.q_net = load_checkpoint(
            ckpt_path,
            obs_dim,
            len(ALL_ACTIONS),
            self.device
        )

        self.current_state = {
            "act": 0, "act_dur": 0,
            "loc": 0, "loc_dur": 0,
            "scene": 0, "scene_dur": 0,
            "light": 0, "light_dur": 0
        }

        self.hist_queues = {
            "act": deque(maxlen=max(1, history_len + 1)),
            "loc": deque(maxlen=max(1, history_len + 1)),
            "scene": deque(maxlen=max(1, history_len + 1)),
            "light": deque(maxlen=max(1, history_len + 1))
        }

        for k in self.hist_queues:
            for _ in range(self.hist_queues[k].maxlen):
                self.hist_queues[k].append((0, 0))

        self._init_mappings()

    def _safe_index(self, lst: list, val: str) -> int:
        try:
            return lst.index(val)
        except ValueError:
            return 0

    def _init_mappings(self):
        self.act_map = {
            ActivityMode.NULL: 0,
            ActivityMode.STROLLING: self._safe_index(ACTIVITIES, "slow walk"),
            ActivityMode.BRISK_WALKING: self._safe_index(ACTIVITIES, "fast walk"),
            ActivityMode.RUNNING: self._safe_index(ACTIVITIES, "fast run"),
            ActivityMode.SITTING: self._safe_index(ACTIVITIES, "stationary"),
            ActivityMode.STANDING: self._safe_index(ACTIVITIES, "stationary"),
            ActivityMode.ELEVATOR: self._safe_index(ACTIVITIES, "elevator"),
            # 交通工具统一映射为 stationary TODO
            ActivityMode.CAR: self._safe_index(ACTIVITIES, "stationary"),
            ActivityMode.SUBWAY: self._safe_index(ACTIVITIES, "stationary"),
        }

        # DataPallet LocationType -> DQN constants.LOCATIONS
        # constants: ["unknown", "other", "Work", "Subway Station", "Park", "Street"]
        self.loc_map = {
            LocationType.NULL: 0,
            LocationType.OTHER: self._safe_index(LOCATIONS, "other"),
            LocationType.WORK: self._safe_index(LOCATIONS, "Work"),
            LocationType.SUBWAY_STATION: self._safe_index(LOCATIONS, "Subway Station"),
            LocationType.PARK: self._safe_index(LOCATIONS, "Park"),
            LocationType.STREET: self._safe_index(LOCATIONS, "Street"),
            # 其他未在 DQN 训练集中出现的地点，映射为 other
            LocationType.HOME: self._safe_index(LOCATIONS, "other"),
            LocationType.COMMERCIAL: self._safe_index(LOCATIONS, "other"),
        }

        # TODO 多维标签未处理
        # DataPallet SceneType -> DQN constants.SCENES
        self.scene_map = {
            SceneType.NULL: 0,
            SceneType.OTHER: self._safe_index(SCENES, "other"),
            SceneType.MEETINGROOM: self._safe_index(SCENES, "conference_room"),
            SceneType.WORKSPACE: self._safe_index(SCENES, "office"),
            SceneType.DINING: self._safe_index(SCENES, "dining_room"),
            SceneType.OUTDOOR_PARK: self._safe_index(SCENES, "park"),
            SceneType.SUBWAY_STATION: self._safe_index(SCENES, "subway_platform"),
        }

        self.light_map = {
            LightIntensity.NULL: 0,
            LightIntensity.EXTREMELY_DARK: self._safe_index(LIGHT_LEVELS, "extremely_dark"),
            LightIntensity.DIM: self._safe_index(LIGHT_LEVELS, "dim"),
            LightIntensity.MODERATE_BRIGHTNESS: self._safe_index(LIGHT_LEVELS, "moderate"),
            LightIntensity.BRIGHT: self._safe_index(LIGHT_LEVELS, "bright"),
            LightIntensity.HARSH_LIGHT: self._safe_index(LIGHT_LEVELS, "harsh"),
        }

    def _map_val(self, key: str, value: Any) -> int:
        if isinstance(value, tuple):
            value = value[1]

        if key == "act":
            return self.act_map.get(value, 0)
        elif key == "loc":
            return self.loc_map.get(value, 0)
        elif key == "scene":
            return self.scene_map.get(value, 0)
        elif key == "light":
            return self.light_map.get(value, 0)

        return 0

    def update_and_predict(self, dp: DataPallet) -> str:
        _, raw_act = dp.get("activity_mode")
        _, raw_loc = dp.get("Location")
        _, raw_scene = dp.get("Scence")
        _, raw_light = dp.get("Light_Intensity")

        display_act = raw_act
        if isinstance(raw_act, tuple):
            raw_act = raw_act[1]
            display_act = raw_act

        if hasattr(raw_scene, 'scene_type'):
            display_scene = raw_scene.scene_type
            raw_scene = raw_scene.scene_type
        else:
            display_scene = raw_scene

        new_indices = {
            "act": self._map_val("act", raw_act),
            "loc": self._map_val("loc", raw_loc),
            "scene": self._map_val("scene", raw_scene),
            "light": self._map_val("light", raw_light)
        }

        for k in new_indices:
            if new_indices[k] == self.current_state[k]:
                self.current_state[f"{k}_dur"] = min(self.current_state[f"{k}_dur"] + 1, MAX_DUR_STEPS)
            else:
                self.hist_queues[k].append((self.current_state[k], self.current_state[f"{k}_dur"]))

                self.current_state[k] = new_indices[k]
                self.current_state[f"{k}_dur"] = 0

        obs = self._make_obs()

        action_id = select_action_greedy(self.q_net, obs, self.device)
        action_name = ALL_ACTIONS[action_id]

        try:
            str_act = ACTIVITIES[self.current_state['act']]
            str_loc = LOCATIONS[self.current_state['loc']]
            str_scene = SCENES[self.current_state['scene']]
            str_light = LIGHT_LEVELS[self.current_state['light']]
        except IndexError:
            str_act, str_loc, str_scene, str_light = "err", "err", "err", "err"

        debug_info = (
            f"Input State: "
            f"Act={str_act}({self.current_state['act_dur']}) | "
            f"Loc={str_loc}({self.current_state['loc_dur']}) | "
            f"Light={str_light}({self.current_state['light_dur']}) | "
            f"Scene={str_scene}({self.current_state['scene_dur']})"
        )

        return action_name, debug_info

    def _make_obs(self) -> np.ndarray:
        dt = datetime.now()
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        theta = 2.0 * np.pi * (seconds_since_midnight / 86400.0)
        time_sin, time_cos = float(np.sin(theta)), float(np.cos(theta))

        def _one_hot(idx, size):
            v = np.zeros((size,), dtype=np.float32)
            if 0 <= idx < size: v[idx] = 1.0
            return v

        def _encode_feature(key, n_classes, queue):
            current_pair = (self.current_state[key], self.current_state[f"{key}_dur"])

            history_pairs = [current_pair] + list(reversed(list(queue)))[:self.history_len]

            encoded = []
            for idx, dur in history_pairs:
                encoded.extend(_one_hot(idx, n_classes).tolist())
                encoded.append(float(dur) / float(MAX_DUR_STEPS))
            return encoded

        feats = []
        feats.extend([time_sin, time_cos])

        feats.extend(_encode_feature("act", len(ACTIVITIES), self.hist_queues["act"]))
        feats.extend(_encode_feature("loc", len(LOCATIONS), self.hist_queues["loc"]))
        feats.extend(_encode_feature("light", len(LIGHT_LEVELS), self.hist_queues["light"]))
        feats.extend(_encode_feature("scene", len(SCENES), self.hist_queues["scene"]))

        return np.array(feats, dtype=np.float32)