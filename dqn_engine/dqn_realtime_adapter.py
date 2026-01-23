import torch
import numpy as np
import time
from datetime import datetime
from collections import deque
from typing import Dict, Any, Tuple, List

from dqn_engine.constants import (
    ACTIVITIES, LOCATIONS, SCENES, SOUND_LEVELS, LIGHT_LEVELS, ALL_ACTIONS
)
# 复用 deploy_dqn 的逻辑
from dqn_engine.deploy_dqn import load_checkpoint, select_action_greedy

from datapallet.enums import (
    ActivityMode, LocationType, SceneType, LightIntensity, SoundIntensity
)
from datapallet.datapallet import DataPallet

# 与 aod_env_v3.py 和 deploy_dqn.py 保持一致
# 180 * 60 = 10800 秒
MAX_DUR_SECONDS = 180 * 60


class DQNEngineAdapter:
    def __init__(self, ckpt_path: str, history_len: int = 0, device: str = "cpu"):
        self.device = torch.device(device)
        self.history_len = history_len

        # 计算 Observation Dimension
        # 必须与 deploy_dqn.py 中的 obs_dim_from_history 保持一致
        # (Time:2) + (Hist+1)*(Act+1) + (Hist+1)*(Loc+1) + (Hist+1)*(Light+1) + (Hist+1)*(Scene+1) + (Flags:2)
        self.obs_dim = (
                2
                + (self.history_len + 1) * (len(ACTIVITIES) + 1)
                + (self.history_len + 1) * (len(LOCATIONS) + 1)
                + (self.history_len + 1) * (len(LIGHT_LEVELS) + 1)
                + (self.history_len + 1) * (len(SCENES) + 1)
                + 2  # 新增: walk_run_flag, relax_flag 0123版本更新新增
        )

        print(f"[DQN Adapter] Loading model from {ckpt_path}, obs_dim={self.obs_dim}, device={device}")
        self.q_net = load_checkpoint(
            ckpt_path,
            self.obs_dim,
            len(ALL_ACTIONS),
            self.device
        )

        # 状态追踪
        self.current_state = {
            "act": 0, "act_dur": 0,
            "loc": 0, "loc_dur": 0,
            "scene": 0, "scene_dur": 0,
            "light": 0, "light_dur": 0
        }

        # 新增：用于计算 Flags 的累积时间
        self.walk_run_secs = 0.0
        self.stationary_secs = 0.0
        self.last_update_time = time.time()

        # 历史队列
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
        self._init_special_indices()

    def _safe_index(self, lst: list, val: str) -> int:
        try:
            return lst.index(val)
        except ValueError:
            return 0

    def _init_special_indices(self):
        """缓存特定活动的索引，用于计算 Flags"""
        self.idx_slow_walk = self._safe_index(ACTIVITIES, "slow walk")
        self.idx_fast_walk = self._safe_index(ACTIVITIES, "fast walk")
        self.idx_fast_run = self._safe_index(ACTIVITIES, "fast run")
        self.idx_stationary = self._safe_index(ACTIVITIES, "stationary")

        self.walk_run_ids = {self.idx_slow_walk, self.idx_fast_walk, self.idx_fast_run}

    def _init_mappings(self):
        # 1. Activity 映射
        self.act_map = {
            ActivityMode.NULL: 0,
            ActivityMode.STROLLING: self._safe_index(ACTIVITIES, "slow walk"),
            ActivityMode.BRISK_WALKING: self._safe_index(ACTIVITIES, "fast walk"),
            ActivityMode.RUNNING: self._safe_index(ACTIVITIES, "fast run"),
            ActivityMode.SITTING: self._safe_index(ACTIVITIES, "stationary"),
            ActivityMode.STANDING: self._safe_index(ACTIVITIES, "stationary"),
            ActivityMode.ELEVATOR: self._safe_index(ACTIVITIES, "elevator"),
            ActivityMode.CAR: self._safe_index(ACTIVITIES, "stationary"),
            # TODO
            ActivityMode.SUBWAY: self._safe_index(ACTIVITIES, "stationary"),
            ActivityMode.RIDING: self._safe_index(ACTIVITIES, "stationary"),  # 暂映射为 stationary
        }

        # 2. Location 映射
        self.loc_map = {
            LocationType.NULL: 0,
            LocationType.OTHER: self._safe_index(LOCATIONS, "other"),
            LocationType.WORK: self._safe_index(LOCATIONS, "Work"),
            LocationType.SUBWAY_STATION: self._safe_index(LOCATIONS, "Subway Station"),
            LocationType.PARK: self._safe_index(LOCATIONS, "Park"),
            LocationType.STREET: self._safe_index(LOCATIONS, "Street"),
            LocationType.HOME: self._safe_index(LOCATIONS, "other"),
            LocationType.COMMERCIAL: self._safe_index(LOCATIONS, "other"),
            LocationType.Research_Institution: self._safe_index(LOCATIONS, "Work"),
        }

        # 3. Scene 映射 (对应 constants.SCENES)
        self.scene_map = {
            SceneType.NULL: 0,
            SceneType.OTHER: self._safe_index(SCENES, "other"),
            SceneType.MEETINGROOM: self._safe_index(SCENES, "conference_room"),
            SceneType.WORKSPACE: self._safe_index(SCENES, "office"),  # 确保 constants.py 里有 "office"
            SceneType.DINING: self._safe_index(SCENES, "dining_room"),
            SceneType.OUTDOOR_PARK: self._safe_index(SCENES, "park"),
            SceneType.SUBWAY_STATION: self._safe_index(SCENES, "subway_platform"),
        }

        # 4. Light 映射
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

    def update_and_predict(self, dp: DataPallet) -> Tuple[str, str]:
        # 1. 计算时间差 (用于精确累加 Duration)
        now = time.time()
        elapsed = now - self.last_update_time
        self.last_update_time = now
        # 防止过大的时间跳跃（比如断点调试时），最大按 10秒 计算
        if elapsed > 60: elapsed = 1.0

        # 2. 获取数据
        _, raw_act = dp.get("activity_mode")
        _, raw_loc = dp.get("Location")
        success, raw_scene = dp.get("Scene")
        _, raw_light = dp.get("Light_Intensity")

        # 提取枚举值用于显示
        display_act = raw_act[1] if isinstance(raw_act, tuple) else raw_act

        raw_scene_enum = SceneType.NULL
        if hasattr(raw_scene, 'scene_type'):
            raw_scene_enum = raw_scene.scene_type
        elif isinstance(raw_scene, SceneType):
            raw_scene_enum = raw_scene
        elif isinstance(raw_scene, int):
            raw_scene_enum = raw_scene

        # 3. 映射到 DQN Index
        new_indices = {
            "act": self._map_val("act", raw_act),
            "loc": self._map_val("loc", raw_loc),
            "scene": self._map_val("scene", raw_scene_enum),
            "light": self._map_val("light", raw_light)
        }

        # 4. 更新基本状态持续时间 (Duration)
        for k in new_indices:
            if new_indices[k] == self.current_state[k]:
                # 累加秒数 (MAX_DUR_SECONDS)
                self.current_state[f"{k}_dur"] = min(self.current_state[f"{k}_dur"] + elapsed, MAX_DUR_SECONDS)
            else:
                # 状态切换，入队历史
                self.hist_queues[k].append((self.current_state[k], self.current_state[f"{k}_dur"]))
                # 重置
                self.current_state[k] = new_indices[k]
                self.current_state[f"{k}_dur"] = 0

        # 5. 更新 Flags 相关的计数器 (Walk / Relax)
        curr_act_idx = new_indices["act"]

        # 判定 Walk/Run
        if curr_act_idx in self.walk_run_ids:
            self.walk_run_secs += elapsed
        else:
            self.walk_run_secs = 0.0  # 中断则归零

        # 判定 Stationary
        if curr_act_idx == self.idx_stationary:
            self.stationary_secs += elapsed
        else:
            self.stationary_secs = 0.0  # 中断则归零

        # 6. 构建 Observation
        obs = self._make_obs()

        # 7. 推理
        action_id = select_action_greedy(self.q_net, obs, self.device)
        action_name = ALL_ACTIONS[action_id]

        # 8. 调试信息
        try:
            str_act = ACTIVITIES[self.current_state['act']]
            str_loc = LOCATIONS[self.current_state['loc']]
            str_scene = SCENES[self.current_state['scene']]
            str_light = LIGHT_LEVELS[self.current_state['light']]
        except IndexError:
            str_act, str_loc, str_scene, str_light = "err", "err", "err", "err"

        debug_info = (
            f"State: Act={str_act}({int(self.current_state['act_dur'])}s) | "
            f"Loc={str_loc} | Light={str_light} | Scene={str_scene} | "
            f"Flags: Walk={int(self.walk_run_secs)}s, Relax={int(self.stationary_secs)}s"
        )

        return action_name, debug_info

    def _make_obs(self) -> np.ndarray:
        # Time encoding
        dt = datetime.now()
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        theta = 2.0 * np.pi * (seconds_since_midnight / 86400.0)
        time_sin, time_cos = float(np.sin(theta)), float(np.cos(theta))

        def _one_hot(idx, size):
            v = np.zeros((size,), dtype=np.float32)
            if 0 <= idx < size: v[idx] = 1.0
            return v

        def _encode_feature(key, n_classes, queue):
            # 当前状态
            current_pair = (self.current_state[key], self.current_state[f"{key}_dur"])
            # 历史回溯
            history_pairs = [current_pair] + list(reversed(list(queue)))[:self.history_len]

            encoded = []
            for idx, dur in history_pairs:
                encoded.extend(_one_hot(idx, n_classes).tolist())
                # 归一化 Duration
                encoded.append(float(dur) / float(MAX_DUR_SECONDS))
            return encoded

        feats = []
        # 1. Time (2)
        feats.extend([time_sin, time_cos])

        # 2. History Features
        feats.extend(_encode_feature("act", len(ACTIVITIES), self.hist_queues["act"]))
        feats.extend(_encode_feature("loc", len(LOCATIONS), self.hist_queues["loc"]))
        feats.extend(_encode_feature("light", len(LIGHT_LEVELS), self.hist_queues["light"]))
        feats.extend(_encode_feature("scene", len(SCENES), self.hist_queues["scene"]))

        # 3. Flags (2) - 新增
        # 阈值参考 aod_env_v3.py: walk>=60, relax>=600
        flag_walk = 1.0 if self.walk_run_secs >= 60 else 0.0
        flag_relax = 1.0 if self.stationary_secs >= 600 else 0.0

        feats.extend([flag_walk, flag_relax])

        return np.array(feats, dtype=np.float32)