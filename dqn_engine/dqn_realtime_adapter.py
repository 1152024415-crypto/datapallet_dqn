import torch
import numpy as np
import time
from datetime import datetime
from typing import Any, Tuple, Optional
from pathlib import Path
from collections import deque
from typing import Dict, Any, Tuple, List, Optional

from dqn_engine.constants import (
    ACTIVITIES, LOCATIONS, SCENES, SOUND_LEVELS, LIGHT_LEVELS, ALL_ACTIONS
)
from dqn_engine.deploy_dqn import load_checkpoint, select_action_greedy

from datapallet.enums import (
    ActivityMode, LocationType, SceneType, LightIntensity
)
from datapallet.datapallet import DataPallet

# 与 aod_env_v5.py 和 deploy_dqn.py 保持一致
# min(truth.act_dur, 180) / 180.0
MAX_DUR_SECONDS = 180.0


class DQNEngineAdapter:
    def __init__(
        self,
        ckpt_path: str,
        history_len: int = 0,
        device: str = "cpu",
        include_act_light_changed: int = 1
    ):
        """
        Args:
            ckpt_path: 模型权重路径
            history_len: 历史长度 (main.py 中目前为 0)
            device: 运算设备
            include_act_light_changed: 是否包含[姿态/光照变化]特征位，需与训练配置一致 (默认1)
        """
        self.device = torch.device(device)
        self.history_len = history_len
        self.include_act_light_changed = include_act_light_changed

        # 计算 Observation Dimension
        # 参考 deploy_dqn.py -> obs_dim_aod_v5
        # 结构: Time(2) + Act(N+1) + Loc(N+1) + Light(N+1) + Scene(N+1) + Flags(2) + Changed(0/1)
        self.obs_dim = (
            2
            + len(ACTIVITIES) + 1
            + len(LOCATIONS) + 1
            + len(LIGHT_LEVELS) + 1
            + len(SCENES) + 1
            + 2
            + (1 if self.include_act_light_changed else 0)
        )

        print(f"[DQN Adapter] Loading model from {ckpt_path}")
        print(f"[DQN Adapter] Config: obs_dim={self.obs_dim}, device={device}, changed_flag={include_act_light_changed}")

        self.q_net = load_checkpoint(
            Path(ckpt_path),
            self.obs_dim,
            len(ALL_ACTIONS),
            self.device
        )

        # 状态追踪
        self.current_state = {
            "act": 0, "act_dur": 0.0,
            "loc": 0, "loc_dur": 0.0,
            "scene": 0, "scene_dur": 0.0,
            "light": 0, "light_dur": 0.0
        }

        # 辅助变量
        self.walk_run_secs = 0.0
        self.stationary_secs = 0.0
        self.last_update_time = time.time()

        # 记录上一次有效的原始值
        self.last_valid_raw_act: Any = None
        self.last_valid_raw_light: Any = None

        # 记录上一次有效值的时间戳 (用于计算是否过期)
        self.last_valid_act_time: float = 0.0
        self.last_valid_light_time: float = 0.0

        self.last_act_idx: Optional[int] = None
        self.last_light_idx: Optional[int] = None

        # 历史队列 (即使 history_len=0 也要初始化防止报错)
        self.hist_queues = {
            "act": deque(maxlen=max(1, history_len + 1)),
            "loc": deque(maxlen=max(1, history_len + 1)),
            "scene": deque(maxlen=max(1, history_len + 1)),
            "light": deque(maxlen=max(1, history_len + 1))
        }

        for k in self.hist_queues:
            for _ in range(self.hist_queues[k].maxlen):
                self.hist_queues[k].append((0, 0))

        # TODO 当前比数据托盘多10s
        self.PERSISTENCE_TIMEOUT = 70.0

        self._init_mappings()
        self._init_special_indices()

    def _safe_index(self, lst: list, val: str) -> int:
        try:
            return lst.index(val)
        except ValueError:
            return 0

    def _init_special_indices(self):
        """缓存特定索引，用于 Flags 计算逻辑"""
        # Activity Indices
        self.idx_act_slow_walk = self._safe_index(ACTIVITIES, "slow walk")
        self.idx_act_fast_walk = self._safe_index(ACTIVITIES, "fast walk")
        self.idx_act_fast_run = self._safe_index(ACTIVITIES, "fast run")
        self.idx_act_stationary = self._safe_index(ACTIVITIES, "stationary")

        self.walk_run_ids = {self.idx_act_slow_walk, self.idx_act_fast_walk, self.idx_act_fast_run}

        # Location Indices
        self.idx_loc_work = self._safe_index(LOCATIONS, "Work")
        self.idx_loc_subway = self._safe_index(LOCATIONS, "Subway Station")

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
            ActivityMode.SUBWAY: self._safe_index(ACTIVITIES, "stationary"),
            ActivityMode.RIDING: self._safe_index(ACTIVITIES, "stationary"),
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
            # 补全之前发现的遗漏
            LocationType.Research_Institution: self._safe_index(LOCATIONS, "Work"),
        }

        # 3. Scene 映射 (对应 constants.SCENES)
        self.scene_map = {
            SceneType.NULL: 0,
            SceneType.OTHER: self._safe_index(SCENES, "other"),
            SceneType.MEETINGROOM: self._safe_index(SCENES, "conference_room"),
            # 确保 constants.py 里有 "office" 或 "office_cubicles"，这里根据文件内容映射
            SceneType.WORKSPACE: self._safe_index(SCENES, "office"),
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
        """
        执行实时推理
        1. Pre-process: 如果当前数据 Unknown，尝试使用"最近一次有效值"进行补全
        2. 只有在补全后依然无效(超时)，才跳过推理。
        """
        # 1. 计算时间差
        now = time.time()
        elapsed = now - self.last_update_time
        if elapsed > 60: elapsed = 1.0

        # 2. 获取数据
        success_act, raw_act = dp.get("activity_mode", only_valid=True)
        success_loc, raw_loc = dp.get("Location", only_valid=True)
        success_scene, raw_scene = dp.get("Scene", only_valid=True)
        success_light, raw_light = dp.get("Light_Intensity", only_valid=True)

        # 状态保持与补全逻辑 (Activity)
        if success_act:
            # 当前数据有效，更新缓存
            self.last_valid_raw_act = raw_act
            self.last_valid_act_time = now
        else:
            # 当前数据无效，检查缓存是否可用
            if self.last_valid_raw_act is not None and (now - self.last_valid_act_time < self.PERSISTENCE_TIMEOUT):
                raw_act = self.last_valid_raw_act
                success_act = True  # 标记为补全成功
                print(f"[DQN] Activity 补全: 使用旧值 {raw_act}")

        # 状态保持与补全逻辑 (Light)
        if success_light:
            self.last_valid_raw_light = raw_light
            self.last_valid_light_time = now
        else:
            if self.last_valid_raw_light is not None and (now - self.last_valid_light_time < self.PERSISTENCE_TIMEOUT):
                raw_light = self.last_valid_raw_light
                success_light = True  # 标记为补全成功
                print(f"[DQN] Light 补全: 使用旧值 {raw_light}")

        # 如果补全失败，则必须停止
        # 仍然需要门控，因为如果系统刚启动或者断连太久，确实无法推理
        if not success_act or not success_light:
            self.last_update_time = now
            print("[Skipped] Data Unknown & Persistence Expired")
            return "NONE", "[Skipped] Data Unknown & Persistence Expired"

        self.last_update_time = now

        # 处理 Loc 和 Scene
        if not success_loc: raw_loc = LocationType.NULL
        if not success_scene: raw_scene = SceneType.NULL

        if isinstance(raw_act, tuple): raw_act = raw_act[1]

        raw_scene_enum = SceneType.NULL
        if hasattr(raw_scene, 'scene_type'):
            raw_scene_enum = raw_scene.scene_type
        elif isinstance(raw_scene, SceneType):
            raw_scene_enum = raw_scene
        elif isinstance(raw_scene, int):
            raw_scene_enum = raw_scene

        # 3. 映射
        new_indices = {
            "act": self._map_val("act", raw_act),
            "loc": self._map_val("loc", raw_loc),
            "scene": self._map_val("scene", raw_scene_enum),
            "light": self._map_val("light", raw_light)
        }

        # 4. Changed Flag 计算
        act_light_changed = 0.0
        if self.include_act_light_changed:
            if self.last_act_idx is None: self.last_act_idx = new_indices["act"]
            if self.last_light_idx is None: self.last_light_idx = new_indices["light"]

            if (new_indices["act"] != self.last_act_idx) or \
                    (new_indices["light"] != self.last_light_idx):
                act_light_changed = 1.0

            self.last_act_idx = new_indices["act"]
            self.last_light_idx = new_indices["light"]

        # 5. 更新 Duration
        for k in new_indices:
            if new_indices[k] == self.current_state[k]:
                self.current_state[f"{k}_dur"] = min(self.current_state[f"{k}_dur"] + elapsed, MAX_DUR_SECONDS)
            else:
                self.hist_queues[k].append((self.current_state[k], self.current_state[f"{k}_dur"]))
                self.current_state[k] = new_indices[k]
                self.current_state[f"{k}_dur"] = 0.0

        # 6. Flags 计算
        curr_act_idx = new_indices["act"]
        if curr_act_idx in self.walk_run_ids:
            self.walk_run_secs += elapsed
        else:
            self.walk_run_secs = 0.0

        if curr_act_idx == self.idx_act_stationary:
            self.stationary_secs += elapsed
        else:
            self.stationary_secs = 0.0

        # 7. 构建 & 推理
        obs = self._make_obs(act_light_changed)
        action_id = select_action_greedy(self.q_net, obs, self.device)
        action_name = ALL_ACTIONS[action_id]

        # 8. Debug Info
        try:
            str_act = ACTIVITIES[self.current_state['act']]
            str_loc = LOCATIONS[self.current_state['loc']]
            str_scene = SCENES[self.current_state['scene']]
            str_light = LIGHT_LEVELS[self.current_state['light']]
        except IndexError:
            str_act, str_loc, str_scene, str_light = "err", "err", "err", "err"

        # 在 Debug 信息中标记是否使用了补全值
        is_patched = "(Patched)" if (
                    not dp.get("activity_mode", only_valid=True)[0] or not dp.get("Light_Intensity", only_valid=True)[
                0]) else ""

        debug_info = (
            f"State{is_patched}: Act={str_act}({int(self.current_state['act_dur'])}s) | "
            f"Loc={str_loc} | Light={str_light} | "
            f"Flags: Walk={int(self.walk_run_secs)}s, Relax={int(self.stationary_secs)}s"
        )

        return action_name, debug_info

    def _one_hot(self, idx: int, size: int) -> np.ndarray:
        v = np.zeros((size,), dtype=np.float32)
        if 0 <= idx < size:
            v[idx] = 1.0
        return v

    def _make_obs(self, act_light_changed_val: float) -> np.ndarray:
        """
        构建 Observation 向量，逻辑严格对齐 aod_env_v5.py 的 _make_obs
        """
        # 1. Time encoding (Time of Day)
        dt = datetime.now()
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        theta = 2.0 * np.pi * (seconds_since_midnight / 86400.0)
        time_sin, time_cos = float(np.sin(theta)), float(np.cos(theta))

        feats = [time_sin, time_cos]

        # 2. Activity (OneHot + Dur)
        act_idx = self.current_state["act"]
        act_dur_norm = float(self.current_state["act_dur"]) / MAX_DUR_SECONDS
        feats.extend(self._one_hot(act_idx, len(ACTIVITIES)).tolist())
        feats.append(act_dur_norm)

        # 3. Location (OneHot + Age/Dur)
        # Location 被我写死了====
        loc_idx = self.current_state["loc"]
        loc_dur_norm = float(self.current_state["loc_dur"]) / MAX_DUR_SECONDS
        feats.extend(self._one_hot(loc_idx, len(LOCATIONS)).tolist())
        feats.append(loc_dur_norm)

        # 4. Light (OneHot + Dur)
        light_idx = self.current_state["light"]
        light_dur_norm = float(self.current_state["light_dur"]) / MAX_DUR_SECONDS
        feats.extend(self._one_hot(light_idx, len(LIGHT_LEVELS)).tolist())
        feats.append(light_dur_norm)

        # 5. Scene (OneHot + Age/Dur)
        scene_idx = self.current_state["scene"]
        scene_dur_norm = float(self.current_state["scene_dur"]) / MAX_DUR_SECONDS
        feats.extend(self._one_hot(scene_idx, len(SCENES)).tolist())
        feats.append(scene_dur_norm)

        # 6. Flags
        # 逻辑参考 aod_env_v5.py line 440-441
        # walk_run_flag: >= 60s AND loc not in [Work, Subway]
        if self.walk_run_secs >= 60.0 and \
           self.current_state["loc"] not in [self.idx_loc_work, self.idx_loc_subway]:
            flag_walk = 1.0
        else:
            flag_walk = 0.0

        # relax_flag: >= 60s AND loc == Work (这里 aod_env_v5.py 写的是 60s, 之前版本是 600s)
        if self.stationary_secs >= 60.0 and \
           self.current_state["loc"] == self.idx_loc_work:
            flag_relax = 1.0
        else:
            flag_relax = 0.0

        feats.extend([flag_walk, flag_relax])

        # 7. Changed Flag (Optional)
        if self.include_act_light_changed:
            feats.append(act_light_changed_val)

        return np.array(feats, dtype=np.float32)