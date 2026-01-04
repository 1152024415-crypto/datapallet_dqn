#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
import sys
import os
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'vlm_engine'))

from aod_env import (
    AODRecommendationEnv, ALL_ACTIONS, ACTIVITIES, LOCATIONS, SCENES, SOUND_LEVELS, LIGHT_LEVELS
)
from datapallet.datapallet import create_datapallet, DataPallet
from datapallet.enums import (
    ActivityMode, LightIntensity, SoundIntensity, LocationType, SceneType, SceneData, to_str
)
from datapallet.testbed import create_testbed, TestBed, PlaybackConfig
from vlm_engine import initialize_vlm_service, SceneAnalysisResult

SCENE_IMAGES_DIR = os.path.join(project_root, "datapallet/scene_images")
RECORDING_FILE = "scenario_meeting_walk.json"


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_n: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AsyncDQNAgent:
    def __init__(self, ckpt_path: Path, obs_dim: int, act_n: int, device: torch.device):
        self.device = device
        self.obs_queue = queue.Queue(maxsize=1)
        self.action_queue = queue.Queue(maxsize=1)
        self.running = True
        self.q_net = self._load_model(ckpt_path, obs_dim, act_n)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _load_model(self, ckpt_path: Path, obs_dim: int, act_n: int) -> QNet:
        q = QNet(obs_dim, act_n).to(self.device)
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                state_dict = ckpt["q_state_dict"] if isinstance(ckpt, dict) and "q_state_dict" in ckpt else ckpt
                q.load_state_dict(state_dict)
                print(f"[DQN] 模型加载成功: {ckpt_path}")
            except Exception as e:
                print(f"[DQN] 模型加载失败: {e}")
        else:
            print(f"[DQN] 未找到模型，使用随机权重。")
        q.eval()
        return q

    def _worker(self):
        while self.running:
            try:
                obs = self.obs_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if obs is None: break

            with torch.no_grad():
                t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                act = int(torch.argmax(self.q_net(t), dim=1).item())
            self.action_queue.put(act)

    def predict(self, obs: np.ndarray) -> int:
        if self.obs_queue.full(): self.obs_queue.get()
        self.obs_queue.put(obs)
        return self.action_queue.get()

    def stop(self):
        self.running = False
        self.obs_queue.put(None)
        self.worker_thread.join()


class ObservationBuilder:
    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim
        self.act_map = self._build_map(ActivityMode, ACTIVITIES)
        self.loc_map = self._build_map(LocationType, LOCATIONS)
        self.scene_map = self._build_map(SceneType, SCENES)

        self.sound_levels = len(SOUND_LEVELS)
        self.light_levels = len(LIGHT_LEVELS)

        self.last_act_id = 0
        self.act_duration = 0

    def _build_map(self, enum_cls, str_list):
        mapping = {}
        str_lower = [s.lower().replace(" ", "_") for s in str_list]
        for e in enum_cls:
            if e == 0: continue
            e_name = e.name.lower()
            match_idx = -1
            for i, s in enumerate(str_lower):
                if s in e_name or e_name in s:
                    match_idx = i
                    break
            if match_idx != -1:
                mapping[e] = match_idx
        return mapping

    def _one_hot(self, idx: int, size: int) -> np.ndarray:
        v = np.zeros((size,), dtype=np.float32)
        if 0 <= idx < size: v[idx] = 1.0
        return v

    def _time_feat(self) -> tuple[float, float]:
        now = datetime.now()
        minutes = now.hour * 60 + now.minute
        theta = 2 * np.pi * (minutes / 1440.0)
        return float(np.sin(theta)), float(np.cos(theta))

    def make_obs(self, dp: DataPallet) -> np.ndarray:
        def get_val(key, default):
            succ, v = dp.get(key)
            return v if succ and v is not None else default

        act_enum = get_val("activity_mode", ActivityMode.NULL)
        loc_enum = get_val("Location", LocationType.NULL)
        scene_obj = get_val("Scence", None)
        scene_enum = scene_obj.scene_type if isinstance(scene_obj, SceneData) else SceneType.NULL

        act_id = self.act_map.get(act_enum, 0)
        loc_id = self.loc_map.get(loc_enum, len(LOCATIONS))
        scene_id = self.scene_map.get(scene_enum, len(SCENES))

        sound_val = get_val("Sound_Intensity", SoundIntensity.NORMAL_SOUND)
        light_val = get_val("Light_Intensity", LightIntensity.MODERATE_BRIGHTNESS)
        sound_id = min(int(sound_val) - 1, self.sound_levels - 1) if int(sound_val) > 0 else 2
        light_id = min(int(light_val) - 1, self.light_levels - 1) if int(light_val) > 0 else 2

        if act_id == self.last_act_id:
            self.act_duration += 1
        else:
            self.act_duration = 0
            self.last_act_id = act_id

        vecs = []
        vecs.extend(self._time_feat())
        vecs.extend(self._one_hot(act_id, len(ACTIVITIES)))
        vecs.append(min(self.act_duration, 180) / 180.0)
        vecs.append(0.0)

        vecs.extend(self._one_hot(loc_id, len(LOCATIONS) + 1))
        vecs.extend(self._one_hot(scene_id, len(SCENES) + 1))
        vecs.extend(self._one_hot(sound_id, self.sound_levels + 1))
        vecs.extend(self._one_hot(light_id, self.light_levels + 1))

        vecs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(vecs, dtype=np.float32)


def run_testbed_scenario(
        dqn_agent: AsyncDQNAgent,
        vlm_module: Any,
        vlm_callback: Any,
        dp: DataPallet,
        scenario_desc: str,
        duration: float
):
    print(f"\n{'=' * 20} 启动测试床场景 {'=' * 20}")
    print(f"场景描述: {scenario_desc}")

    tb = create_testbed(dp)

    if not os.path.exists(RECORDING_FILE):
        print("[TestBed] 正在调用 LLM 生成仿真数据...")
        tb.record_data("meeting_walk", scenario_desc, duration=duration, interval=2.0)
        tb.save_recording(RECORDING_FILE)
    else:
        print(f"[TestBed] 加载本地录制文件: {RECORDING_FILE}")
        tb.load_recording(RECORDING_FILE)

    playback_config = PlaybackConfig(speed=1.0, loop=False, interval=1.0)
    tb.start_playback(playback_config)

    temp_env = AODRecommendationEnv()
    obs_builder = ObservationBuilder(temp_env.obs_dim)

    step_count = 0

    try:
        while tb.playing:
            start_time = time.time()
            step_count += 1

            obs = obs_builder.make_obs(dp)

            action_id = dqn_agent.predict(obs)
            action_name = ALL_ACTIONS[action_id]

            print(f"\n[Step {step_count}] Time: {datetime.now().strftime('%H:%M:%S')}")

            _, curr_act = dp.get("activity_mode")
            _, curr_scene = dp.get("Scence")
            scene_str = to_str("Scence", curr_scene)
            print(f"  感知状态: Activity={to_str('activity_mode', curr_act)} | Scene={scene_str[:30]}...")
            print(f"  DQN 决策: {action_name}")

            print("  -> 触发 VLM 分析...")
            success = vlm_callback(start_idx=0, batch_size=1)

            if success:
                wait_vlm = time.time()
                while vlm_module.is_busy():
                    time.sleep(0.1)
                    if time.time() - wait_vlm > 60:
                        print("  [Warn] VLM 超时")
                        break

                res = vlm_module.get_last_result()
                if res and res.scenes:
                    s = res.scenes[0]
                    print(f"  [VLM 结果] 活动: {s.main_activity} | 描述: {s.description[:50]}...")
                else:
                    print("  [VLM 结果] 无有效场景数据")
            else:
                print("  [Warn] VLM 忙，跳过")

            process_time = time.time() - start_time
            sleep_time = max(0, 2.0 - process_time)
            time.sleep(sleep_time)

            if not tb.playing:
                print("\n[TestBed] 数据回放结束。")
                break

    finally:
        tb.stop_playback()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="dqn_aod_ckpt.pt")
    parser.add_argument("--api_type", type=str, default="openai")
    default_desc = "在明亮的走廊里快走前往会议室，周围声音嘈杂。10秒后进入会议室，环境变安静，坐下开始开会，会议室光线明亮。"
    parser.add_argument("--scene_desc", type=str, default=default_desc)
    parser.add_argument("--duration", type=float, default=40.0, help="场景持续时间(秒)")
    args = parser.parse_args()

    print("--- 初始化服务 ---")
    dp = create_datapallet(ttl=10)

    default_img = os.path.join(SCENE_IMAGES_DIR, "其他.png")
    if not os.path.exists(default_img):
        print(f"[Err] 请确保 {default_img} 存在！")
        return

    dp.receive_data("Scence", SceneData(SceneType.OTHER, default_img), datetime.now())
    time.sleep(0.2)

    try:
        vlm_module, vlm_callback = initialize_vlm_service(api_type=args.api_type, datapallet=dp)
    except Exception as e:
        print(f"[Err] VLM 初始化失败: {e}")
        dp.stop()
        return

    temp_env = AODRecommendationEnv()
    dqn_agent = AsyncDQNAgent(
        Path(args.ckpt),
        temp_env.obs_dim,
        temp_env.action_n,
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        run_testbed_scenario(
            dqn_agent,
            vlm_module,
            vlm_callback,
            dp,
            args.scene_desc,
            args.duration
        )
    except KeyboardInterrupt:
        print("用户停止")
    finally:
        print("正在清理资源...")
        dqn_agent.stop()
        dp.stop()


if __name__ == "__main__":
    main()