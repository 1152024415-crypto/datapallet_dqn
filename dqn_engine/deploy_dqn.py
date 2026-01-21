import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

from dqn_engine.aod_env_demo import AODDemoEnv
from dqn_engine.constants import (
    ALL_ACTIONS,
    ACTIVITIES,
    LOCATIONS,
    SCENES,
    SOUND_LEVELS,
    LIGHT_LEVELS,
    PROBE_ACTIONS,
    TruthStep,
)
from dqn_engine.train_dqn_tensorboard_v3 import QNetwork as QNet

DEFAULT_STEP_SECONDS = 10
MAX_DUR_STEPS = 180 * 6  # must match meeting env _make_obs max_dur
UNK_IDX = 0


def _fmt_time_from_seconds(seconds: int) -> str:
    total = int(seconds) % 86400
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def decode_obs(obs: np.ndarray, history_len: int, step_seconds: int) -> Dict[str, Any]:
    o = obs.astype(float)
    idx = 0
    time_sin, time_cos = o[idx], o[idx + 1]
    idx += 2

    def decode_onehot(vec: np.ndarray, labels: List[str], unknown_index: int = UNK_IDX) -> str:
        j = int(np.argmax(vec))
        if j == unknown_index:
            return "unknown"
        if 0 <= j < len(labels):
            return labels[j]
        return str(j)

    def _dur_norm_to_steps(dur_norm: float) -> float:
        return float(dur_norm) * float(MAX_DUR_STEPS)

    def _dur_norm_to_minutes(dur_norm: float) -> float:
        return _dur_norm_to_steps(dur_norm) * step_seconds / 60.0

    def _decode_history(n: int, labels: List[str], unknown_index: int = UNK_IDX) -> List[Dict[str, Any]]:
        nonlocal idx
        entries: List[Dict[str, Any]] = []
        for _ in range(history_len + 1):
            vec = o[idx: idx + n]
            idx += n
            dur_norm = float(o[idx])
            idx += 1
            label = decode_onehot(vec, labels, unknown_index)
            dur_min = _dur_norm_to_minutes(dur_norm)
            entries.append(
                {
                    "label": label,
                    "dur_norm": round(float(dur_norm), 6),
                    "dur_steps": round(_dur_norm_to_steps(dur_norm), 3),
                    "dur_min": round(dur_min, 4),
                }
            )
        return entries

    if history_len > 0:
        act_hist = _decode_history(len(ACTIVITIES), ACTIVITIES, UNK_IDX)
        loc_hist = _decode_history(len(LOCATIONS), LOCATIONS, UNK_IDX)
        light_hist = _decode_history(len(LIGHT_LEVELS), LIGHT_LEVELS, UNK_IDX)
        scene_hist = _decode_history(len(SCENES), SCENES, UNK_IDX)

        act_label = act_hist[0]["label"]
        act_dur_min = act_hist[0]["dur_min"]
        loc = loc_hist[0]["label"]
        light = light_hist[0]["label"]
        scene = scene_hist[0]["label"]
        loc_dur_min = loc_hist[0]["dur_min"]
        light_dur_min = light_hist[0]["dur_min"]
        scene_dur_min = scene_hist[0]["dur_min"]
    else:
        act_vec = o[idx: idx + len(ACTIVITIES)]
        act_i = int(np.argmax(act_vec))
        act_label = ACTIVITIES[act_i] if 0 <= act_i < len(ACTIVITIES) else str(act_i)
        idx += len(ACTIVITIES)
        act_dur_norm = float(o[idx])
        idx += 1
        act_dur_min = _dur_norm_to_minutes(act_dur_norm)

        loc_vec = o[idx: idx + len(LOCATIONS)]
        idx += len(LOCATIONS)
        loc_dur_norm = float(o[idx])
        idx += 1
        light_vec = o[idx: idx + len(LIGHT_LEVELS)]
        idx += len(LIGHT_LEVELS)
        light_dur_norm = float(o[idx])
        idx += 1
        scene_vec = o[idx: idx + len(SCENES)]
        idx += len(SCENES)
        scene_dur_norm = float(o[idx])
        idx += 1

        loc = decode_onehot(loc_vec, LOCATIONS, UNK_IDX)
        light = decode_onehot(light_vec, LIGHT_LEVELS, UNK_IDX)
        scene = decode_onehot(scene_vec, SCENES, UNK_IDX)

        loc_dur_min = _dur_norm_to_minutes(loc_dur_norm)
        light_dur_min = _dur_norm_to_minutes(light_dur_norm)
        scene_dur_min = _dur_norm_to_minutes(scene_dur_norm)

        act_hist = []
        loc_hist = []
        light_hist = []
        scene_hist = []

    return {
        "time_sin": time_sin,
        "time_cos": time_cos,
        "activity": act_label,
        "activity_dur_min": round(float(act_dur_min), 4),
        "loc_obs": loc,
        "scene_obs": scene,
        "light_obs": light,
        "loc_dur_min": round(float(loc_dur_min), 4),
        "scene_dur_min": round(float(scene_dur_min), 4),
        "light_dur_min": round(float(light_dur_min), 4),
        "activity_history": act_hist,
        "loc_history": loc_hist,
        "light_history": light_hist,
        "scene_history": scene_hist,
        "step_seconds": step_seconds,
    }


def obs_dim_from_history(history_len: int) -> int:
    if history_len > 0:
        return (
                2
                + (history_len + 1) * (len(ACTIVITIES) + 1)
                + (history_len + 1) * (len(LOCATIONS) + 1)
                + (history_len + 1) * (len(LIGHT_LEVELS) + 1)
                + (history_len + 1) * (len(SCENES) + 1)
        )
    return (
            2
            + len(ACTIVITIES) + 1
            + len(LOCATIONS) + 1
            + len(LIGHT_LEVELS) + 1
            + len(SCENES) + 1
    )


def load_checkpoint(ckpt_path: Path, obs_dim: int, act_n: int, device: torch.device) -> QNet:
    q = QNet(obs_dim, act_n).to(device)
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Using randomly initialized network.")
        return q

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "q_state_dict" in ckpt:
        q.load_state_dict(ckpt["q_state_dict"])
        if "actions" in ckpt and ckpt["actions"] != ALL_ACTIONS:
            print("[WARN] Action list in checkpoint differs from current ALL_ACTIONS.")
            print("       If you changed actions between training and testing, results will be invalid.")
        print(f"[OK] Loaded checkpoint: {ckpt_path}")
    else:
        try:
            q.load_state_dict(ckpt)
            print(f"[OK] Loaded raw state_dict from: {ckpt_path}")
        except Exception as e:
            raise RuntimeError(f"Unrecognized checkpoint format in {ckpt_path}: {e}") from e
    q.eval()
    return q


@torch.no_grad()
def select_action_greedy(q: QNet, obs: np.ndarray, device: torch.device) -> int:
    s = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    return int(torch.argmax(q(s), dim=1).item())


def deploy_replay_env(
        meeting_path: Path,
        step_seconds: int,
        bucket_seconds: int,
        history_len: int,
        q: QNet,
        device: torch.device,
        out_dir: Path,
        logger: logging.Logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "deploy_actions.jsonl"
    txt_path = out_dir / "deploy_actions.txt"

    env = AODDemoEnv(
        meeting_path=str(meeting_path),
        step_seconds=step_seconds,
        bucket_seconds=bucket_seconds,
        history_len=history_len,
        episode_steps=10 ** 9,
    )

    obs, info = env.reset(seed=0)
    profile = info.get("profile", meeting_path.stem)

    with jsonl_path.open("w", encoding="utf-8") as jf, txt_path.open("w", encoding="utf-8") as tf:
        tf.write(f"Meeting path: {meeting_path}\n")
        tf.write(f"Bucket seconds: {bucket_seconds}\n")
        tf.write(f"Step seconds: {step_seconds}\n")
        tf.write(f"History len: {history_len}\n")
        tf.write(f"Profile: {profile}\n")
        tf.write("-" * 100 + "\n")

        t = 0
        done = False
        while not done:
            truth: Optional[TruthStep] = None
            try:
                truth = env._day[t] if t < len(env._day) else None
            except Exception:
                truth = None

            action_id = select_action_greedy(q, obs, device)
            action_name = ALL_ACTIONS[action_id]

            obs2, reward, terminated, truncated, info2 = env.step(action_id)
            done = bool(terminated or truncated)

            time_str = getattr(truth, "time_str",
                               _fmt_time_from_seconds(int(getattr(truth, "t", 0)))) if truth else "unknown"
            rec = {
                "t": t,
                "time": time_str,
                "action_id": action_id,
                "action": action_name,
                "reward": float(reward),
                "obs": decode_obs(obs, history_len=history_len, step_seconds=step_seconds),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tf.write(
                f"{rec['time']} step={rec['t']:4d} | a={action_name:<22s} r={rec['reward']:>7.2f}\n"
            )

            obs = obs2
            t += 1

    logger.info("Replay complete: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/Users/cannkit/Downloads/datapallet/dqn_engine/dqn_aod_ckpt_episode_1100.pt")
    parser.add_argument("--stream_path", type=str, default="/Users/cannkit/Downloads/datapallet/dqn_engine/meeting.json")
    parser.add_argument("--step_seconds", type=int, default=1)
    parser.add_argument("--bucket_seconds", type=int, default=1)
    parser.add_argument("--history_len", type=int, default=0)
    parser.add_argument("--poll_seconds", type=float, default=0.5)
    parser.add_argument("--idle_timeout", type=float, default=5.0)
    parser.add_argument("--start_at_end", action="store_true")
    parser.add_argument("--out_dir", type=str, default="test_runs_meeting")
    parser.add_argument("--cuda", type=int, default=0)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(str(Path(args.out_dir) / "dqn_deploy.log"), mode="w"))

    if args.cuda >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    obs_dim = obs_dim_from_history(args.history_len)
    q = load_checkpoint(Path(args.ckpt), obs_dim, len(ALL_ACTIONS), torch.device(device))

    stream_path = Path(args.stream_path)

    deploy_replay_env(
        meeting_path=stream_path,
        step_seconds=args.step_seconds,
        bucket_seconds=args.bucket_seconds,
        history_len=args.history_len,
        q=q,
        device=torch.device(device),
        out_dir=Path(args.out_dir),
        logger=logger,
    )


if __name__ == "__main__":
    main()
