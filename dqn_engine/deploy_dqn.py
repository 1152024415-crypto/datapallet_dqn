import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dqn_engine.aod_env_demo import AODDemoEnv
from dqn_engine.constants import (
    ALL_ACTIONS,
    ACTIVITIES,
    LOCATIONS,
    SCENES,
    LIGHT_LEVELS,
    TruthStep,
)
from dqn_engine.train_dqn_tensorboard_v5 import QNetwork as QNet


MAX_DUR_SECONDS = 180  # match aod_env_v5: min(dur, 180)/180
UNK_IDX = 0


def _fmt_time_from_seconds(seconds: int) -> str:
    total = int(seconds) % 86400
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def decode_obs(
    obs: np.ndarray,
    history_len: int,
    step_seconds: int,
    include_act_light_changed: int = 0,
) -> Dict[str, Any]:
    o = obs.astype(float)
    idx = 0
    time_sin, time_cos = o[idx], o[idx + 1]
    idx += 2
    act_light_changed = None

    def decode_onehot(vec: np.ndarray, labels: List[str], unknown_index: int = UNK_IDX) -> str:
        j = int(np.argmax(vec))
        if j == unknown_index:
            return "unknown"
        if 0 <= j < len(labels):
            return labels[j]
        return str(j)

    def _dur_norm_to_seconds(dur_norm: float) -> float:
        return float(dur_norm) * float(MAX_DUR_SECONDS)

    def _dur_norm_to_steps(dur_norm: float) -> float:
        denom = float(step_seconds) if step_seconds > 0 else 1.0
        return _dur_norm_to_seconds(dur_norm) / denom

    def _dur_norm_to_minutes(dur_norm: float) -> float:
        return _dur_norm_to_seconds(dur_norm) / 60.0

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
        walk_run_flag = int(round(float(o[idx]))) if idx < len(o) else 0
        idx += 1
        relax_flag = int(round(float(o[idx]))) if idx < len(o) else 0
        idx += 1

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

        walk_run_flag = int(round(float(o[idx]))) if idx < len(o) else 0
        idx += 1
        relax_flag = int(round(float(o[idx]))) if idx < len(o) else 0
        idx += 1
        act_light_changed = int(round(float(o[idx]))) if include_act_light_changed and idx < len(o) else None
        if include_act_light_changed and idx < len(o):
            idx += 1

        act_hist = []
        loc_hist = []
        light_hist = []
        scene_hist = []

    out = {
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
        "walk_run_flag": int(walk_run_flag),
        "relax_flag": int(relax_flag),
    }
    if include_act_light_changed:
        out["act_light_changed"] = act_light_changed
    return out


def obs_dim_aod_v5(include_act_light_changed: int = 0) -> int:
    """Observation dim for aod_env_v5 flat obs (no history)."""
    return (
        2
        + len(ACTIVITIES) + 1
        + len(LOCATIONS) + 1
        + len(LIGHT_LEVELS) + 1
        + len(SCENES) + 1
        + 2
        + (1 if include_act_light_changed else 0)
    )


def obs_dim_from_history(history_len: int) -> int:
    if history_len > 0:
        return (
            2
            + (history_len + 1) * (len(ACTIVITIES) + 1)
            + (history_len + 1) * (len(LOCATIONS) + 1)
            + (history_len + 1) * (len(LIGHT_LEVELS) + 1)
            + (history_len + 1) * (len(SCENES) + 1)
            + 2
        )
    return (
        2
        + len(ACTIVITIES) + 1
        + len(LOCATIONS) + 1
        + len(LIGHT_LEVELS) + 1
        + len(SCENES) + 1
        + 2
    )


def _get_truth_act_light_loc_scene(env: Any) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Match training: truth act/light + current observed loc/scene."""
    try:
        if hasattr(env, "_day") and hasattr(env, "_t"):
            t = int(env._t)
            if 0 <= t < len(env._day):
                truth = env._day[t]
                return (
                    int(getattr(truth, "act", -1)),
                    int(getattr(truth, "light", -1)),
                    int(getattr(env, "_loc_obs", -1)),
                    int(getattr(env, "_scene_obs", -1)),
                )
    except Exception:
        pass
    return None, None, None, None


def _get_walk_relax_flags(env: Any) -> Tuple[Optional[int], Optional[int]]:
    """Read walk/run and relax flags from env if available."""
    try:
        walk_run_secs = getattr(env, "_walk_run_secs", None)
        stationary_secs = getattr(env, "_stationary_secs", None)
        if walk_run_secs is None or stationary_secs is None:
            return None, None
        return int(walk_run_secs >= 60), int(stationary_secs >= 60)
    except Exception:
        return None, None


def _should_invoke(
    step_idx: int,
    interval_steps: int,
    last_act: Optional[int],
    last_light: Optional[int],
    last_loc: Optional[int],
    last_scene: Optional[int],
    last_walk_flag: Optional[int],
    last_relax_flag: Optional[int],
    cur_act: Optional[int],
    cur_light: Optional[int],
    cur_loc: Optional[int],
    cur_scene: Optional[int],
    cur_walk_flag: Optional[int],
    cur_relax_flag: Optional[int],
    invoke_mode: str,
) -> bool:
    if interval_steps <= 0:
        return True
    on_interval = (step_idx % interval_steps) == 0
    if (
        last_act is None
        or last_light is None
        or last_walk_flag is None
        or last_relax_flag is None
    ):
        return True
    changed = (
        (cur_act != last_act)
        or (cur_light != last_light)
        or (cur_loc != last_loc)
        or (cur_scene != last_scene)
        or (cur_walk_flag != last_walk_flag)
        or (cur_relax_flag != last_relax_flag)
    )
    mode = (invoke_mode or "both").lower()
    if mode == "interval":
        return on_interval
    if mode == "change":
        return changed
    return on_interval or changed


def load_checkpoint(ckpt_path: Path, obs_dim: int, act_n: int, device: torch.device) -> QNet:
    q = QNet(obs_dim, act_n).to(device)
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Using randomly initialized network.")
        return q

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
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
    stream_path: Path,
    step_seconds: int,
    bucket_seconds: int,
    history_len: int,
    invoke_interval_seconds: int,
    invoke_mode: str,
    include_act_light_changed: int,
    q: QNet,
    device: torch.device,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "deploy_actions.jsonl"
    txt_path = out_dir / "deploy_actions.txt"

    if stream_path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported stream_path (expected .json): {stream_path}")
    env = AODDemoEnv(
        meeting_path=str(stream_path),
        step_seconds=step_seconds,
        bucket_seconds=bucket_seconds,
        history_len=history_len,
        include_act_light_changed=include_act_light_changed,
        episode_steps=10**9,
    )
    
    obs, info = env.reset(seed=0)
    profile = info.get("profile", stream_path.stem)

    with jsonl_path.open("w", encoding="utf-8") as jf, txt_path.open("w", encoding="utf-8") as tf:
        tf.write(f"Meeting path: {stream_path}\n")
        tf.write(f"Step seconds: {step_seconds}\n")
        tf.write(f"Bucket seconds: {bucket_seconds}\n")
        tf.write(f"History len: {history_len}\n")
        tf.write(f"Profile: {profile}\n")
        tf.write("-" * 100 + "\n")

        t = 0
        done = False
        last_act: Optional[int] = None
        last_light: Optional[int] = None
        last_loc: Optional[int] = None
        last_scene: Optional[int] = None
        last_walk_flag: Optional[int] = None
        last_relax_flag: Optional[int] = None
        interval_steps = max(1, int(step_seconds and invoke_interval_seconds // step_seconds))
        while not done:
            truth: Optional[TruthStep] = None
            try:
                truth = env._day[t] if t < len(env._day) else None
            except Exception:
                truth = None

            cur_act, cur_light, cur_loc, cur_scene = _get_truth_act_light_loc_scene(env)
            cur_walk_flag, cur_relax_flag = _get_walk_relax_flags(env)
            invoke = _should_invoke(
                step_idx=t,
                interval_steps=interval_steps,
                last_act=last_act,
                last_light=last_light,
                last_loc=last_loc,
                last_scene=last_scene,
                last_walk_flag=last_walk_flag,
                last_relax_flag=last_relax_flag,
                cur_act=cur_act,
                cur_light=cur_light,
                cur_loc=cur_loc,
                cur_scene=cur_scene,
                cur_walk_flag=cur_walk_flag,
                cur_relax_flag=cur_relax_flag,
                invoke_mode=invoke_mode,
            )
            last_act, last_light, last_loc, last_scene = cur_act, cur_light, cur_loc, cur_scene
            last_walk_flag, last_relax_flag = cur_walk_flag, cur_relax_flag

            if invoke:
                action_id = select_action_greedy(q, obs, device)
            else:
                action_id = ALL_ACTIONS.index("NONE")
            action_name = ALL_ACTIONS[action_id]

            obs2, reward, terminated, truncated, info2 = env.step(action_id, invoke=invoke)
            done = bool(terminated or truncated)

            time_str = getattr(truth, "time_str", _fmt_time_from_seconds(int(getattr(truth, "t", 0)))) if truth else "unknown"
            rec = {
                "t": t,
                "time": time_str,
                "invoke": bool(invoke),
                "action_id": action_id,
                "action": action_name,
                "oracle": info2.get("oracle", "NONE"),
                "reward": float(reward),
                "obs": decode_obs(obs, history_len=history_len, step_seconds=step_seconds, include_act_light_changed=include_act_light_changed),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tf.write(
                f"{rec['time']} step={rec['t']:4d} | invoke={int(invoke)} "
                f"| a={action_name:<22s} oracle={rec['oracle']:<14s} r={rec['reward']:>7.2f}\n"
            )

            obs = obs2
            t += 1

    logger.info("Replay complete: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="logs_v5/dqn_aod_ckpt_episode_2200.pt")
    parser.add_argument("--stream_path", type=str, default="demo/meeting.json")
    parser.add_argument("--step_seconds", type=int, default=1)
    parser.add_argument("--bucket_seconds", type=int, default=1, help="Bucket size for event grouping (should match step_seconds)")
    parser.add_argument("--history_len", type=int, default=0)
    parser.add_argument("--invoke_interval_seconds", type=int, default=1)
    parser.add_argument(
        "--invoke_mode",
        type=str,
        choices=["interval", "change", "both"],
        default="both",
        help="Invoke policy: interval | change | both",
    )
    parser.add_argument("--out_dir", type=str, default="test_runs_meeting")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--include_act_light_changed", type=int, default=1, help="Match aod_env_v5 obs (1=include act/light-changed flag)")

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

    obs_dim = obs_dim_aod_v5(include_act_light_changed=args.include_act_light_changed)
    q = load_checkpoint(Path(args.ckpt), obs_dim, len(ALL_ACTIONS), torch.device(device))

    stream_path = Path(args.stream_path)

    deploy_replay_env(
        stream_path=stream_path,
        step_seconds=args.step_seconds,
        bucket_seconds=args.bucket_seconds,
        history_len=args.history_len,
        invoke_interval_seconds=args.invoke_interval_seconds,
        invoke_mode=args.invoke_mode,
        include_act_light_changed=args.include_act_light_changed,
        q=q,
        device=torch.device(device),
        out_dir=Path(args.out_dir),
        logger=logger,
    )


if __name__ == "__main__":
    main()
