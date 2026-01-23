"""
Deep Q-Network (DQN) Training Script for AOD Recommendation Environment

This script trains a DQN agent to learn when to probe sensors and when to make
recommendations in the Always-On-Display (AOD) recommendation environment.

Features:
- TensorBoard logging for training metrics
- Random baseline comparisons during training and evaluation
- Configurable hyperparameters via command-line arguments
- Checkpoint saving for model persistence

Usage:
    python train_dqn_tensorboard_v2.py [--seed 0] [--train_episodes 5000] ...

TensorBoard:
    tensorboard --logdir logs
"""

from __future__ import annotations

# Standard library imports
import argparse
import logging
from logging import Logger
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Tuple, Optional

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Local imports
from dqn_engine.aod_env_v3 import (
    AODRecommendationEnv,
    ALL_ACTIONS,
    PROBE_ACTIONS,
    RECOMMEND_ACTIONS,
    TRAIN_EPISODES,
    EVAL_EPISODES,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DQNConfig:
    """Configuration for DQN training."""
    
    # Random seed
    seed: int = 0
    
    # Device settings
    device: str = "cuda"
    cuda: int = 0
    
    # Learning hyperparameters
    gamma: float = 0.99  # Discount factor
    lr: float = 5e-4  # Learning rate
    batch_size: int = 64
    replay_size: int = 36_0000
    warmup_steps: int = 3_6000  # Steps before training starts
    train_every: int = 4  # Train every N steps
    target_update_every: int = 5_000  # Update target network every N steps
    grad_clip: float = 10.0  # Gradient clipping threshold
    history_len: int = 0
    loc_always_available: bool = False

    # Invocation control (data step = 10s)
    step_seconds: int = 10
    invoke_interval_seconds: int = 60  # e.g., 60s or 30s
    invoke_mode: str = "both"  # interval | change | both
    
    # Exploration hyperparameters
    eps_start: float = 1.0  # Initial epsilon (exploration rate)
    eps_end: float = 0.05  # Final epsilon (minimum exploration)
    eps_decay_steps: int = 3600*400  # Steps to decay epsilon
    
    # Training episodes
    train_episodes: int = 5000
    train_episode_steps: int = 36000  # One trajectory per episode
    
    # Evaluation settings
    eval_every: int = 100  # Evaluate every N episodes
    eval_episodes: int = 10  # Number of episodes per evaluation
    eval_episode_steps: int = 36000  # One trajectory per evaluation
    
    # Logging settings
    random_train_every: int = 10  # Log random baseline every N episodes (0 to disable)
    logdir: str = "logs"
    run_name: str = "run"
    log_file: str = "train_dqn_tensorboard_v2_balancedbf.log"
    debug_invoke_episodes: int = 0  # If >0, run invoke debug and exit


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_epsilon(
    eps_start: float, 
    eps_end: float, 
    eps_decay_steps: int, 
    step: int
) -> float:
    """Compute epsilon-greedy exploration rate for current step."""
    decay_fraction = min(1.0, step / float(eps_decay_steps))
    return eps_start + decay_fraction * (eps_end - eps_start)


def _get_truth_act_light(env: AODRecommendationEnv) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Read current truth act/light if available; otherwise return (None, None)."""
    try:
        if hasattr(env, "_day") and hasattr(env, "_t"):
            t = int(env._t)
            if 0 <= t < len(env._day):
                truth = env._day[t]
                return int(getattr(truth, "act", -1)), int(getattr(truth, "light", -1)), int(env._loc_obs), int(env._scene_obs)
    except Exception:
        pass
    return None, None, None, None


def _get_walk_relax_flags(env: AODRecommendationEnv) -> Tuple[Optional[int], Optional[int]]:
    """Read walk/run and relax flags from env if available."""
    try:
        walk_run_secs = getattr(env, "_walk_run_secs", None)
        stationary_secs = getattr(env, "_stationary_secs", None)
        if walk_run_secs is None or stationary_secs is None:
            return None, None
        return int(walk_run_secs >= 60), int(stationary_secs >= 600)
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


def debug_invoke_logic(config: DQNConfig, logger: Logger) -> None:
    """Run random episodes and log invoke decision details."""
    env = AODRecommendationEnv(
        seed=config.seed + 123,
        history_len=config.history_len,
        loc_always_available=config.loc_always_available,
    )
    interval_steps = max(1, int(config.step_seconds and config.invoke_interval_seconds // config.step_seconds))
    total_steps = 0
    invoke_steps = 0
    change_steps = 0
    interval_steps_hit = 0
    invoke_steps_interval = 0
    invoke_steps_change = 0
    invoke_steps_both = 0

    logger.info("=== Invoke Debug ===")
    logger.info(
        f"episodes={config.debug_invoke_episodes} interval_steps={interval_steps} "
        f"invoke_mode={config.invoke_mode}"
    )

    for episode in range(1, config.debug_invoke_episodes + 1):
        env.episode_steps = config.eval_episode_steps
        day_id = np.random.randint(1, TRAIN_EPISODES)
        observation, _ = env.reset(seed=config.seed + episode * 31, day_id=day_id, logger=logger)
        done = False
        last_act: Optional[int] = None
        last_light: Optional[int] = None
        last_loc: Optional[int] = None
        last_scene: Optional[int] = None
        last_walk_flag: Optional[int] = None
        last_relax_flag: Optional[int] = None

        while not done:
            cur_act, cur_light, cur_loc, cur_scene = _get_truth_act_light(env)
            cur_walk_flag, cur_relax_flag = _get_walk_relax_flags(env)
            truth_loc = None
            truth_scene = None
            truth_act_dur = None
            truth_loc_dur = None
            truth_scene_dur = None
            truth_light_dur = None
            try:
                t = int(env._t)
                if 0 <= t < len(env._day):
                    truth = env._day[t]
                    truth_loc = int(getattr(truth, "loc", -1))
                    truth_scene = int(getattr(truth, "scene", -1))
                    truth_act_dur = int(getattr(truth, "act_dur", -1))
                    truth_loc_dur = int(getattr(truth, "loc_dur", -1))
                    truth_scene_dur = int(getattr(truth, "scene_dur", -1))
                    truth_light_dur = int(getattr(truth, "light_dur", -1))
            except Exception:
                pass
            on_interval = (int(env._t) % interval_steps) == 0
            changed = (
                last_act is None
                or last_light is None
                or last_walk_flag is None
                or last_relax_flag is None
                or (cur_act != last_act)
                or (cur_light != last_light)
                or (cur_loc != last_loc)
                or (cur_scene != last_scene)
                or (cur_walk_flag != last_walk_flag)
                or (cur_relax_flag != last_relax_flag)
            )
            invoke = _should_invoke(
                step_idx=int(env._t),
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
                invoke_mode=config.invoke_mode,
            )

            walk_run_secs = int(getattr(env, "_walk_run_secs", -1))
            stationary_secs = int(getattr(env, "_stationary_secs", -1))
            walk_run_flag = int(walk_run_secs >= 60) if walk_run_secs >= 0 else -1
            relax_flag = int(stationary_secs >= 600) if stationary_secs >= 0 else -1

            logger.info(
                "ep=%03d t=%05d invoke=%s interval=%s changed=%s "
                "last(act=%s light=%s loc=%s scene=%s) "
                "cur(act=%s light=%s loc=%s scene=%s) "
                "obs(loc=%s scene=%s) truth(loc=%s scene=%s) "
                "truth_dur(act=%s light=%s loc=%s scene=%s) obs_age(loc=%s scene=%s light=%s) "
                "walk_run_secs=%s stationary_secs=%s flags(walk=%s relax=%s)",
                episode,
                int(env._t),
                int(invoke),
                int(on_interval),
                int(changed),
                last_act,
                last_light,
                last_loc,
                last_scene,
                cur_act,
                cur_light,
                cur_loc,
                cur_scene,
                env._loc_obs,
                env._scene_obs,
                truth_loc,
                truth_scene,
                truth_act_dur,
                truth_light_dur,
                truth_loc_dur,
                truth_scene_dur,
                env._age_loc,
                env._age_scene,
                env._age_light,
                walk_run_secs,
                stationary_secs,
                walk_run_flag,
                relax_flag,
            )

            total_steps += 1
            invoke_steps += int(invoke)
            change_steps += int(changed)
            interval_steps_hit += int(on_interval)
            invoke_steps_interval += int(on_interval)
            invoke_steps_change += int(changed)
            invoke_steps_both += int(on_interval or changed)

            if invoke:
                action = env.sample_random_action()
            else:
                action = env.none_id
            observation, _, terminated, truncated, _ = env.step(action, invoke=invoke)
            done = terminated or truncated

            last_act, last_light, last_loc, last_scene = cur_act, cur_light, cur_loc, cur_scene
            last_walk_flag, last_relax_flag = cur_walk_flag, cur_relax_flag
            last_walk_flag, last_relax_flag = cur_walk_flag, cur_relax_flag

    logger.info(
        "Invoke debug summary: steps=%d invoked=%d changed=%d interval_hits=%d",
        total_steps,
        invoke_steps,
        change_steps,
        interval_steps_hit,
    )
    logger.info(
        "Invoke debug by mode: interval=%d change=%d both=%d",
        invoke_steps_interval,
        invoke_steps_change,
        invoke_steps_both,
    )


# ============================================================================
# Neural Network Models
# ============================================================================

class QNetwork(nn.Module):
    """Q-Network for DQN: maps observations to action values."""
    
    def __init__(self, observation_dim: int, num_actions: int):
        """
        Initialize Q-Network.
        
        Args:
            observation_dim: Dimension of observation space
            num_actions: Number of possible actions
        """
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(observation_dim, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     # nn.LayerNorm(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_actions),
        # )
        self.net = nn.Sequential(
            nn.Linear(observation_dim, 256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.Linear(256, 512),
            # # nn.LayerNorm(512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )
        
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute Q-values for all actions."""
        return self.net(observation)


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = int(capacity)
        self.buffer: Deque = deque(maxlen=self.capacity)
    
    def add(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of transitions, always including the most recent transition.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        # We'll return up to min(batch_size, n) items (no replacement).
        bs = min(batch_size, n)

        # Always include the most recent transition (last element of deque)
        recent = self.buffer[-1]

        if bs == 1:
            batch = [recent]
        else:
            # Sample the remaining bs-1 indices from [0, ..., n-2] (exclude the last index)
            idx = np.random.choice(n - 1, size=bs - 1, replace=False)
            batch = [self.buffer[i] for i in idx]
            batch.append(recent)

        # Optional: shuffle so the recent transition isn't always in the last position
        np.random.shuffle(batch)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def evaluate_greedy_policy(
    env: AODRecommendationEnv,
    q_network: QNetwork,
    device: torch.device,
    num_episodes: int,
    episode_steps: int,
    seed: int,
    step_seconds: int,
    invoke_interval_seconds: int,
    invoke_mode: str,
    logger: Logger = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate trained agent with greedy policy (no exploration).
    
    Returns:
        Tuple of (aggregated_stats, action_statistics)
    """
    q_network.eval()
    
    aggregated_stats: Dict[str, float] = {}
    action_statistics: Dict[str, float] = {action: 0.0 for action in RECOMMEND_ACTIONS}
    
    for episode in range(num_episodes):
        env.episode_steps = episode_steps

        random_day_id = np.random.randint(TRAIN_EPISODES, TRAIN_EPISODES + EVAL_EPISODES)
        observation, _ = env.reset(seed=seed + episode * 101, day_id=random_day_id, logger=logger)
        done = False
        prev_oracle_action = None
        last_act: Optional[int] = None
        last_light: Optional[int] = None
        last_loc: Optional[int] = None
        last_scene: Optional[int] = None
        last_walk_flag: Optional[int] = None
        last_relax_flag: Optional[int] = None
        interval_steps = max(1, int(step_seconds and invoke_interval_seconds // step_seconds))
        
        while not done:
            cur_act, cur_light, cur_loc, cur_scene = _get_truth_act_light(env)
            cur_walk_flag, cur_relax_flag = _get_walk_relax_flags(env)
            invoke = _should_invoke(
                step_idx=int(env._t),
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


            # Greedy action selection
            if invoke:
                state_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
                action = int(torch.argmax(q_network(state_tensor), dim=1).item())
            else:
                action = env.none_id
            
            observation, _, terminated, truncated, info = env.step(action, invoke=invoke)
            done = terminated or truncated
            
            # Track oracle action changes
            if prev_oracle_action != info["oracle"]:
                action_statistics[info["oracle"]] += 1
            prev_oracle_action = info["oracle"]
        
        # Aggregate statistics
        for key, value in env.stats.items():
            aggregated_stats[key] = aggregated_stats.get(key, 0.0) + float(value)
    
    # Average statistics
    for key in aggregated_stats:
        aggregated_stats[key] /= float(num_episodes)
    for action in action_statistics:
        action_statistics[action] /= float(num_episodes)
    
    q_network.train()
    return aggregated_stats, action_statistics


def run_random_episode(
    env: AODRecommendationEnv,
    episode_steps: int,
    seed: int,
    day_id: int,
    step_seconds: int,
    invoke_interval_seconds: int,
    invoke_mode: str,
    logger: Logger = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Run a single episode with random actions.
    
    Returns:
        Tuple of (episode_stats, action_statistics)
    """
    rng = np.random.default_rng(seed)
    env.episode_steps = episode_steps
    observation, _ = env.reset(seed=seed, day_id=day_id, logger=logger)
    done = False
    
    action_statistics: Dict[str, float] = {action: 0.0 for action in RECOMMEND_ACTIONS}
    prev_oracle_action = None
    last_act: Optional[int] = None
    last_light: Optional[int] = None
    last_loc: Optional[int] = None
    last_scene: Optional[int] = None
    last_walk_flag: Optional[int] = None
    last_relax_flag: Optional[int] = None
    interval_steps = max(1, int(step_seconds and invoke_interval_seconds // step_seconds))
    
    while not done:
        cur_act, cur_light, cur_loc, cur_scene = _get_truth_act_light(env)
        cur_walk_flag, cur_relax_flag = _get_walk_relax_flags(env)
        invoke = _should_invoke(
            step_idx=int(env._t),
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
            action = int(rng.integers(0, env.action_n))
        else:
            action = env.none_id
        observation, _, terminated, truncated, info = env.step(action, invoke=invoke)
        done = terminated or truncated
        
        # Track oracle action changes
        if prev_oracle_action != info["oracle"]:
            action_statistics[info["oracle"]] += 1
        prev_oracle_action = info["oracle"]
    
    return dict(env.stats), action_statistics


def evaluate_random_policy(
    env: AODRecommendationEnv,
    num_episodes: int,
    episode_steps: int,
    seed: int,
    step_seconds: int,
    invoke_interval_seconds: int,
    invoke_mode: str,
    logger: Logger = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate random policy baseline.
    
    Returns:
        Tuple of (aggregated_stats, action_statistics)
    """
    aggregated_stats: Dict[str, float] = {}
    action_statistics: Dict[str, float] = {action: 0.0 for action in RECOMMEND_ACTIONS}
    
    for episode in range(num_episodes):

        random_day_id = np.random.randint(TRAIN_EPISODES, TRAIN_EPISODES + EVAL_EPISODES)
        episode_stats, episode_action_stats = run_random_episode(
            env,
            episode_steps,
            seed + episode * 97,
            random_day_id,
            step_seconds,
            invoke_interval_seconds,
            invoke_mode,
            logger=logger,
        )
        
        # Aggregate statistics
        for key, value in episode_stats.items():
            aggregated_stats[key] = aggregated_stats.get(key, 0.0) + float(value)
        for action in action_statistics:
            action_statistics[action] += episode_action_stats[action]
    
    # Average statistics
    for key in aggregated_stats:
        aggregated_stats[key] /= float(num_episodes)
    for action in action_statistics:
        action_statistics[action] /= float(num_episodes)
    
    return aggregated_stats, action_statistics


# ============================================================================
# Logging Functions
# ============================================================================

def log_statistics(
    writer: SummaryWriter,
    prefix: str,
    stats: Dict[str, float],
    step: int
) -> None:
    """
    Log statistics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        prefix: Prefix for metric names (e.g., "train/trained")
        stats: Dictionary of statistics to log
        step: Step/episode number
    """
    # Log main metrics
    main_metrics = ["return", "success", "wrong", "miss", "sensor_cost", "delay_pen", "redundant"]
    for metric in main_metrics:
        if metric in stats:
            writer.add_scalar(f"{prefix}/{metric}", stats[metric], step)
    
    # Log scenario-specific metrics
    for key, value in stats.items():
        if key.startswith("succ_") or key.startswith("miss_"):
            writer.add_scalar(f"{prefix}/{key}", value, step)

# ============================================================================
# Training Function
# ============================================================================

def train(config: DQNConfig) -> None:
    """
    Main training loop for DQN agent.
    
    Args:
        config: Training configuration
    """
    # Setup
    set_seed(config.seed)
    config.device = f"cuda:{config.cuda}"  # TODO: Make this configurable
    device = torch.device(config.device)

    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"aod_dqn_seed{config.seed}_{timestamp}"
    run_dir = Path(config.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(run_dir / config.log_file)],
        force=True,
    )   
    logger.info(f"Training on device: {config.device}")
    logger.info(f"Config: {config}")
    logger.info(f"History length: {config.history_len}")
    logger.info(f"Run name: {config.run_name}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"NONE action probability: {1}")

    if config.debug_invoke_episodes > 0:
        debug_invoke_logic(config, logger)
        return

    # Initialize environment
    env = AODRecommendationEnv(
        seed=config.seed,
        history_len=config.history_len,
        episode_steps=config.train_episode_steps,
        loc_always_available=config.loc_always_available,
    )
    observation_dim = env.obs_dim
    num_actions = env.action_n
    
    # Initialize networks
    q_network = QNetwork(observation_dim, num_actions).to(device)
    target_network = QNetwork(observation_dim, num_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    # Initialize optimizer and replay buffer
    optimizer = optim.Adam(q_network.parameters(), lr=config.lr)
    replay_buffer = ReplayBuffer(config.replay_size)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_text("meta/actions", "\n".join([f"{i}: {a}" for i, a in enumerate(ALL_ACTIONS)]), 0)
    writer.add_text("meta/profiles", "trajectory_file", 0)
    writer.add_text("meta/config", str(asdict(config)), 0)
    
    logger.info(f"Observation dim: {observation_dim}, Actions: {num_actions} "
                f"(probes={len(PROBE_ACTIONS)}, recommendations={len(RECOMMEND_ACTIONS)})")
    logger.info("Training profiles: trajectory_file")
    logger.info(f"TensorBoard logdir: {run_dir}")
    
    # Training state
    global_step = 0
    epislon_step = 0
    episode_returns = deque(maxlen=20)  # Moving average window
    start_time = time.time()
    env_random = AODRecommendationEnv(
        seed=config.seed + 12345,
        history_len=config.history_len,
        loc_always_available=config.loc_always_available,
    )
    
    # Training loop
    logger.info(f"Training for {TRAIN_EPISODES} episodes")
    for episode in range(1, TRAIN_EPISODES + 1):
        env.episode_steps = config.train_episode_steps

        day_id = np.random.randint(1, TRAIN_EPISODES)
        observation, _ = env.reset(seed=config.seed + episode * 13, day_id=day_id, logger=logger)
        done = False
        episode_return = 0.0
        last_act: Optional[int] = None
        last_light: Optional[int] = None
        last_loc: Optional[int] = None
        last_scene: Optional[int] = None
        last_walk_flag: Optional[int] = None
        last_relax_flag: Optional[int] = None
        interval_steps = max(1, int(config.step_seconds and config.invoke_interval_seconds // config.step_seconds))
        
        # Episode loop
        while not done:
            cur_act, cur_light, cur_loc, cur_scene = _get_truth_act_light(env)
            cur_walk_flag, cur_relax_flag = _get_walk_relax_flags(env)
            invoke = _should_invoke(
                step_idx=int(env._t),
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
                invoke_mode=config.invoke_mode,
            )
            last_act, last_light, last_loc, last_scene = cur_act, cur_light, cur_loc, cur_scene
            last_walk_flag, last_relax_flag = cur_walk_flag, cur_relax_flag

            # Compute epsilon and select action
            if invoke:
                epsilon = compute_epsilon(
                    config.eps_start, config.eps_end, config.eps_decay_steps, epislon_step
                )
                if len(replay_buffer) < config.warmup_steps:
                    action = env.sample_random_action()
                elif random.random() < epsilon:
                    action = env.sample_random_action()
                    epislon_step += 1
                else:
                    # Exploitation: use greedy action
                    epislon_step += 1
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
                        action = int(torch.argmax(q_network(state_tensor), dim=1).item())
            else:
                action = env.none_id

            # Take step in environment
            next_observation, reward, terminated, truncated, info = env.step(action, invoke=invoke)
            done = terminated or truncated

            # Store transition in replay buffer only when DQN was invoked
            if invoke:
                if info["oracle"] == "NONE":
                    if random.random() < 1:
                        replay_buffer.add(observation, action, reward, next_observation, truncated)
                else:
                    replay_buffer.add(observation, action, reward, next_observation, truncated)

            observation = next_observation
            episode_return += float(reward)
            global_step += 1

            # Train network if ready
            if len(replay_buffer) >= config.warmup_steps and (global_step % config.train_every == 0) and invoke:
                # Sample batch from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size)

                # Convert to tensors
                states_tensor = torch.from_numpy(states).float().to(device)
                next_states_tensor = torch.from_numpy(next_states).float().to(device)
                actions_tensor = torch.from_numpy(actions).long().to(device)
                rewards_tensor = torch.from_numpy(rewards).float().to(device)
                dones_tensor = torch.from_numpy(dones).float().to(device)

                # Compute Q-values for current states
                q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                # Compute target Q-values using target network
                with torch.no_grad():
                    next_actions = torch.argmax(q_network(next_states_tensor), dim=1)
                    next_q_values = target_network(next_states_tensor).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)
                    target_q_values = rewards_tensor + config.gamma * (1.0 - dones_tensor) * next_q_values

                # Compute loss and update network
                loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), config.grad_clip)
                optimizer.step()

                # Log training loss
                writer.add_scalar("train/trained/loss", float(loss.item()), global_step)

                # Update target network periodically
                if global_step % config.target_update_every == 0:
                    target_network.load_state_dict(q_network.state_dict())

        # Log episode statistics
        episode_returns.append(episode_return)
        writer.add_scalar("train/trained/episode_return", episode_return, episode)
        writer.add_scalar(
            "train/trained/epsilon",
            compute_epsilon(config.eps_start, config.eps_end, config.eps_decay_steps, epislon_step),
            episode
        )
        log_statistics(writer, "train/trained", env.stats, episode)

        # Log random baseline
        if config.random_train_every > 0 and (episode % config.random_train_every == 0):
            random_stats, _ = run_random_episode(
                env_random,
                config.train_episode_steps,
                seed=config.seed + 50_000 + episode,
                day_id=day_id,
                step_seconds=config.step_seconds,
                invoke_interval_seconds=config.invoke_interval_seconds,
                invoke_mode=config.invoke_mode,
                logger=logger,
            )
            log_statistics(writer, "train/random", random_stats, episode)

        # Print progress
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_return = np.mean(episode_returns)
            current_epsilon = compute_epsilon(
                config.eps_start, config.eps_end, config.eps_decay_steps, epislon_step
            )
            logger.info(
                f"Ep {episode:4d} | steps {global_step:7d} | eps {current_epsilon:.3f} | "
                f"R(ep) {episode_return:8.2f} | R(ma20) {avg_return:8.2f} | elapsed {elapsed_time:6.1f}s"
            )

        # Periodic evaluation
        if episode % config.eval_every == 0:
            env_eval = AODRecommendationEnv(
                seed=config.seed + 1,
                history_len=config.history_len,
                loc_always_available=config.loc_always_available,
            )
            env_rand = AODRecommendationEnv(
                seed=config.seed + 1,
                history_len=config.history_len,
                loc_always_available=config.loc_always_available,
            )

            # Evaluate trained agent
            trained_stats, trained_action_stats = evaluate_greedy_policy(
                env_eval,
                q_network,
                device,
                config.eval_episodes,
                config.eval_episode_steps,
                config.seed + 10_000 + episode,
                config.step_seconds,
                config.invoke_interval_seconds,
                config.invoke_mode,
                logger=logger,
            )

            # Evaluate random baseline
            random_stats, random_action_stats = evaluate_random_policy(
                env_rand,
                config.eval_episodes,
                config.eval_episode_steps,
                config.seed + 20_000 + episode,
                config.step_seconds,
                config.invoke_interval_seconds,
                config.invoke_mode,
                logger=logger,
            )

            # Log evaluation results
            log_statistics(writer, "eval/trained", trained_stats, episode)
            log_statistics(writer, "eval/random", random_stats, episode)

            # Print evaluation summary
            logger.info("--- Eval (avg) ---")
            logger.info(
                f"Trained | Return {trained_stats.get('return', 0):7.2f} | "
                f"Succ {trained_stats.get('success', 0):6.2f} | "
                f"Wrong {trained_stats.get('wrong', 0):6.2f} | "
                f"Miss {trained_stats.get('miss', 0):6.2f} | "
                f"Cost {trained_stats.get('sensor_cost', 0):6.2f}"
            )
            logger.info(f"Statistics for trained episodes {episode}: {trained_action_stats}")
            logger.info(
                f"Random  | Return {random_stats.get('return', 0):7.2f} | "
                f"Succ {random_stats.get('success', 0):6.2f} | "
                f"Wrong {random_stats.get('wrong', 0):6.2f} | "
                f"Miss {random_stats.get('miss', 0):6.2f} | "
                f"Cost {random_stats.get('sensor_cost', 0):6.2f}"
            )
            logger.info(f"Statistics for random episodes {episode}: {random_action_stats}")
            logger.info("----------------\n")

            checkpoint = {
                "q_state_dict": q_network.state_dict(),
                "config": asdict(config),
                "actions": ALL_ACTIONS
            }
            checkpoint_path = run_dir / f"dqn_aod_ckpt_episode_{episode}.pt"
            torch.save(checkpoint, checkpoint_path)

    # Save checkpoint
    checkpoint = {
        "q_state_dict": q_network.state_dict(),
        "config": asdict(config),
        "actions": ALL_ACTIONS
    }
    checkpoint_path = run_dir / "dqn_aod_ckpt.pt"
    torch.save(checkpoint, checkpoint_path)
    writer.add_text("meta/checkpoint", str(checkpoint_path.resolve()), config.train_episodes)
    writer.close()

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"TensorBoard logdir: {run_dir}")


# ============================================================================
# Command-Line Interface
# ============================================================================

def parse_arguments() -> DQNConfig:
    """Parse command-line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Train DQN agent for AOD recommendation environment"
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train_episodes", type=int, default=8000, help="Number of training episodes")
    parser.add_argument("--train_episode_steps", type=int, default=10*60*60, help="Training episode length (steps)")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate every N episodes")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--eval_episode_steps", type=int, default=10*60*60, help="Evaluation episode length (steps)")
    parser.add_argument("--logdir", type=str, default="logs_new", help="Log directory")
    parser.add_argument("--run_name", type=str, default="run", help="Run name")
    parser.add_argument("--log_file", type=str, default="train_dqn_tensorboard_v2_balancedbf.log", help="Log file name")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--replay_size", type=int, default=14400, help="Replay buffer size")
    parser.add_argument("--warmup_steps", type=int, default=14400, help="Warmup steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clipping threshold")
    parser.add_argument("--history_len", type=int, default=0, help="Has history")
    parser.add_argument(
        "--loc_always_available",
        action="store_true",
        default=False,
        help="Expose location without probe (reduces action space)",
    )
    parser.add_argument("--eps_decay_steps", type=int, default=3600*60, help="Steps to decay epsilon")
    parser.add_argument("--invoke_interval_seconds", type=int, default=60, help="Invoke DQN every N seconds")
    parser.add_argument(
        "--invoke_mode",
        type=str,
        choices=["interval", "change", "both"],
        default="both",
        help="Invoke policy: interval | change | both",
    )
    parser.add_argument("--step_seconds", type=int, default=1, help="Seconds per data step")
    parser.add_argument(
        "--debug_invoke_episodes",
        type=int,
        default=0,
        help="Run invoke debug on N random episodes and exit",
    )

    parser.add_argument(
        "--random_train_every",
        type=int,
        default=25,
        help="Log random baseline every N episodes (0 to disable)"
    )

    args = parser.parse_args()

    return DQNConfig(
        seed=args.seed,
        train_episodes=args.train_episodes,
        train_episode_steps=args.train_episode_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_episode_steps=args.eval_episode_steps,
        random_train_every=args.random_train_every,
        logdir=args.logdir,
        run_name=args.run_name,
        log_file=args.log_file,
        cuda=args.cuda,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        history_len=args.history_len,
        loc_always_available=args.loc_always_available,
        eps_decay_steps=args.eps_decay_steps,
        invoke_interval_seconds=args.invoke_interval_seconds,
        invoke_mode=args.invoke_mode,
        step_seconds=args.step_seconds,
        debug_invoke_episodes=args.debug_invoke_episodes,
    )


if __name__ == "__main__":
    config = parse_arguments()
    train(config)