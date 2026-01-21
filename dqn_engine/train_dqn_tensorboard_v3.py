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
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Tuple

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

    # Exploration hyperparameters
    eps_start: float = 1.0  # Initial epsilon (exploration rate)
    eps_end: float = 0.05  # Final epsilon (minimum exploration)
    eps_decay_steps: int = 3600 * 400  # Steps to decay epsilon

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

        while not done:
            # Greedy action selection
            state_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
            action = int(torch.argmax(q_network(state_tensor), dim=1).item())

            observation, _, terminated, truncated, info = env.step(action)
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

    while not done:
        action = int(rng.integers(0, env.action_n))
        observation, _, terminated, truncated, info = env.step(action)
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
            env, episode_steps, seed + episode * 97, random_day_id, logger=logger
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

    # Initialize environment
    env = AODRecommendationEnv(seed=config.seed, history_len=config.history_len)
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
    env_random = AODRecommendationEnv(seed=config.seed + 12345, history_len=config.history_len)

    # Training loop
    for episode in range(1, config.train_episodes + 1):
        env.episode_steps = config.train_episode_steps

        day_id = np.random.randint(0, TRAIN_EPISODES)
        observation, _ = env.reset(seed=config.seed + episode * 13, day_id=day_id, logger=logger)
        done = False
        episode_return = 0.0

        # Episode loop
        while not done:
            # Compute epsilon and select action
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

            # Take step in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            # logger.info("--------------------------------")
            # logger.info(ALL_ACTIONS[action])
            # logger.info(env._loc_obs)
            # logger.info(env._scene_obs)
            # logger.info(env._loc_hist)
            # logger.info(env._scene_hist)
            # logger.info("--------------------------------")
            done = terminated or truncated

            # Store transition in replay buffer
            if info["oracle"] == "NONE":
                if random.random() < 0.1:
                    replay_buffer.add(observation, action, reward, next_observation, truncated)
            else:
                replay_buffer.add(observation, action, reward, next_observation, truncated)

            observation = next_observation
            episode_return += float(reward)
            global_step += 1

            # Train network if ready
            if len(replay_buffer) >= config.warmup_steps and (global_step % config.train_every == 0):
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
                env_random, config.train_episode_steps, seed=config.seed + 50_000 + episode, day_id=day_id,
                logger=logger
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
            env_eval = AODRecommendationEnv(seed=config.seed + 1, history_len=config.history_len)
            env_rand = AODRecommendationEnv(seed=config.seed + 1, history_len=config.history_len)

            # Evaluate trained agent
            trained_stats, trained_action_stats = evaluate_greedy_policy(
                env_eval, q_network, device, config.eval_episodes,
                config.eval_episode_steps, config.seed + 10_000 + episode,
                logger=logger
            )

            # Evaluate random baseline
            random_stats, random_action_stats = evaluate_random_policy(
                env_rand, config.eval_episodes,
                config.eval_episode_steps, config.seed + 20_000 + episode,
                logger=logger
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
    parser.add_argument("--train_episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--train_episode_steps", type=int, default=24 * 60 * 6, help="Training episode length (steps)")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate every N episodes")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--eval_episode_steps", type=int, default=24 * 60 * 6, help="Evaluation episode length (steps)")
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
    parser.add_argument("--eps_decay_steps", type=int, default=3600 * 100, help="Steps to decay epsilon")

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
        eps_decay_steps=args.eps_decay_steps,

    )


if __name__ == "__main__":
    config = parse_arguments()
    train(config)