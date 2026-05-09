"""
train_dqn.py
Headless DQN training with curriculum advancement and CSV metric logging.

Usage:
    python train_dqn.py
    python train_dqn.py --steps 100000 --level 0
"""
from __future__ import annotations
import torch
import argparse
import csv
import os
import time

import numpy as np

from agents.dqn import DQNAgent
from config import CurriculumConfig, CurriculumTracker, TrainingConfig
from env.wartime_env import WartimeEnv
from datetime import datetime
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None, help="Override total_steps")
    p.add_argument("--level", type=int, default=0, help="Starting curriculum level (0-2)")
    p.add_argument("--out", type=str, default="results", help="Output directory")
    return p.parse_args()


def train():
    args = parse_args()

    cfg = TrainingConfig()
    if args.steps:
        cfg.total_steps = args.steps
    cur_cfg = CurriculumConfig()

    env = WartimeEnv(render_mode="rgb_array", curriculum_level=args.level)
    raw_env = env  # keep reference before wrapping
    env = RecordVideo(env, video_folder=os.path.join(args.out, "videos"), episode_trigger=lambda ep: ep % 100 == 0)
    env = RecordEpisodeStatistics(env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg)
    tracker = CurriculumTracker(raw_env, cur_cfg)  # use raw_env

    os.makedirs(args.out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out, f"dqn_training_log_{timestamp}.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Metadata header
        writer.writerow(["# run_timestamp", timestamp])
        writer.writerow(["# total_steps", cfg.total_steps])
        writer.writerow(["# learning_rate", cfg.learning_rate])
        writer.writerow(["# gamma", cfg.gamma])
        writer.writerow(["# tau", cfg.tau])
        writer.writerow(["# eps_decay", cfg.eps_decay])
        writer.writerow(["# grad_clip", cfg.grad_clip])
        writer.writerow(["# batch_size", cfg.batch_size])
        writer.writerow(["# replay_capacity", cfg.replay_capacity])
        writer.writerow(["# beginner_max_steps", cur_cfg.phase[0].max_steps])
        writer.writerow(["# beginner_agent_armies", cur_cfg.phase[0].agent_start_armies])
        writer.writerow(["# beginner_enemy_armies", cur_cfg.phase[0].enemy_start_armies])
        writer.writerow(["# reward_clip", "none"])
        writer.writerow([])  # blank line separator
        # Column headers
        writer.writerow([
            "episode", "global_step", "outcome", "ep_steps", "reward",
            "agent_terr", "enemy_terr", "epsilon", "curriculum_level", "mean_loss",
        ])

    episode = 0
    ep_reward = 0.0
    ep_steps = 0
    ep_losses: list[float] = []
    start = time.time()

    obs, _ = env.reset()

    while agent.total_steps < cfg.total_steps:
        action = agent.select_action(obs, env=env)
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.push(obs, action, reward, next_obs, terminated)

        loss = agent.train_step()
        if loss is not None:
            ep_losses.append(loss)

        obs = next_obs
        ep_reward += reward
        ep_steps += 1

        if done:
            episode += 1
            outcome = (
                "win"     if info["enemy_territories"] == 0 else
                "loss"    if info["agent_territories"] == 0 else
                "timeout"
            )
            tracker.record(outcome)
            mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    episode, agent.total_steps, outcome, ep_steps,
                    f"{ep_reward:.2f}", info["agent_territories"],
                    info["enemy_territories"], f"{agent.epsilon():.4f}",
                    env.curriculum_level, f"{mean_loss:.6f}",
                ])

            if episode % 100 == 0:
                elapsed = time.time() - start
                print(
                    f"Ep {episode:5d} | steps {agent.total_steps:7d} | "
                    f"WR {tracker.win_rate():.1%} | eps {agent.epsilon():.3f} | "
                    f"loss {mean_loss:.4f} | lvl {env.curriculum_level} | {elapsed:.0f}s"
                )
                
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_losses.clear()

    env.close()
    save_path = os.path.join(args.out, f"dqn_final_{timestamp}.pt")
    agent.save(save_path)
    print(f"\nDQN training complete — {episode} episodes, {agent.total_steps} steps")
    print(f"Model : {save_path}")
    print(f"Log   : {log_path}")


if __name__ == "__main__":
    train()