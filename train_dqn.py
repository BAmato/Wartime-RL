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


def encode_final_state(raw_env) -> str:
    """Encode territory state as '0A8|1E3|2N1|...'"""
    owner_code = {"agent": "A", "enemy": "E", "neutral": "N"}
    parts = []
    for idx, (name, data) in enumerate(raw_env.state.items()):
        parts.append(f"{idx}{owner_code[data['owner']]}{data['armies']}")
    return "|".join(parts)


def encode_continents(raw_env) -> str:
    """Encode continent control as 'NA:A,SA:N,...'"""
    from env.map_config import CONTINENTS
    codes = {"North America": "NA", "South America": "SA"}
    results = []
    for cont, data in CONTINENTS.items():
        owners = [raw_env.state[t]["owner"] for t in data["territories"]]
        if all(o == "agent" for o in owners):
            ctrl = "A"
        elif all(o == "enemy" for o in owners):
            ctrl = "E"
        else:
            ctrl = "N"
        results.append(f"{codes.get(cont, cont)}:{ctrl}")
    return ",".join(results)


def train():
    args = parse_args()

    cfg = TrainingConfig()
    if args.steps:
        cfg.total_steps = args.steps
    cur_cfg = CurriculumConfig()

    env = WartimeEnv(render_mode="rgb_array", curriculum_level=args.level)
    raw_env = env  # keep unwrapped reference
    env = RecordVideo(env, video_folder=os.path.join(args.out, "videos"), episode_trigger=lambda ep: ep % 100 == 0)
    env = RecordEpisodeStatistics(env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg)
    tracker = CurriculumTracker(raw_env, cur_cfg)

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
        writer.writerow(["# beginner_max_turns", cur_cfg.phase[0].max_turns])
        writer.writerow(["# beginner_agent_armies", cur_cfg.phase[0].agent_start_armies])
        writer.writerow(["# beginner_enemy_armies", cur_cfg.phase[0].enemy_start_armies])
        writer.writerow(["# reward_clip", "none"])
        writer.writerow([])  # blank line separator
        # Column headers
        writer.writerow([
            "episode", "global_step", "outcome", "ep_steps", "turns", "reward",
            "agent_terr", "enemy_terr", "epsilon", "curriculum_level", "mean_loss",
            "agent_armies", "enemy_armies",
            "phase_reinforce", "phase_attack", "phase_fortify",
            "combat_wins", "combat_losses", "territory_losses", "fortify_count",
            "continents", "final_state",
        ])

    episode = 0
    ep_reward = 0.0
    ep_steps = 0
    ep_losses: list[float] = []
    start = time.time()

    # per-episode trackers
    phase_counts = {"reinforce": 0, "attack": 0, "fortify": 0}
    combat_wins = 0
    combat_losses = 0
    territory_losses = 0
    fortify_count = 0

    obs, _ = env.reset()

    while agent.total_steps < cfg.total_steps:
        action = agent.select_action(obs, env=raw_env)
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.push(obs, action, reward, next_obs, terminated)

        loss = agent.train_step()
        if loss is not None:
            ep_losses.append(loss)

        # track per-step info
        phase_counts[info.get("turn_phase", "attack")] = (
            phase_counts.get(info.get("turn_phase", "attack"), 0) + 1
        )
        cr = info.get("combat_result", "none")
        if cr == "win_territory":
            combat_wins += 1
        elif cr == "lose_combat":
            combat_losses += 1
        elif cr == "lose_territory":
            territory_losses += 1
        if info.get("action_type") == "fortify":
            fortify_count += 1

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

            agent_armies = sum(
                d["armies"] for d in raw_env.state.values() if d["owner"] == "agent"
            )
            enemy_armies = sum(
                d["armies"] for d in raw_env.state.values() if d["owner"] == "enemy"
            )

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    episode, agent.total_steps, outcome, ep_steps,
                    info.get("turns", 0),
                    f"{ep_reward:.2f}", info["agent_territories"],
                    info["enemy_territories"], f"{agent.epsilon():.4f}",
                    raw_env.curriculum_level, f"{mean_loss:.6f}",
                    agent_armies, enemy_armies,
                    phase_counts["reinforce"], phase_counts["attack"], phase_counts["fortify"],
                    combat_wins, combat_losses, territory_losses, fortify_count,
                    encode_continents(raw_env),
                    encode_final_state(raw_env),
                ])

            if episode % 100 == 0:
                elapsed = time.time() - start
                print(
                    f"Ep {episode:5d} | steps {agent.total_steps:7d} | "
                    f"WR {tracker.win_rate():.1%} | eps {agent.epsilon():.3f} | "
                    f"loss {mean_loss:.4f} | lvl {raw_env.curriculum_level} | {elapsed:.0f}s"
                )

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_losses.clear()
            # reset per-episode trackers
            phase_counts = {"reinforce": 0, "attack": 0, "fortify": 0}
            combat_wins = 0
            combat_losses = 0
            territory_losses = 0
            fortify_count = 0

    env.close()
    save_path = os.path.join(args.out, f"dqn_final_{timestamp}.pt")
    agent.save(save_path)
    print(f"\nDQN training complete — {episode} episodes, {agent.total_steps} steps")
    print(f"Model : {save_path}")
    print(f"Log   : {log_path}")


if __name__ == "__main__":
    train()