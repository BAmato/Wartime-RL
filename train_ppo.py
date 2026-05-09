"""
train_ppo.py
PPO training loop with curriculum advancement and CSV metric logging.

Usage:
    python train_ppo.py
    python train_ppo.py --steps 200000 --level 0
"""
from __future__ import annotations
import torch
import argparse
import csv
import os
import time

import numpy as np

from agents.ppo import PPOAgent, PPOConfig
from config import CurriculumConfig, CurriculumTracker
from env.wartime_env import WartimeEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None, help="Override total_steps")
    p.add_argument("--level", type=int, default=0, help="Starting curriculum level (0-2)")
    p.add_argument("--out", type=str, default="results", help="Output directory")
    return p.parse_args()


def train():
    args = parse_args()

    ppo_cfg = PPOConfig()
    if args.steps:
        ppo_cfg.total_steps = args.steps
    cur_cfg = CurriculumConfig()

    env = WartimeEnv(render_mode=None, curriculum_level=args.level)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(obs_dim, n_actions, ppo_cfg)
    tracker = CurriculumTracker(env, cur_cfg)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "ppo_training_log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "global_step", "outcome", "ep_steps", "reward",
            "agent_terr", "enemy_terr", "curriculum_level",
        ])

    # Rollout buffers (reused each iteration)
    obs_buf:  list = []
    act_buf:  list = []
    rew_buf:  list = []
    val_buf:  list = []
    logp_buf: list = []
    done_buf: list = []

    episode = 0
    global_step = 0
    ep_reward = 0.0
    ep_steps = 0
    start = time.time()

    obs, _ = env.reset()

    while global_step < ppo_cfg.total_steps:
        # ---- collect one rollout ----
        obs_buf.clear(); act_buf.clear(); rew_buf.clear()
        val_buf.clear(); logp_buf.clear(); done_buf.clear()

        for _ in range(ppo_cfg.rollout_steps):
            action, log_prob, value = agent.select_action(obs, env=env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs.copy())
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)
            logp_buf.append(log_prob)
            done_buf.append(float(done))

            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            global_step += 1

            if done:
                episode += 1
                outcome = (
                    "win"     if info["enemy_territories"] == 0 else
                    "loss"    if info["agent_territories"] == 0 else
                    "timeout"
                )
                tracker.record(outcome)

                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        episode, global_step, outcome, ep_steps,
                        f"{ep_reward:.2f}", info["agent_territories"],
                        info["enemy_territories"], env.curriculum_level,
                    ])

                if episode % 50 == 0:
                    elapsed = time.time() - start
                    print(
                        f"Ep {episode:5d} | steps {global_step:7d} | "
                        f"WR {tracker.win_rate():.1%} | lvl {env.curriculum_level} | "
                        f"{elapsed:.0f}s"
                    )

                obs, _ = env.reset()
                ep_reward = 0.0
                ep_steps = 0

        # ---- PPO update ----
        _, _, last_val = agent.select_action(obs)
        adv, ret = agent.compute_advantages(rew_buf, val_buf, done_buf, last_val)
        agent.update(
            np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.int64),
            np.array(logp_buf, dtype=np.float32),
            adv, ret,
        )

    env.close()
    save_path = os.path.join(args.out, "ppo_final.pt")
    agent.save(save_path)
    print(f"\nPPO training complete — {episode} episodes, {global_step} steps")
    print(f"Model : {save_path}")
    print(f"Log   : {log_path}")


if __name__ == "__main__":
    train()