"""
agents/ppo.py
Proximal Policy Optimization (clip variant) for Wartime-RL.
Uses a shared-trunk actor-critic network and GAE advantage estimation.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    rollout_steps: int = 2048
    batch_size: int = 64
    total_steps: int = 200_000


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class PPOAgent:
    """PPO with clipped surrogate objective and GAE."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: PPOConfig = None):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, env=None) -> tuple[int, float, float]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.net.act(obs_t)
        return action.item(), log_prob.item(), value.item()

    # ------------------------------------------------------------------
    def compute_advantages(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[float],
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    def update(
        self,
        obs_buf: np.ndarray,
        act_buf: np.ndarray,
        logp_buf: np.ndarray,
        adv_buf: np.ndarray,
        ret_buf: np.ndarray,
    ) -> float:
        obs_t = torch.FloatTensor(obs_buf).to(self.device)
        act_t = torch.LongTensor(act_buf).to(self.device)
        logp_old = torch.FloatTensor(logp_buf).to(self.device)
        adv_t = torch.FloatTensor(adv_buf).to(self.device)
        ret_t = torch.FloatTensor(ret_buf).to(self.device)
        
        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_t)
        total_loss = 0.0
        for _ in range(self.cfg.n_epochs):
            idxs = np.random.permutation(n)
            for start in range(0, n, self.cfg.batch_size):
                b = idxs[start: start + self.cfg.batch_size]
                logits, values = self.net(obs_t[b])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[b])
                entropy = dist.entropy().mean()

                ratio = (logp - logp_old[b]).exp()
                clipped = ratio.clamp(1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
                pg_loss = -torch.min(ratio * adv_t[b], clipped * adv_t[b]).mean()
                v_loss = nn.functional.mse_loss(values, ret_t[b])

                loss = pg_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])