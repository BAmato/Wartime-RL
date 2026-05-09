"""
agents/dqn.py
Double DQN agent with experience replay and epsilon-greedy exploration.
Uses soft target-network updates (tau) instead of hard periodic copies.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.replay_buffer import ReplayBuffer
from config import TrainingConfig


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Double DQN: online network selects actions, target network evaluates them."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: TrainingConfig = None):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = QNetwork(obs_dim, n_actions).to(self.device)
        self.target = QNetwork(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.cfg.learning_rate)
        self.buffer = ReplayBuffer(self.cfg.replay_capacity, obs_dim, self.device)
        self.total_steps = 0

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        ratio = min(self.total_steps / self.cfg.eps_decay, 1.0)
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * (1.0 - ratio)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> int:
        if np.random.random() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return int(self.online(obs_t).argmax(dim=1).item())

    # ------------------------------------------------------------------
    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

    # ------------------------------------------------------------------
    def train_step(self) -> float | None:
        if not self.buffer.ready_for(self.cfg.batch_size):
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            # Double DQN: online selects, target evaluates
            next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        current_q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        # Soft target update
        for op, tp in zip(self.online.parameters(), self.target.parameters()):
            tp.data.copy_(self.cfg.tau * op.data + (1.0 - self.cfg.tau) * tp.data)

        return float(loss.item())

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]