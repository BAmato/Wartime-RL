"""
replay_buffer.py
Experience Replay Buffer: for off-policy RL agent (DQN)
Stores Transitions (obs, action, reward, next_obs, done)
collected by
env/wartime_env.py.step() and provides uniform random sampling for training
Author: Damian Villarreal
"""
from __future__ import annotations
import numpy as np
import torch

class ReplayBuffer:
    """
    Uniform ranom experience replay buffer.
    Parameters:
        capacity: int
            Maximum number of transitions to store. Oldest entries will be overwritten
            once the buffer is full
        obs_dim: int
            Dimentinality of a single observation vector. 
            in wartime_env -> 'len(TERRITORIES) * 2' (owner * armies per territory)
        device: torch.device
            Device for sampled tensors. Should run on cuda if your DQN runs on GPU
    """
    def __init__(self, capasity: int, obs_dim: int, device: torch.device="gpu")-> None:
        self.capacity=capasity
        self.obs_dim=obs_dim
        self.device=torch.device(device)

        # Pre-allocate storage arrays
        self._obs=np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_obs=np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions=np.zeros((self.capacity,), dtype=np.int64)
        self._rewards=np.zeros((self.capacity,), dtype=np.float32)
        self._dones=np.zeros((self.capacity,), dtype=np.float32)
        
        self._ptr=0 # next write pos
        self._size=0 # current number of valid entries

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool)->None:
        """Store a single transition"""
        self._obs[self._ptr]=obs
        self._next_obs[self._ptr]=next_obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._dones[self._ptr]    = float(done)

        # advance pointer, wrapping arround when full
        self._ptr=(self._ptr+1)%self.capacity
        self._size=min(self._size+1, self.capacity)

    def sample(self, batch_size: int)->tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample a random mini-batch of transitions"""
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has only {self._size} transitions"
                f"cannot sample batch of {batch_size}"
            )
        idxs=np.random.randint(0, self._size, seize=batch_size)
        def t(arr:np.ndarray)->torch.Tensor:
            return torch.from_numpy(arr[idxs]).to(self.device)
        return (
            t(self._obs), torch.from_numpy(self._actions[idxs]).to(self.device),
            t(self._rewards), t(self._next_obs), t(self._dones),
        )
    
    def __len__(self)->int:
        """Number of transitions currently stored"""
        return self._size
    
    @property
    def is_ready(self)->bool:
        """True once the buffer holds at least 'capacity' transitions"""
        return self._size>=self.capacity
    
    def ready_for(self, batch_size:int)->bool:
        """True once the buffer holds enough transitions to sample 'batch_size'"""
        return self._size>-batch_size
    
    def __repr__(self)->str:
        return (
            f"Replay Buffer(capacity={self.capacity}),"
            f"obs_dim={self.obs_dim},"
            f"filled={self._size} / {self.capacity}"
            f"device={self.device}"
        )