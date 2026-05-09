"""
config.py
All reward weights, training hyperparameters, and 
curriculum settings live here. They can be changed here in one place instead of hardcoding 
agent code

Author: Damian Villarreal
"""

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """All scalar reward components used in wartime_env.step()
    Tweak these instead of inside env
    """

    capture_neutral: float = +1.0
    win_combat: float = +2.0
    lose_combat: float = -1.0      # was -3.0 — too punishing

    border_pressure: float = -0.1   # penalty per turn based on enemy army ratio at borders
    defensive_fortify: float = +0.3  # reward for fortifying a territory adjacent to an enemy

    survival: float = +0.01
    continent_scale: float = +0.5

    win_game: float = +20.0
    lose_game: float = -20.0       # was -40.0 — match the win reward
    
    supply_drop: float = +1.0
    reinforcements: float = +0.5

@dataclass
class GameplayConfig:
    """Core non-learning game mechanics."""
    min_reinforcements: int = 3
    territories_per_reinforcement: int = 3
    max_armies_per_territory: int = 20
    max_pending_reinforcements: int = 20
    attack_bonus_dice: int = 1   # extra die granted by reinforcements event
    supply_drop_armies: int = 2
    retreat_penalty_armies: int = 1
    combat_loss_armies: int = 1


@dataclass
class TrainingConfig:
    """DQN training hyperparameters"""
    total_steps: int = 200_000
    batch_size: int = 64
    replay_capacity: int = 50_000
    learning_rate: float = 2.5e-4
    gamma: float = 0.95
    tau: float = 0.01
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 50_000
    grad_clip: float = 10.0


@dataclass
class CurriculumPhase:
    """Defines one difficulty phase of training"""
    name: str
    max_steps: int
    agent_start_armies: int
    enemy_start_armies: int
    n_neutral_armies: int
    random_event_prob: float


@dataclass
class CurriculumConfig:
    """Ordered list of curriculum phases.
    Pass curriculum_level=0/1/2 into wartime_env to select phase.
    Advance phases based on win rate over a rolling window of episodes.
    """
    phase: list = field(default_factory=lambda: [
        CurriculumPhase(
            name="Beginner",
            max_steps=300,
            agent_start_armies=8,
            enemy_start_armies=1,
            n_neutral_armies=1,
            random_event_prob=0.05,
        ),
        CurriculumPhase(
            name="Intermediate",
            max_steps=150,
            agent_start_armies=3,
            enemy_start_armies=3,
            n_neutral_armies=1,
            random_event_prob=0.10,
        ),
        CurriculumPhase(
            name="Hard",
            max_steps=200,
            agent_start_armies=3,
            enemy_start_armies=5,
            n_neutral_armies=2,
            random_event_prob=0.15,
        )
    ])
    # win rate threshold to advance to next phase
    advance_win_rate: float = 0.70
    # number of recent episodes to measure win rate over
    eval_window: int = 50


class CurriculumTracker:
    """Tracks episode outcomes and decides when to advance the curriculum.
    Used in training loop.
    """

    def __init__(self, env, cfg: CurriculumConfig = None):
        self.env = env
        self.cfg = cfg or CurriculumConfig()
        self._history: list[str] = []

    def record(self, outcome: str) -> bool:
        """Record an episode outcome and advance the phase if ready"""
        self._history.append(outcome)
        if len(self._history) < self.cfg.eval_window:
            return False
        win_rate = self.win_rate()
        max_level = len(self.cfg.phase) - 1

        if win_rate >= self.cfg.advance_win_rate and self.env.curriculum_level < max_level:
            self.env.curriculum_level += 1
            self._history.clear()
            print(f"[Curriculum] Win rate {win_rate:.0%} >= {self.cfg.advance_win_rate:.0%} "
                  f"— advanced to level {self.env.curriculum_level} "
                  f"({self.cfg.phase[self.env.curriculum_level].name})")
            return True
        return False

    def win_rate(self) -> float:
        """Current Win Rate over the last eval_window episodes"""
        if not self._history:
            return 0.0
        return self._history.count("win") / len(self._history)

    def __repr__(self) -> str:
        phase = self.cfg.phase[self.env.curriculum_level]
        return (f"CurriculumTracker(level={self.env.curriculum_level}, "
                f"phase={phase.name}, "
                f"win_rate={self.win_rate():.0%}, "
                f"episodes={len(self._history)}/{self.cfg.eval_window})")