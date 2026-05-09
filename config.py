"""
config.py
All reward weights, training hyperparameters, and
curriculum settings live here. They can be changed here in one place instead of hardcoding
agent code

Author: Damian Villarreal, Bryan Amato
"""

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """All scalar reward components used in wartime_env.step()
    Tweak these instead of inside env
    """

    capture_neutral: float = +1.5   # was +1.0
    win_combat: float = +25.0    # was +5.0
    lose_combat: float = -1.0      # was -1.0 dice loss without territory change
    lose_territory: float = -5.0   # applied when enemy captures an agent territory
    deploy: float = +0.5 #0.05          # per army placed during reinforce phase
    pass_attack: float = -0.1#-0.3  # was -0.5 — much stronger penalty

    border_pressure: float = -0.0   # penalty per turn based on enemy army ratio at borders
    defensive_fortify: float = +0.50  # reward for fortifying a territory adjacent to an enemy

    survival: float = +0.01
    continent_scale: float = +1.5

    win_game: float = +100.0
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
    attack_bonus_dice: int = 2   # extra die granted by reinforcements event
    supply_drop_armies: int = 2
    retreat_penalty_armies: int = 1
    combat_loss_armies: int = 1


@dataclass
class TrainingConfig:
    """DQN training hyperparameters"""
    total_steps: int = 300_000
    batch_size: int = 64 #64
    replay_capacity: int = 500_000
    learning_rate: float = 1e-4#5e-5   # back down from 2.5e-4
    gamma: float = 0.95
    tau: float = 0.01
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 500_000 #200_000
    grad_clip: float = 1.0#10.0


@dataclass
class CurriculumPhase:
    """Defines one difficulty phase of training"""
    name: str
    max_turns: int
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
            max_turns=300,
            agent_start_armies=8,
            enemy_start_armies=1,
            n_neutral_armies=1,
            random_event_prob=0.05,
        ),
        CurriculumPhase(
            name="Intermediate",
            max_turns=200,
            agent_start_armies=5,
            enemy_start_armies=2,
            n_neutral_armies=1,
            random_event_prob=0.10,
        ),
        CurriculumPhase(
            name="Hard",
            max_turns=150,
            agent_start_armies=4,
            enemy_start_armies=2,
            n_neutral_armies=2,
            random_event_prob=0.15,
        )
    ])
    # win rate threshold to advance to next phase
    advance_win_rate: float = 0.60 #0.70
    # number of recent episodes to measure win rate over
    eval_window: int = 50 #50


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