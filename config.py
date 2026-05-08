"""
config.py
All reward weights, training hyperparameters, and 
curriculum settings live here. They can be changed here in on place instead of hardcoding 
agent code

Author: Damian Villarreal
"""

from dataclasses import dataclass, field

@dataclass
class RewardConfig:
    """All scalar reward components used in wartime_env.step()
    Tweak these instead of inside env
    """
    #invalid/ wasteful actions
    invalid_action: float= -0.5
    friendly_fire: float= -0.5

    #territory actions
    capture_neutral: float= +1.0
    win_combat: float= +3.0
    lose_combat: float= -3.0

    #per-step signals
    survival: float= +0.0001
    continent_scale: float= +0.5

    #terminal outcomes
    win_game: float= +20.0
    lose_game: float= -40.0

    #random events
    supply_drop: float= +1.0
    reinforcements: float= +0.5

@dataclass
class TrainingConfig:
    """DQN training hyperparameters"""
    total_steps: int=200_000
    batch_size: int=64
    replay_capacity: int=50_000
    learning_rate: float=1e-4
    gamma: float=0.99
    tau: float=0.005
    eps_start: float=1.0
    eps_end: float=0.05
    eps_decay: int=50_000
    grad_clip: float=10.0

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
    """Ordered list of curriculum phases
        Pass Curriculum_leve=0/1/2 into wartime_env to select phase
        Advance phases based on win rate over a rolloing window of episodes
    """
    phase: list= field(default_factory=lambda:[
        CurriculumPhase(
            name="Beginner",
            max_steps=100,
            agent_start_armies=5,
            enemy_start_armies=2,
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
    #win rate threashold to advance to next phase
    advance_win_rate: float=0.70
    #number of recent episodes to measure win rate over
    eval_window: int=50

class CurriculumTracker:
    """Tracks episode outcomes and decides when to advance the curriculum
    Used in training loop
    """
    def __init__(self, env, cfg:CurriculumConfig=None):
        self.env=env
        self.cfg=cfg or CurriculumConfig()
        self._history:list[str]=[]
    
    def record(self, outcome:str)->bool:
        """Record an episode outcome and advance the phase if ready"""
        self._history.append(outcome)
        if len(self._history)<self.cfg.eval_window:
            return False
        # win_rate=self._history.count("win") / len(self._history)
        win_rate=self.win_rate()
        max_level=len(self.cfg.phase)-1

        if win_rate>=self.cfg.advance_win_rate and self.env.curriculum_level < max_level:
            self.env.curriculum_level+=1
            self._history.clear()
            print(f"[Curriculum] Win rate {win_rate:.0%} >= {self.cfg.advance_win_rate:.0%} "
                  f"— advanced to level {self.env.curriculum_level} "
                  f"({self.cfg.phase[self.env.curriculum_level].name})")
            return True
        return False
    
    def win_rate(self)->float:
        """Current Win Rate over the last eval_window episodes"""
        if not self._history:
            return 0.0
        return self._history.count("win") / len(self._history)
    
    def __repr__(self)->str:
        phase=self.cfg.phase[self.env.curriculum_level]
        return (f"CurriculumTracker(level={self.env.curriculum_level}, "
                f"phase={phase.name}, "
                f"win_rate={self.win_rate:.0%}, "
                f"episodes={len(self._history)}/{self.cfg.eval_window})")


