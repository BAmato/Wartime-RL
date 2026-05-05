import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import sys
import os
from env.map_config import (
    TERRITORIES, CONTINENTS, ATTACK_PAIRS,
    TERRITORY_COLORS, OWNER_TINT,
    SPRITE_W, SPRITE_H, WIN_W, WIN_H
)


class WartimeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.attack_bonus = False

        # Observation: for each territory [owner (0-2), armies (normalized)]
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(len(TERRITORIES) * 2,),
            dtype=np.float32
        )

        # Actions: index into ATTACK_PAIRS
        self.action_space = spaces.Discrete(len(ATTACK_PAIRS))

        self._screen = None
        self._clock = None
        self._font = None
        self._sprites = None

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all territories as neutral with 1 army
        self.state = {
            name: {"owner": "neutral", "armies": 1}
            for name in TERRITORIES
        }

        # Agent starts in Alaska
        self.state["Alaska"]["owner"] = "agent"
        self.state["Alaska"]["armies"] = 3

        # Enemy starts in Argentina
        self.state["Argentina"]["owner"] = "enemy"
        self.state["Argentina"]["armies"] = 3

        self.steps = 0
        self.max_steps = 200
        self.attack_bonus = False

        return self._get_obs(), {}

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps

        # Decode action into attack pair
        src, tgt = ATTACK_PAIRS[action]

        # Validate action - agent must own src and have 2+ armies
        if self.state[src]["owner"] != "agent" or self.state[src]["armies"] < 2:
            reward = -0.1  # invalid action penalty
        else:
            tgt_owner = self.state[tgt]["owner"]

            if tgt_owner == "agent":
                reward = -0.05  # attacking own territory

            elif tgt_owner == "neutral":
                # Capture neutral territory
                self.state[tgt]["owner"] = "agent"
                self.state[tgt]["armies"] = 1
                self.state[src]["armies"] -= 1
                reward = +1.0

            elif tgt_owner == "enemy":
                # Combat via dice roll
                attacker_dice = 2 + (1 if self.attack_bonus else 0)
                if self._resolve_combat(attacker_dice=attacker_dice, defender_dice=1):
                    # Agent wins
                    self.state[tgt]["owner"] = "agent"
                    self.state[tgt]["armies"] = 1
                    self.state[src]["armies"] -= 1
                    reward = +3.0
                else:
                    # Agent loses
                    self.state[src]["armies"] -= 1
                    if self.state[src]["armies"] < 1:
                        self.state[src]["armies"] = 1
                    reward = -3.0

            self.attack_bonus = False

        # Continent bonus armies
        reward += self._apply_continent_bonus()

        # Survival bonus
        reward += 0.1

        # Check win condition - enemy has no territories left
        enemy_territories = [n for n, d in self.state.items() if d["owner"] == "enemy"]
        if len(enemy_territories) == 0:
            reward += 20.0
            terminated = True

        # Enemy takes its turn
        enemy_reward = self._enemy_turn()
        reward += enemy_reward

        # Check lose condition - agent has no territories left
        agent_territories = [n for n, d in self.state.items() if d["owner"] == "agent"]
        if len(agent_territories) == 0:
            reward -= 10.0
            terminated = True

        # Random event
        event_reward, _ = self._random_event()
        reward += event_reward

        return self._get_obs(), reward, terminated, truncated, {}

    # -------------------------------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------------------------------
    def _get_obs(self):
        obs = []
        owner_map = {"neutral": 0.0, "agent": 0.5, "enemy": 1.0}
        for name in TERRITORIES:
            d = self.state[name]
            obs.append(owner_map[d["owner"]])
            obs.append(min(d["armies"] / 10.0, 1.0))
        return np.array(obs, dtype=np.float32)

    # -------------------------------------------------------------------------
    # CONTINENT BONUS
    # -------------------------------------------------------------------------
    def _apply_continent_bonus(self):
        bonus = 0.0
        for cont, data in CONTINENTS.items():
            owners = [self.state[t]["owner"] for t in data["territories"]]
            if all(o == "agent" for o in owners):
                for t in data["territories"]:
                    self.state[t]["armies"] += 1
                bonus += data["bonus_armies"] * 0.5
        return bonus

    # -------------------------------------------------------------------------
    # DICE
    # -------------------------------------------------------------------------
    def _roll_dice(self, num_dice):
        return max(self.np_random.integers(1, 7) for _ in range(num_dice))

    def _resolve_combat(self, attacker_dice=2, defender_dice=1):
        return self._roll_dice(attacker_dice) > self._roll_dice(defender_dice)

    # -------------------------------------------------------------------------
    # ENEMY AI
    # -------------------------------------------------------------------------
    def _enemy_turn(self):
        reward = 0.0

        # Find all enemy territories that can attack
        enemy_srcs = [
            name for name, d in self.state.items()
            if d["owner"] == "enemy" and d["armies"] >= 2
        ]

        if not enemy_srcs:
            # Give enemy a free army if it has none to attack with
            enemy_territories = [n for n, d in self.state.items() if d["owner"] == "enemy"]
            if enemy_territories:
                self.state[enemy_territories[0]]["armies"] += 1
            return reward

        # Enemy picks the attack that targets an agent territory if possible
        best_src, best_tgt = None, None
        for src in enemy_srcs:
            for tgt in TERRITORIES[src]["adjacent"]:
                if tgt not in TERRITORIES:
                    continue
                if self.state[tgt]["owner"] == "agent":
                    best_src, best_tgt = src, tgt
                    break
            if best_src:
                break

        # If no agent territory adjacent, attack neutral
        if not best_src:
            for src in enemy_srcs:
                for tgt in TERRITORIES[src]["adjacent"]:
                    if tgt not in TERRITORIES:
                        continue
                    if self.state[tgt]["owner"] == "neutral":
                        best_src, best_tgt = src, tgt
                        break
                if best_src:
                    break

        if best_src and best_tgt:
            if self._resolve_combat(attacker_dice=1, defender_dice=2):
                self.state[best_tgt]["owner"] = "enemy"
                self.state[best_tgt]["armies"] = 1
                self.state[best_src]["armies"] -= 1
                reward = -3.0
            else:
                self.state[best_src]["armies"] -= 1
                if self.state[best_src]["armies"] < 1:
                    self.state[best_src]["armies"] = 1

        return reward

    # -------------------------------------------------------------------------
    # RANDOM EVENTS
    # -------------------------------------------------------------------------
    def _random_event(self):
        if self.np_random.random() > 0.10:
            return 0.0, "No event"

        events = ["supply_drop", "enemy_retreat", "ambush", "reinforcements"]
        event = events[self.np_random.integers(0, len(events))]
        reward = 0.0

        if event == "supply_drop":
            agent_territories = [n for n, d in self.state.items() if d["owner"] == "agent"]
            if agent_territories:
                idx = self.np_random.integers(0, len(agent_territories))
                self.state[agent_territories[idx]]["armies"] += 2
                reward = +1.0

        elif event == "enemy_retreat":
            enemy_territories = [n for n, d in self.state.items() if d["owner"] == "enemy"]
            if enemy_territories:
                idx = self.np_random.integers(0, len(enemy_territories))
                self.state[enemy_territories[idx]]["armies"] = max(
                    1, self.state[enemy_territories[idx]]["armies"] - 1
                )

        elif event == "ambush":
            reward += self._enemy_turn()

        elif event == "reinforcements":
            self.attack_bonus = True
            reward = +0.5

        return reward, event

    # -------------------------------------------------------------------------
    # RENDER
    # -------------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption("Wartime-RL")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("Arial", 12, bold=True)
            self._load_sprites()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self._screen.fill((180, 210, 230))

        # Draw neutral sprites first
        for name in TERRITORIES:
            self._screen.blit(self._sprites[name], (0, 0))

        # Draw ownership color overlay using scaled polygons
        owner_colors = {
            "agent":   (100, 200, 100, 140),
            "enemy":   (220, 80,  80,  140),
            "neutral": None
        }

        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

        def sp(pt):
            return (int(pt[0] * WIN_W / SPRITE_W),
                    int(pt[1] * WIN_H / SPRITE_H))

        for name, data in self.state.items():
            color = owner_colors[data["owner"]]
            if color is not None:
                scaled_poly = [sp(p) for p in TERRITORIES[name]["polygon"]]
                pygame.draw.polygon(overlay, color, scaled_poly)

        self._screen.blit(overlay, (0, 0))

        # Draw labels and army counts
        for name, data in self.state.items():
            cx, cy = sp(TERRITORIES[name]["center"])
            label = self._font.render(name, True, (20, 20, 20))
            armies = self._font.render(f"[{data['armies']}]", True, (20, 20, 20))
            self._screen.blit(label, (cx - label.get_width()//2, cy - 10))
            self._screen.blit(armies, (cx - armies.get_width()//2, cy + 5))

        step_text = self._font.render(f"Step: {self.steps}", True, (0, 0, 0))
        self._screen.blit(step_text, (10, 10))

        pygame.display.flip()
        self._clock.tick(10)

    def _load_sprites(self):
        """Load sprites as neutral only, ownership shown via polygon overlay."""
        self._sprites = {}
        for name in TERRITORIES:
            filename = name.lower().replace(" ", "_") + ".png"
            path = os.path.join("assets", filename)
            raw = pygame.image.load(path).convert_alpha()
            scaled = pygame.transform.scale(raw, (WIN_W, WIN_H))
            self._sprites[name] = scaled

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None