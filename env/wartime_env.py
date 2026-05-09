import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import os
from env.map_config import (
    TERRITORIES, CONTINENTS, ATTACK_PAIRS,
    SPRITE_W, SPRITE_H, WIN_W, WIN_H, OWNER_TINT
)
from config import RewardConfig, CurriculumConfig, GameplayConfig


class WartimeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    PHASE_REINFORCE = "reinforce"
    PHASE_ATTACK = "attack"
    PHASE_FORTIFY = "fortify"
    HUD_W = 340

    def __init__(
        self,
        render_mode=None,
        reward_cfg: RewardConfig=None,
        curriculum_cfg: CurriculumConfig=None,
        gameplay_cfg: GameplayConfig=None,
        curriculum_level: int=0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.attack_bonus = False
        self.cfg = reward_cfg or RewardConfig()
        self.curriculum = curriculum_cfg or CurriculumConfig()
        self.gameplay = gameplay_cfg or GameplayConfig()

        self.curriculum_level = max(
            0,
            min(curriculum_level, len(self.curriculum.phase) - 1),
        )

        self.deploy_action_count = len(TERRITORIES)
        self.attack_action_offset = self.deploy_action_count
        self.pass_action = self.deploy_action_count + len(ATTACK_PAIRS)

        # Observation: for each territory [owner (0-2), armies (normalized)],
        # followed by current phase and pending reinforcement count.
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(len(TERRITORIES) * 2 + 2,),
            dtype=np.float32
        )

        # Actions: deploy to each territory, attack/fortify along each edge, or pass.
        self.action_space = spaces.Discrete(self.pass_action + 1)

        self._screen = None
        self._clock = None
        self._font = None
        self._sprites = None

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        phase = self.curriculum.phase[self.curriculum_level]

        self.state = {
            name: {"owner": "neutral", "armies": phase.n_neutral_armies}
            for name in TERRITORIES
        }

        self.state["Alaska"]["owner"] = "agent"
        self.state["Alaska"]["armies"] = phase.agent_start_armies

        self.state["Argentina"]["owner"] = "enemy"
        self.state["Argentina"]["armies"] = phase.enemy_start_armies

        self.steps = 0
        self.max_steps = phase.max_turns
        self.attack_bonus = False
        self.attacked_this_turn = False
        self.turn_phase = self.PHASE_REINFORCE
        self.pending_reinforcements = self._reinforcement_count("agent")

        return self._get_obs(), {}

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps

        event = "No event"
        action_type = "none"
        combat_result = "none"
        border_pressure_val = 0.0
        agent_reinforcements = 0
        enemy_reinforcements = 0
        fortify_source = None
        fortify_target = None

        if self.turn_phase == self.PHASE_REINFORCE:
            agent_reinforcements = self._handle_deploy_action(action)
            if agent_reinforcements:
                action_type = "deploy"

        elif self.turn_phase == self.PHASE_ATTACK:
            action_type, action_reward, combat_result, turn_ended = self._handle_attack_action(action)
            reward += action_reward

            if turn_ended:
                self.attack_bonus = False
                enemy_territories = self._owned_territories("enemy")
                if len(enemy_territories) == 0:
                    reward += self.cfg.win_game
                    terminated = True

                if not terminated:
                    if self._agent_has_fortify_targets():
                        self.turn_phase = self.PHASE_FORTIFY
                    else:
                        end_r, terminated, event, enemy_reinforcements, border_pressure_val = (
                            self._end_player_turn()
                        )
                        reward += end_r

        elif self.turn_phase == self.PHASE_FORTIFY:
            action_type, action_reward, turn_ended, fortify_source, fortify_target = (
                self._handle_fortify_action(action)
            )
            reward += action_reward

            if turn_ended:
                end_r, terminated, event, enemy_reinforcements, border_pressure_val = (
                    self._end_player_turn()
                )
                reward += end_r

        reward += self.cfg.survival

        agent_territories = self._owned_territories("agent")
        enemy_territories = self._owned_territories("enemy")
        phase = self.curriculum.phase[self.curriculum_level]

        info = {
            "step": self.steps,
            "event": event,
            "action_type": action_type,
            "turn_phase": self.turn_phase,
            "combat_result": combat_result,
            "border_pressure": border_pressure_val,
            "pending_reinforcements": self.pending_reinforcements,
            "agent_territories": len(agent_territories),
            "enemy_territories": len(enemy_territories),
            "agent_reinforcements": agent_reinforcements,
            "enemy_reinforcements": enemy_reinforcements,
            "attack_bonus_active": self.attack_bonus,
            "fortify_source": fortify_source,
            "fortify_target": fortify_target,
            "curriculum_level": self.curriculum_level,
            "curriculum_phase": phase.name,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------------------------------
    def _get_obs(self):
        obs = []
        owner_map = {"neutral": 0.0, "agent": 0.5, "enemy": 1.0}
        for name in TERRITORIES:
            d = self.state[name]
            obs.append(owner_map[d["owner"]])
            obs.append(min(d["armies"] / self.gameplay.max_armies_per_territory, 1.0))
        phase_enc = {
            self.PHASE_REINFORCE: 0.0,
            self.PHASE_ATTACK: 0.5,
            self.PHASE_FORTIFY: 1.0,
        }
        obs.append(phase_enc[self.turn_phase])
        obs.append(min(
            self.pending_reinforcements / self.gameplay.max_pending_reinforcements,
            1.0,
        ))
        return np.array(obs, dtype=np.float32)

    # -------------------------------------------------------------------------
    # ACTION HELPERS
    # -------------------------------------------------------------------------
    def valid_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.turn_phase == self.PHASE_REINFORCE:
            for idx, name in enumerate(TERRITORIES):
                if self._can_deploy_to(name):
                    mask[idx] = True

        elif self.turn_phase == self.PHASE_ATTACK:
            for idx, (src, tgt) in enumerate(ATTACK_PAIRS):
                action = self.attack_action_offset + idx
                if self._can_attack(src, tgt):
                    mask[action] = True
            mask[self.pass_action] = True

        elif self.turn_phase == self.PHASE_FORTIFY:
            for idx, (src, tgt) in enumerate(ATTACK_PAIRS):
                action = self.attack_action_offset + idx
                if self._can_fortify(src, tgt):
                    mask[action] = True
            mask[self.pass_action] = True

        return mask

    def sample_valid_action(self):
        valid_actions = np.flatnonzero(self.valid_action_mask())
        if len(valid_actions) == 0:
            return self.pass_action
        idx = self.np_random.integers(0, len(valid_actions))
        return int(valid_actions[idx])

    def describe_action(self, action):
        """Describe the action in context of the current phase. Call BEFORE step()."""
        if 0 <= action < self.deploy_action_count:
            territory = list(TERRITORIES)[action]
            return f"deploy:{territory}"
        if self.attack_action_offset <= action < self.pass_action:
            src, tgt = ATTACK_PAIRS[action - self.attack_action_offset]
            verb = "fortify" if self.turn_phase == self.PHASE_FORTIFY else "attack"
            return f"{verb}:{src}->{tgt}"
        if action == self.pass_action:
            return "pass"
        return "unknown"

    # -------------------------------------------------------------------------
    # TURN END SEQUENCE
    # -------------------------------------------------------------------------
    def _end_player_turn(self):
        """Run enemy turn, random events, and start new agent turn.
        Returns (reward, terminated, event, enemy_reinforcements, border_pressure_penalty)."""
        reward = 0.0
        terminated = False
        event = "No event"

        enemy_reinforcements = self._assign_enemy_reinforcements()
        reward += self._enemy_turn()

        agent_territories = self._owned_territories("agent")
        if len(agent_territories) == 0:
            reward += self.cfg.lose_game
            terminated = True

        if not terminated:
            event_reward, event = self._random_event()
            reward += event_reward
            self._start_agent_turn()

        border_pressure_penalty = self._border_pressure_penalty()
        reward += border_pressure_penalty

        return reward, terminated, event, enemy_reinforcements, border_pressure_penalty
    
    # -------------------------------------------------------------------------
    # REINFORCEMENTS
    # -------------------------------------------------------------------------
    def _assign_enemy_reinforcements(self):
        owned = self._owned_territories("enemy")
        if not owned:
            return 0

        reinforcements = self._reinforcement_count("enemy")
        placed = 0
        for _ in range(reinforcements):
            target = self._choose_enemy_reinforcement_target()
            if target is None:
                break
            self.state[target]["armies"] = min(
                self.state[target]["armies"] + 1,
                self.gameplay.max_armies_per_territory,
            )
            placed += 1
        return placed

    def _handle_deploy_action(self, action):
        if action < 0 or action >= self.deploy_action_count:
            return 0

        territory = list(TERRITORIES)[action]
        if self.pending_reinforcements <= 0:
            self.turn_phase = self.PHASE_ATTACK
            return 0
        if not self._can_deploy_to(territory):
            return 0

        self.state[territory]["armies"] += 1
        self.pending_reinforcements -= 1
        if self.pending_reinforcements == 0 or not self._agent_has_deploy_targets():
            self.turn_phase = self.PHASE_ATTACK
            self.pending_reinforcements = 0
        return 1

    def _handle_attack_action(self, action):
        if action == self.pass_action:
            penalty = 0.0 if self.attacked_this_turn else self.cfg.pass_attack
            return "pass", penalty, "none", True

        attack_idx = action - self.attack_action_offset
        src, tgt = ATTACK_PAIRS[attack_idx]
        reward, combat_result = self._resolve_agent_attack(src, tgt)
        self.attacked_this_turn = True
        return "attack", reward, combat_result, True

    def _handle_fortify_action(self, action):
        """Returns (action_type, reward, turn_ended, src, tgt)."""
        if action == self.pass_action:
            return "pass", 0.0, True, None, None

        attack_idx = action - self.attack_action_offset
        if 0 <= attack_idx < len(ATTACK_PAIRS):
            src, tgt = ATTACK_PAIRS[attack_idx]
            if self._can_fortify(src, tgt):
                reward = self._resolve_fortify(src, tgt)
                return "fortify", reward, True, src, tgt

        return "pass", 0.0, True, None, None

    def _resolve_agent_attack(self, src, tgt):
        src_armies = self.state[src]["armies"]
        tgt_owner = self.state[tgt]["owner"]

        attacker_dice = self._attacker_dice_count(src_armies)
        if self.attack_bonus:
            attacker_dice = min(attacker_dice + self.gameplay.attack_bonus_dice, 3)

        if tgt_owner == "neutral":
            self.state[tgt]["owner"] = "agent"
            moved = self._move_armies_after_capture(src, attacker_dice)
            self.state[tgt]["armies"] = moved
            return self.cfg.capture_neutral, "none"

        attacker_losses, defender_losses = self._resolve_combat(attacker_dice, src, tgt)
        self._remove_armies(src, attacker_losses)

        remaining_defender = self.state[tgt]["armies"] - defender_losses
        if remaining_defender <= 0:
            self.state[tgt]["armies"] = 0
            self.state[tgt]["owner"] = "agent"
            moved = self._move_armies_after_capture(src, attacker_dice)
            self.state[tgt]["armies"] = moved
            return self.cfg.win_combat, "win"

        self.state[tgt]["armies"] = remaining_defender
        return self.cfg.lose_combat, "loss"

    def _resolve_fortify(self, src, tgt):
        available = self.state[src]["armies"] - 1
        cap = self.gameplay.max_armies_per_territory - self.state[tgt]["armies"]
        moved = max(0, min(available, cap))
        self.state[src]["armies"] -= moved
        self.state[tgt]["armies"] += moved
        tgt_borders_enemy = any(
            self.state.get(adj, {}).get("owner") == "enemy"
            for adj in TERRITORIES[tgt]["adjacent"]
            if adj in TERRITORIES
        )
        return self.cfg.defensive_fortify if tgt_borders_enemy else 0.0

    def _border_pressure_penalty(self):
        """Penalty for agent border territories where enemy armies outnumber by >1.5x."""
        total_penalty = 0.0
        for name, data in self.state.items():
            if data["owner"] != "agent":
                continue
            enemy_adjacent = sum(
                self.state[adj]["armies"]
                for adj in TERRITORIES[name]["adjacent"]
                if adj in TERRITORIES and self.state[adj]["owner"] == "enemy"
            )
            if enemy_adjacent == 0:
                continue
            agent_armies = max(data["armies"], 1)  # floor at 1 to avoid division by zero
            ratio = enemy_adjacent / agent_armies
            if ratio > 1.5:
                total_penalty += self.cfg.border_pressure * ratio
        return total_penalty

    def _start_agent_turn(self):
        self.turn_phase = self.PHASE_REINFORCE
        self.pending_reinforcements = self._reinforcement_count("agent")
        self.attacked_this_turn = False
        if self.pending_reinforcements == 0 or not self._agent_has_deploy_targets():
            self.turn_phase = self.PHASE_ATTACK
            self.pending_reinforcements = 0


    def _can_deploy_to(self, territory):
        return (
            self.pending_reinforcements > 0
            and self.state[territory]["owner"] == "agent"
            and self.state[territory]["armies"] < self.gameplay.max_armies_per_territory
        )

    def _can_attack(self, src, tgt):
        return (
            self.state[src]["owner"] == "agent"
            and self.state[src]["armies"] >= 2
            and self.state[tgt]["owner"] != "agent"
        )

    def _can_fortify(self, src, tgt):
        return (
            self.state[src]["owner"] == "agent"
            and self.state[tgt]["owner"] == "agent"
            and self.state[src]["armies"] >= 2
        )

    def _agent_has_deploy_targets(self):
        return any(self._can_deploy_to(name) for name in TERRITORIES)

    def _agent_has_fortify_targets(self):
        return any(self._can_fortify(src, tgt) for src, tgt in ATTACK_PAIRS)

    def _reinforcement_count(self, owner):
        owned = self._owned_territories(owner)
        base = max(
            self.gameplay.min_reinforcements,
            len(owned) // self.gameplay.territories_per_reinforcement,
        )
        return base + self._continent_bonus_armies(owner)

    def _continent_bonus_armies(self, owner):
        bonus = 0
        for data in CONTINENTS.values():
            owners = [self.state[t]["owner"] for t in data["territories"]]
            if all(o == owner for o in owners):
                bonus += data["bonus_armies"]
        return bonus

    def _choose_enemy_reinforcement_target(self):
        targets = self._reinforcement_targets("enemy")
        if not targets:
            return None

        scores = [
            (self._enemy_reinforcement_score(name), name)
            for name in targets
        ]
        best_score = max(score for score, _ in scores)
        candidates = [name for score, name in scores if score == best_score]
        idx = self.np_random.integers(0, len(candidates))
        return candidates[idx]

    def _reinforcement_targets(self, owner):
        return [
            name for name in self._owned_territories(owner)
            if self.state[name]["armies"] < self.gameplay.max_armies_per_territory
        ]

    def _enemy_reinforcement_score(self, name):
        enemy_pressure = 0
        neutral_options = 0
        for adjacent in TERRITORIES[name]["adjacent"]:
            if adjacent not in TERRITORIES:
                continue
            adjacent_owner = self.state[adjacent]["owner"]
            if adjacent_owner == "agent":
                enemy_pressure += self.state[adjacent]["armies"]
            elif adjacent_owner == "neutral":
                neutral_options += 1

        return (enemy_pressure * 10) + (neutral_options * 3) - self.state[name]["armies"]

    def _owned_territories(self, owner):
        return [name for name, data in self.state.items() if data["owner"] == owner]

    def _move_armies_after_capture(self, src, min_to_move=1):
        """Move armies into a newly captured territory. Leaves at least 1 in src."""
        available = self.state[src]["armies"] - 1
        if available <= 0:
            return 0
        moved = min(max(min_to_move, 1), available)
        self.state[src]["armies"] -= moved
        return moved

    def _remove_armies(self, territory, amount):
        self.state[territory]["armies"] = max(
            1,
            self.state[territory]["armies"] - amount,
        )

    # -------------------------------------------------------------------------
    # DICE
    # -------------------------------------------------------------------------
    def _attacker_dice_count(self, src_armies):
        """Official Risk attacker dice: needs 4+ for 3, 3+ for 2, 2+ for 1."""
        if src_armies >= 4:
            return 3
        if src_armies >= 3:
            return 2
        return 1

    def _resolve_combat(self, attacker_dice, src, tgt):
        """Proper Risk dice: roll both sides, sort descending, compare pairs.
        Returns (attacker_losses, defender_losses)."""
        defender_dice = min(2, self.state[tgt]["armies"])

        atk = sorted(
            [int(self.np_random.integers(1, 7)) for _ in range(attacker_dice)],
            reverse=True,
        )
        dfn = sorted(
            [int(self.np_random.integers(1, 7)) for _ in range(defender_dice)],
            reverse=True,
        )

        attacker_losses = 0
        defender_losses = 0
        for a, d in zip(atk, dfn):
            if a > d:
                defender_losses += 1
            else:
                attacker_losses += 1
        return attacker_losses, defender_losses

    # -------------------------------------------------------------------------
    # ENEMY AI
    # -------------------------------------------------------------------------
    def _enemy_turn(self):
        reward = 0.0

        enemy_srcs = [
            name for name, d in self.state.items()
            if d["owner"] == "enemy" and d["armies"] >= 2
        ]

        if not enemy_srcs:
            enemy_territories = [n for n, d in self.state.items() if d["owner"] == "enemy"]
            if enemy_territories:
                target = enemy_territories[0]
                self.state[target]["armies"] = min(
                    self.state[target]["armies"] + 1,
                    self.gameplay.max_armies_per_territory,
                )
            return reward

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
            src_armies = self.state[best_src]["armies"]
            enemy_attacker_dice = self._attacker_dice_count(src_armies)

            attacker_losses, defender_losses = self._resolve_combat(
                enemy_attacker_dice, best_src, best_tgt
            )
            self._remove_armies(best_src, attacker_losses)

            remaining_defender = self.state[best_tgt]["armies"] - defender_losses
            if remaining_defender <= 0:
                self.state[best_tgt]["owner"] = "enemy"
                self.state[best_tgt]["armies"] = 0
                moved = self._move_armies_after_capture(best_src, enemy_attacker_dice)
                self.state[best_tgt]["armies"] = moved
                reward = self.cfg.lose_combat
            else:
                self.state[best_tgt]["armies"] = remaining_defender

        return reward

    # -------------------------------------------------------------------------
    # RANDOM EVENTS
    # -------------------------------------------------------------------------
    def _random_event(self):
        phase = self.curriculum.phase[self.curriculum_level]
        if self.np_random.random() > phase.random_event_prob:
            return 0.0, "No event"

        events = ["supply_drop", "enemy_retreat", "ambush", "reinforcements"]
        event = events[self.np_random.integers(0, len(events))]
        reward = 0.0

        if event == "supply_drop":
            agent_territories = [n for n, d in self.state.items() if d["owner"] == "agent"]
            if agent_territories:
                idx = self.np_random.integers(0, len(agent_territories))
                target = agent_territories[idx]
                self.state[target]["armies"] = min(
                    self.state[target]["armies"] + self.gameplay.supply_drop_armies,
                    self.gameplay.max_armies_per_territory,
                )
                reward = self.cfg.supply_drop

        elif event == "enemy_retreat":
            enemy_territories = [n for n, d in self.state.items() if d["owner"] == "enemy"]
            if enemy_territories:
                idx = self.np_random.integers(0, len(enemy_territories))
                self._remove_armies(
                    enemy_territories[idx],
                    self.gameplay.retreat_penalty_armies,
                )

        elif event == "ambush":
            reward += self._enemy_turn()

        elif event == "reinforcements":
            self.attack_bonus = True
            reward = self.cfg.reinforcements

        return reward, event

    # -------------------------------------------------------------------------
    # RENDER
    # -------------------------------------------------------------------------
    def render(self, hud=None):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((WIN_W + self.HUD_W, WIN_H))
            pygame.display.set_caption("Wartime-RL")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("Arial", 12, bold=True)
            self._hud_font = pygame.font.SysFont("Arial", 14)
            self._hud_font_bold = pygame.font.SysFont("Arial", 15, bold=True)
            self._load_sprites()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._screen = None
                return

        self._screen.fill((180, 210, 230))

        for name in TERRITORIES:
            self._screen.blit(self._sprites[name], (0, 0))

        owner_colors = {
            "agent":   OWNER_TINT["agent"],
            "enemy":   OWNER_TINT["enemy"],
            "neutral": OWNER_TINT["neutral"]
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
        self._draw_action_highlights(hud or {}, sp)

        for name, data in self.state.items():
            cx, cy = sp(TERRITORIES[name]["center"])
            label = self._font.render(name, True, (20, 20, 20))
            armies = self._font.render(f"[{data['armies']}]", True, (20, 20, 20))
            rect_w = max(label.get_width(), armies.get_width()) + 8
            rect_h = label.get_height() + armies.get_height() + 8
            rect = pygame.Rect(cx - rect_w // 2, cy - 14, rect_w, rect_h)
            label_bg = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA)
            label_bg.fill((245, 248, 242, 190))
            self._screen.blit(label_bg, rect)
            pygame.draw.rect(self._screen, (65, 75, 65), rect, 1, border_radius=3)
            self._screen.blit(label, (cx - label.get_width() // 2, cy - 10))
            self._screen.blit(armies, (cx - armies.get_width() // 2, cy + 5))

        self._draw_hud(hud or {})
        if hud and hud.get("episode_overlay"):
            self._draw_episode_overlay(hud["episode_overlay"])

        pygame.display.flip()
        self._clock.tick(10)

        if self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self._screen), axes=(1, 0, 2)
            )

    def _draw_action_highlights(self, hud, sp):
        active_deploy = hud.get("active_deploy")
        active_source = hud.get("active_source")
        active_target = hud.get("active_target")
        active_fortify_src = hud.get("active_fortify_src")
        active_fortify_tgt = hud.get("active_fortify_tgt")

        def draw_poly(name, color, width):
            if name not in TERRITORIES:
                return
            points = [sp(p) for p in TERRITORIES[name]["polygon"]]
            pygame.draw.polygon(self._screen, color, points, width)

        if active_deploy:
            draw_poly(active_deploy, (80, 230, 170), 5)
        if active_source:
            draw_poly(active_source, (255, 210, 90), 5)
        if active_target:
            draw_poly(active_target, (255, 95, 95), 5)
        if active_fortify_src:
            draw_poly(active_fortify_src, (90, 160, 255), 5)
        if active_fortify_tgt:
            draw_poly(active_fortify_tgt, (90, 220, 255), 5)

        if active_source in TERRITORIES and active_target in TERRITORIES:
            source_center = sp(TERRITORIES[active_source]["center"])
            target_center = sp(TERRITORIES[active_target]["center"])
            pygame.draw.line(
                self._screen, (255, 235, 125), source_center, target_center, 4,
            )
            pygame.draw.circle(self._screen, (255, 235, 125), source_center, 7)
            pygame.draw.circle(self._screen, (255, 95, 95), target_center, 7)

        if active_fortify_src in TERRITORIES and active_fortify_tgt in TERRITORIES:
            src_center = sp(TERRITORIES[active_fortify_src]["center"])
            tgt_center = sp(TERRITORIES[active_fortify_tgt]["center"])
            pygame.draw.line(
                self._screen, (120, 180, 255), src_center, tgt_center, 4,
            )
            pygame.draw.circle(self._screen, (90, 160, 255), src_center, 7)
            pygame.draw.circle(self._screen, (90, 220, 255), tgt_center, 7)

    def _draw_hud(self, hud):
        panel_x = WIN_W
        padding = 14
        content_w = self.HUD_W - (padding * 2)
        y = padding
        pygame.draw.rect(self._screen, (26, 31, 38), (panel_x, 0, self.HUD_W, WIN_H))
        pygame.draw.line(self._screen, (85, 96, 108), (panel_x, 0), (panel_x, WIN_H), 2)

        def fit_text(text, font, max_width):
            text = str(text)
            if font.size(text)[0] <= max_width:
                return text
            ellipsis = "..."
            while text and font.size(text + ellipsis)[0] > max_width:
                text = text[:-1]
            return text + ellipsis if text else ellipsis

        def draw_text(text, x, y_pos, color=(232, 236, 240), font=None, max_width=None):
            font = font or self._hud_font
            if max_width is not None:
                text = fit_text(text, font, max_width)
            rendered = font.render(str(text), True, color)
            self._screen.blit(rendered, (x, y_pos))
            return y_pos + rendered.get_height() + 3

        def draw_section(title, y_pos):
            y_pos += 5
            y_pos = draw_text(title.upper(), panel_x + padding, y_pos, (158, 176, 196), self._hud_font_bold)
            pygame.draw.line(
                self._screen,
                (65, 75, 86),
                (panel_x + padding, y_pos),
                (panel_x + self.HUD_W - padding, y_pos),
                1,
            )
            return y_pos + 6

        def metric(label, value, y_pos, color=(232, 236, 240), bg=None):
            if bg is not None:
                pygame.draw.rect(
                    self._screen,
                    bg,
                    (panel_x + padding - 4, y_pos - 2, content_w + 8, 20),
                    border_radius=4,
                )
            return draw_text(
                f"{label}: {value}",
                panel_x + padding,
                y_pos,
                color,
                max_width=content_w,
            )

        def draw_badge(text, x, y_pos, active, color, width=88):
            bg = color if active else (54, 62, 72)
            fg = (20, 24, 28) if active else (190, 200, 210)
            rect = pygame.Rect(x, y_pos, width, 24)
            pygame.draw.rect(self._screen, bg, rect, border_radius=4)
            pygame.draw.rect(self._screen, (90, 100, 112), rect, 1, border_radius=4)
            rendered = self._hud_font_bold.render(text, True, fg)
            self._screen.blit(
                rendered,
                (x + (rect.width - rendered.get_width()) // 2, y_pos + 4),
            )

        info = hud.get("info") or {}
        phase = info.get("turn_phase", self.turn_phase)

        phase_color_map = {
            self.PHASE_REINFORCE: (93, 195, 151),
            self.PHASE_ATTACK:    (237, 151, 91),
            self.PHASE_FORTIFY:   (100, 160, 240),
        }
        phase_color = phase_color_map.get(phase, (232, 236, 240))

        y = draw_text("WarTime-RL", panel_x + padding, y, (255, 255, 255), self._hud_font_bold)
        y = draw_text("Mechanics HUD", panel_x + padding, y, (184, 194, 204))

        y = draw_section("Turn", y)

        # Three phase badges: REINFORCE -> ATTACK -> FORTIFY
        badge_w = 88
        arrow_gap = 12
        x0 = panel_x + padding
        x1 = x0 + badge_w + arrow_gap
        x2 = x1 + badge_w + arrow_gap

        draw_badge("REINFORCE", x0, y, phase == self.PHASE_REINFORCE, (93, 195, 151), badge_w)
        arrow = self._hud_font_bold.render("->", True, (158, 176, 196))
        self._screen.blit(arrow, (x0 + badge_w + 1, y + 4))

        draw_badge("ATTACK", x1, y, phase == self.PHASE_ATTACK, (237, 151, 91), badge_w)
        self._screen.blit(arrow, (x1 + badge_w + 1, y + 4))

        draw_badge("FORTIFY", x2, y, phase == self.PHASE_FORTIFY, (100, 160, 240), badge_w)
        y += 30

        y = metric("Phase", phase, y, phase_color)
        y = metric("Pending", info.get("pending_reinforcements", self.pending_reinforcements), y)
        y = metric("Step", info.get("step", self.steps), y)
        y = metric("Agent terr", info.get("agent_territories", len(self._owned_territories("agent"))), y)
        y = metric("Enemy terr", info.get("enemy_territories", len(self._owned_territories("enemy"))), y)

        y = draw_section("Action", y)
        y = metric("Last", hud.get("action_label", "none"), y)
        y = metric("Type", info.get("action_type", "none"), y)
        for entry in hud.get("action_log", [])[-4:]:
            y = draw_text(entry, panel_x + padding, y, (205, 214, 224), max_width=content_w)

        y = draw_section("Reward", y)
        y = metric("Step reward", f"{hud.get('step_reward', 0.0):+.2f}", y)
        y = metric("Episode total", f"{hud.get('episode_reward', 0.0):+.2f}", y)

        y = draw_section("Events", y)
        event_name = info.get("event", "No event")
        event_bg = (92, 72, 34) if event_name != "No event" else None
        event_color = (255, 230, 160) if event_name != "No event" else (232, 236, 240)
        y = metric("Event", event_name, y, event_color, event_bg)
        y = metric("Agent reinf", info.get("agent_reinforcements", 0), y)
        y = metric("Enemy reinf", info.get("enemy_reinforcements", 0), y)
        y = metric("Attack bonus", info.get("attack_bonus_active", self.attack_bonus), y)
        y = metric("Speed", f"{hud.get('speed_label', 'Medium')} ({hud.get('render_fps', 10)} fps)", y)

        y = draw_section("Episodes", y)
        completed = hud.get("completed_episodes", 0)
        wins = hud.get("wins", 0)
        losses = hud.get("losses", 0)
        timeouts = hud.get("timeouts", 0)
        avg_reward = hud.get("avg_reward", 0.0)
        win_rate = wins / completed if completed else 0.0
        y = metric("Current", hud.get("episode", 1), y)
        y = metric("Completed", completed, y)
        y = metric("W/L/T", f"{wins}/{losses}/{timeouts}", y)
        y = metric("Win rate", f"{win_rate:.1%}", y)
        y = metric("Avg reward", f"{avg_reward:+.2f}", y)

        y = max(y + 10, WIN_H - 48)
        draw_text("1 Slow | 2 Med | 3 Fast", panel_x + padding, y, (158, 176, 196), max_width=content_w)
        draw_text("Space pause | Q/Esc quit", panel_x + padding, y + 18, (158, 176, 196), max_width=content_w)

    def _draw_episode_overlay(self, overlay):
        surface = pygame.Surface((WIN_W + self.HUD_W, WIN_H), pygame.SRCALPHA)
        surface.fill((8, 12, 18, 170))
        self._screen.blit(surface, (0, 0))

        card_w, card_h = 440, 220
        card = pygame.Rect(
            ((WIN_W + self.HUD_W) - card_w) // 2,
            (WIN_H - card_h) // 2,
            card_w, card_h,
        )
        pygame.draw.rect(self._screen, (245, 248, 250), card, border_radius=8)
        pygame.draw.rect(self._screen, (45, 55, 66), card, 2, border_radius=8)

        outcome = overlay.get("outcome", "EPISODE END")
        outcome_color = {
            "WIN": (52, 145, 95),
            "LOSS": (190, 70, 70),
            "TIMEOUT": (194, 135, 45),
        }.get(outcome, (40, 48, 56))
        title = pygame.font.SysFont("Arial", 28, bold=True).render(outcome, True, outcome_color)
        self._screen.blit(title, (card.centerx - title.get_width() // 2, card.y + 24))

        lines = [
            f"Steps: {overlay.get('steps', 0)}",
            f"Episode reward: {overlay.get('reward', 0.0):+.2f}",
            f"Agent territories: {overlay.get('agent_territories', 0)}",
            f"Enemy territories: {overlay.get('enemy_territories', 0)}",
        ]
        y = card.y + 76
        for line in lines:
            rendered = self._hud_font.render(line, True, (30, 36, 42))
            self._screen.blit(rendered, (card.x + 54, y))
            y += 28

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