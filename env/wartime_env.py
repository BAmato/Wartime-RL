import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pygame
import sys

# Tile type constants
EMPTY = 0
FRIENDLY = 1
ENEMY = 2
RESOURCE = 3
OBSTACLE = 4
BASE = 5

class WartimeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, grid_size=8, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Observation: flattened grid
        self.observation_space = spaces.Box(
            low=0, high=5,
            shape=(grid_size * grid_size,),
            dtype=np.float32
        )

        # Actions: 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Direction mappings
        self.directions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Place friendly base in top-left
        self.agent_pos = [0, 0]
        self.grid[0][0] = BASE
        
        # Place enemy base in bottom-right
        self.enemy_pos = [self.grid_size-1, self.grid_size-1]
        self.grid[self.grid_size-1][self.grid_size-1] = ENEMY
        
        # Place some resource tiles
        self.grid[2][2] = RESOURCE
        self.grid[4][4] = RESOURCE
        self.grid[2][5] = RESOURCE
        
        # Place obstacle tiles
        obstacles = [
            (1, 3), (2, 3), (3, 3),  # vertical wall
            (5, 2), (5, 3), (5, 4),  # horizontal wall
        ]
        for r, c in obstacles:
            self.grid[r][c] = OBSTACLE

        self.steps = 0
        self.max_steps = 200

        return self._get_obs(), {}

    def step(self, action):
            self.steps += 1
            reward = 0.0
            terminated = False
            truncated = self.steps >= self.max_steps

            # Calculate new position
            dr, dc = self.directions[action]
            new_r = self.agent_pos[0] + dr
            new_c = self.agent_pos[1] + dc

            # Check bounds
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                tile = self.grid[new_r][new_c]

                if tile == OBSTACLE:
                    reward = -0.1  # bumped into wall

                elif tile == RESOURCE:
                    reward = +2.0  # captured resource tile
                    self.grid[new_r][new_c] = FRIENDLY
                    self.agent_pos = [new_r, new_c]

                elif tile == ENEMY:
                # Stochastic combat via dice roll
                    if self._resolve_combat(attacker_dice=2, defender_dice=1):
                        # Agent wins combat
                        reward = +3.0
                        self.grid[new_r][new_c] = FRIENDLY
                        self.agent_pos = [new_r, new_c]
                        if [new_r, new_c] == self.enemy_pos:
                            reward += 20.0
                            terminated = True
                    else:
                        # Agent loses combat - pushed back, loses current tile
                        reward = -3.0
                        self.grid[self.agent_pos[0]][self.agent_pos[1]] = EMPTY

                elif tile == EMPTY:
                    reward = +1.0  # captured neutral territory
                    self.grid[new_r][new_c] = FRIENDLY
                    self.agent_pos = [new_r, new_c]

                elif tile == FRIENDLY:
                    reward = -0.05  # already own this tile, slight penalty

            else:
                reward = -0.1  # out of bounds

            # Time survival bonus every step
            reward += 0.1

            # Check lose condition - agent returns to base and base is surrounded
            # Check lose condition
            if self._check_lose_condition():
                reward = -10.0
                terminated = True

            # Enemy takes its turn
            enemy_reward = self._move_enemy()
            reward += enemy_reward

            # Random event
            event_reward, event_name = self._random_event()
            reward += event_reward
    
            return self._get_obs(), reward, terminated, truncated, {}

    def _check_lose_condition(self):
        # Lose if enemy reaches the friendly base tile directly
        base_r, base_c = 0, 0
        e_r, e_c = self.enemy_pos
        return [e_r, e_c] == [base_r, base_c]

    def _get_obs(self):
        return self.grid.flatten()

    def render(self):
        if self.render_mode != "human":
            return

        # Initialize pygame on first render call
        if not hasattr(self, '_screen'):
            pygame.init()
            self.cell_size = 80
            self.screen_size = self.grid_size * self.cell_size
            self._screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Wartime-RL")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("Arial", 18, bold=True)

        # Colors for each tile type
        colors = {
            0: (200, 200, 200),   # EMPTY - light gray
            1: (100, 180, 100),   # FRIENDLY - green
            2: (200, 80,  80),    # ENEMY - red
            3: (220, 180, 50),    # RESOURCE - gold
            4: (80,  60,  40),    # OBSTACLE - dark brown
            5: (50,  100, 200),   # BASE - blue
        }

        tile_labels = {
            0: "",
            1: "F",
            2: "E",
            3: "R",
            4: "X",
            5: "B",
        }

        # Handle window close button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw grid
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                tile = int(self.grid[r][c])
                color = colors.get(tile, (200, 200, 200))
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                   self.cell_size, self.cell_size)

                # Fill tile
                pygame.draw.rect(self._screen, color, rect)

                # Draw border
                pygame.draw.rect(self._screen, (50, 50, 50), rect, 1)

                # Draw label
                label = tile_labels.get(tile, "")
                if label:
                    text = self._font.render(label, True, (255, 255, 255))
                    text_rect = text.get_rect(center=rect.center)
                    self._screen.blit(text, text_rect)

        # Highlight agent position
        ar, ac = self.agent_pos
        agent_rect = pygame.Rect(ac * self.cell_size, ar * self.cell_size,
                                  self.cell_size, self.cell_size)
        pygame.draw.rect(self._screen, (255, 255, 255), agent_rect, 4)

        # Draw step counter
        step_text = self._font.render(f"Step: {self.steps}", True, (0, 0, 0))
        self._screen.blit(step_text, (5, 5))

        pygame.display.flip()
        self._clock.tick(10)  # 10 FPS

    def close(self):
        if hasattr(self, '_screen'):
            pygame.quit()

    def close(self):
        pass

    def _roll_dice(self, num_dice):
        """Roll num_dice six-sided dice and return the highest value."""
        return max(self.np_random.integers(1, 7) for _ in range(num_dice))

    def _resolve_combat(self, attacker_dice=2, defender_dice=1):
        """
        Resolve combat between attacker and defender.
        Returns True if attacker wins, False if defender wins.
        Ties go to defender.
        """
        attacker_roll = self._roll_dice(attacker_dice)
        defender_roll = self._roll_dice(defender_dice)
        return attacker_roll > defender_roll
    
    def _move_enemy(self):
        """Simple enemy AI - moves toward the friendly base."""
        base_r, base_c = 0, 0
        e_r, e_c = self.enemy_pos

        # Calculate direction toward base
        dr = np.sign(base_r - e_r)
        dc = np.sign(base_c - e_c)

        # Try to move toward base, prioritize row then column
        moved = False
        for new_r, new_c in [(e_r + dr, e_c), (e_r, e_c + dc)]:
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                tile = self.grid[new_r][new_c]

                if tile == OBSTACLE:
                    continue

                elif tile == FRIENDLY or tile == BASE:
                    # Enemy attacks friendly tile via dice roll
                    if self._resolve_combat(attacker_dice=1, defender_dice=2):
                        # Enemy wins - captures tile
                        self.grid[e_r][e_c] = EMPTY
                        self.grid[new_r][new_c] = ENEMY
                        self.enemy_pos = [new_r, new_c]
                        return -3.0  # agent loses a tile
                    else:
                        # Enemy loses combat
                        return 0.0
                    moved = True
                    break

                elif tile == EMPTY:
                    # Enemy moves into empty tile
                    self.grid[e_r][e_c] = EMPTY
                    self.grid[new_r][new_c] = ENEMY
                    self.enemy_pos = [new_r, new_c]
                    moved = True
                    break

        return 0.0
    
    def _random_event(self):
        """
        Random events that can help or hinder the agent.
        10% chance of triggering each step.
        """
        if self.np_random.random() > 0.10:
            return 0.0, "No event"

        events = [
            "supply_drop",      # bonus resources appear
            "enemy_retreat",    # enemy moves away from base
            "ambush",          # enemy gets extra move
            "reinforcements",  # agent gets attack bonus next turn
        ]

        event = events[self.np_random.integers(0, len(events))]
        reward = 0.0

        if event == "supply_drop":
            # Place a new resource tile in a random empty cell
            empty_cells = list(zip(*np.where(self.grid == EMPTY)))
            if empty_cells:
                idx = self.np_random.integers(0, len(empty_cells))
                r, c = empty_cells[idx]
                self.grid[r][c] = RESOURCE
                reward = +1.0

        elif event == "enemy_retreat":
            # Enemy moves away from base randomly
            e_r, e_c = self.enemy_pos
            directions = [(-1,0),(1,0),(0,-1),(0,1)]
            self.np_random.shuffle(directions)
            for dr, dc in directions:
                new_r, new_c = e_r + dr, e_c + dc
                if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                    if self.grid[new_r][new_c] == EMPTY:
                        self.grid[e_r][e_c] = EMPTY
                        self.grid[new_r][new_c] = ENEMY
                        self.enemy_pos = [new_r, new_c]
                        break

        elif event == "ambush":
            # Enemy gets an extra attack this turn
            reward += self._move_enemy()

        elif event == "reinforcements":
            # Agent gets attack bonus next turn
            self.attack_bonus = True
            reward = +0.5

        return reward, event