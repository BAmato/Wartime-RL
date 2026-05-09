# WarTime-RL Game Mechanics

## Current Mechanics

- The game uses a 13-territory North/South America board.
- The agent starts in Alaska and the enemy starts in Argentina.
- Each episode is bounded by `max_turns` full agent turns (not raw steps).
- Turns are split into three mandatory phases every round:
  - `reinforce`: the agent must place ALL pending armies before advancing.
  - `attack`: the agent attacks a neighboring non-agent territory or passes.
  - `fortify`: the agent moves armies between adjacent owned territories, or passes.
- The action space is unified:
  - deploy actions for each territory
  - attack/fortify actions for each valid adjacency pair
  - one pass action (valid during attack and fortify only)
- `valid_action_mask()` exposes legal actions for the current phase.
- `sample_valid_action()` is used by the visual demo as a valid random policy.

## Turn Structure

```
REINFORCE → ATTACK → FORTIFY → ENEMY TURN → REINFORCE → ...
```

Every turn follows this exact sequence. The fortify phase always happens after
attack — `_end_player_turn()` (which runs the enemy turn and starts the next
agent turn) is only ever called from the fortify phase handler.

`self.turns` increments once per completed agent turn (at the end of the
fortify phase). `truncated` is triggered when `self.turns >= self.max_turns`.

## Reinforcement, Troops and Armies

### Mandatory Deploy (Risk rules)

- Every turn starts with the reinforce phase. It cannot be skipped.
- The agent receives `pending_reinforcements` computed from territory count plus
  continent bonuses (minimum `GameplayConfig.min_reinforcements`).
- During reinforce, **pass is not a valid action**. The agent must place all
  pending armies before the phase ends.
- Each deploy action places 1 army on one owned territory.
- The phase transitions to attack automatically when `pending_reinforcements` reaches 0.
- If all owned territories are at `max_armies_per_territory` before all
  reinforcements are placed, remaining armies are auto-placed on the least-full
  owned territory to avoid deadlock.
- A `deploy` reward (`+0.05` per army placed) provides a learning signal that
  deploying is useful.

### Army Caps and Captures

- Army counts are capped by `GameplayConfig.max_armies_per_territory`.
- Capturing a territory moves armies from the attacking territory into the
  captured territory, leaving at least 1 behind.
- Reinforcement counts are based on territory count plus continent bonuses.

## Combat and Rewards

### Agent Attack Outcomes

| `combat_result` value | Situation | Reward |
|---|---|---|
| `"neutral_capture"` | Agent takes undefended neutral territory | `+capture_neutral` |
| `"win_territory"` | Agent defeats enemy defender and captures territory | `+win_combat` |
| `"lose_combat"` | Agent loses dice comparison; enemy holds territory | `lose_combat` |
| `"none"` | Non-attack action (deploy, pass, fortify) | — |

### Enemy Attack Outcomes

When the enemy captures an agent territory during the enemy turn:
- Reward applied: `lose_combat + lose_territory` (default `−1.0 + −5.0 = −6.0`)
- `info["combat_result"]` is set to `"lose_territory"` for that step

When the enemy captures a neutral territory:
- Reward applied: `lose_combat` only (default `−1.0`)

This separates the "lost a dice roll" signal from the "lost owned ground" signal,
giving the agent a stronger gradient to defend its territories.

### Other Events

- Random events are controlled by curriculum phase probability:
  - `supply_drop` — agent gains armies and `+supply_drop` reward
  - `enemy_retreat` — enemy loses armies (no reward)
  - `ambush` — enemy gets a bonus turn (reward: same as normal enemy attack)
  - `reinforcements` — agent gains attack_bonus_dice for next attack
- Event army effects are configurable in `GameplayConfig`.
- Losing now correctly applies the negative `lose_game` reward.

## Curriculum

- `CurriculumConfig` defines beginner, intermediate, and hard phases.
- Phases control:
  - `max_turns` — episode length in full agent turns (30 / 25 / 20)
  - starting army counts
  - neutral army counts
  - random event probability
- `CurriculumTracker.__repr__()` safely reports the current win rate.

## Map and Sprite Integrity

- The current board intentionally excludes off-map Risk territories.
- Alaska no longer references missing `Kamchatka`.
- Gameplay uses `env.map_config` as the canonical board.
- `generate_sprites.py` remains a standalone utility script.

## Visual Debug UI

Run the visual simulation with:

```bash
python3 test_render.py
```

Controls:

- `SPACE`: pause or resume
- `1`: slow speed
- `2`: medium speed
- `3`: fast speed
- `Q` or `Esc`: quit and print terminal summary

The Pygame HUD shows:

- turn phase badges (REINFORCE → ATTACK → FORTIFY)
- pending reinforcements and completed turns
- last action, action type, and combat result
- fortify source/target
- reward breakdown
- random event status
- reinforcement counts and attack bonus state
- curriculum level and phase name
- episode counters and win/loss/timeout summary

At episode end, a short overlay shows the outcome, steps, reward, and final
territory counts before reset.

## CSV Log Columns (train_dqn.py)

| Column | Description |
|---|---|
| `ep_steps` | Raw env steps in the episode |
| `turns` | Full agent turns completed in the episode |
| `combat_wins` | Steps where `combat_result == "win_territory"` |
| `combat_losses` | Steps where `combat_result == "lose_combat"` |
| `territory_losses` | Steps where `combat_result == "lose_territory"` |

## Current Limitations

- The visual demo is still a valid random policy, not a trained agent.
- Combat logs show action and reward outcome, not exact dice rolls.
- Random events do not yet report affected territories for map flashing.
- The board is still small and fixed-start; randomized starts and map expansion
  are future mechanics work.
