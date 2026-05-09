# WarTime-RL Game Mechanics

## Current Mechanics

- The game uses a 13-territory North/South America board.
- The agent starts in Alaska and the enemy starts in Argentina.
- Turns are split into two phases:
  - `reinforce`: the agent deploys one pending army per deploy action.
  - `attack`: the agent attacks a neighboring non-agent territory or passes.
- The action space is unified:
  - deploy actions for each territory
  - attack actions for each valid adjacency pair
  - one pass action
- `valid_action_mask()` exposes legal actions for the current phase.
- `sample_valid_action()` is used by the visual demo as a valid random policy.

## Reinforcement, Troops and Armies
Troop dynamics changed from mostly static/automatic to phase-based, action-controlled deployment for the agent.

REINFORCE -> ATTACK -> ENEMY TURN -> REINFORCE

What changed specifically:

The agent now gets pending_reinforcements.
During the reinforce phase, each deploy action places 1 army on an owned territory.
Once all pending reinforcements are placed, the game moves to the attack phase.
The agent can then attack or pass.
Capturing a territory moves armies from the source territory into the captured territory.
Army counts are capped by max_armies_per_territory.
Reinforcement counts are based on territory count plus continent bonuses.

- Army counts are capped by `GameplayConfig.max_armies_per_territory`.
- Capturing a territory moves `GameplayConfig.capture_move_armies` from the
  attacking territory while keeping at least one army behind.

## Combat and Events

- Combat uses configurable dice values from `GameplayConfig`.
- Failed attacks remove `GameplayConfig.combat_loss_armies`.
- Existing capture, combat, survival, terminal, and event rewards remain in
  `RewardConfig`.
- This branch does not add continent-completion or territory-control reward
  shaping.
- Random events are controlled by curriculum phase probability:
  - `supply_drop`
  - `enemy_retreat`
  - `ambush`
  - `reinforcements`
- Event army effects are configurable in `GameplayConfig`.
- Losing now correctly applies the negative `lose_game` reward.

## Curriculum

- `CurriculumConfig` defines beginner, intermediate, and hard phases.
- Phases control:
  - max episode steps
  - starting army counts
  - neutral army counts
  - random event probability
- `CurriculumTracker.__repr__()` now safely reports the current win rate.

## Map and Sprite Integrity

- The current board intentionally excludes off-map Risk territories.
- Alaska no longer references missing `Kamchatka`.
- Gameplay uses `env.map_config` as the canonical board.
- `generate_sprites.py` remains a standalone utility script and was left in its
  original shape for now.

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

- turn phase and pending reinforcements
- turn-flow badges for reinforce and attack
- last action and action type
- reward breakdown
- random event status
- reinforcement counts
- episode counters and win/loss/timeout summary
- current speed
- last four action log entries

The map also highlights:

- the latest deploy target
- attack source and target
- attack path line between territories

At episode end, a short overlay shows the outcome, steps, reward, and final
territory counts before reset.

## Current Limitations

- The visual demo is still a valid random policy, not a trained agent.
- Combat logs show action and reward outcome, not exact dice rolls.
- Random events do not yet report affected territories for map flashing.
- The board is still small and fixed-start; randomized starts and map expansion
  are future mechanics work.
- Replay buffer and DQN training code are still in works.
