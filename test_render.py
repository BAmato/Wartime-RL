"""
test_render.py
Runs the live WarTime Env with a random agent and prints per episode stats
Should be used for verifying if the environment is stepping correctly

Controls:
Q/Esc   -quit
SPACE   - pause/ unpause
"""
import pygame
import sys
from env.wartime_env import WartimeEnv

# CONFIG
MAX_EPISODES = 50
SPEEDS = {
    pygame.K_1: ("Slow", 3),
    pygame.K_2: ("Medium", 10),
    pygame.K_3: ("Fast", 24),
}
DEFAULT_SPEED_KEY = pygame.K_2
OVERLAY_MS = 1500

# SETUP

env = WartimeEnv(render_mode="human", curriculum_level=0) #Beginner
# env = WartimeEnv(render_mode="human", curriculum_level=1) #Intermediate
# env = WartimeEnv(render_mode="human", curriculum_level=2) #Hard
obs, _ = env.reset()
episode = 1
episode_reward = 0.0
episode_steps = 0
paused = False
episode_results = []
action_log = []
speed_label, render_fps = SPEEDS[DEFAULT_SPEED_KEY]


def simulation_stats():
    completed = len(episode_results)
    wins = sum(1 for result in episode_results if result["outcome"] == "WIN")
    losses = sum(1 for result in episode_results if result["outcome"] == "LOSS")
    timeouts = sum(1 for result in episode_results if result["outcome"] == "TIMEOUT")
    total_reward = sum(result["reward"] for result in episode_results)
    avg_reward = total_reward / completed if completed else 0.0
    return {
        "completed_episodes": completed,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "avg_reward": avg_reward,
    }


def parse_highlights(action_label):
    if action_label.startswith("deploy:"):
        return {"active_deploy": action_label.split(":", 1)[1]}
    if action_label.startswith("attack:") and "->" in action_label:
        route = action_label.split(":", 1)[1]
        source, target = route.split("->", 1)
        return {"active_source": source, "active_target": target}
    return {}


def append_action_log(action_label, info, reward):
    entry = action_label
    if entry.startswith("deploy:"):
        entry = entry.replace("deploy:", "Deploy ")
    elif entry.startswith("attack:"):
        entry = entry.replace("attack:", "")
    entry = f"{entry}: {info.get('action_type', 'none')} {reward:+.2f}"
    action_log.append(entry)
    del action_log[:-4]


def build_hud(info=None, action_label="none", step_reward=0.0, episode_overlay=None):
    hud = simulation_stats()
    hud.update({
        "episode": episode,
        "episode_reward": episode_reward,
        "step_reward": step_reward,
        "action_label": action_label,
        "action_log": action_log,
        "speed_label": speed_label,
        "render_fps": render_fps,
        "episode_overlay": episode_overlay,
        "info": info or {},
    })
    hud.update(parse_highlights(action_label))
    return hud


env.render(build_hud())


def record_episode(outcome, steps, total_reward, agent_territories, enemy_territories):
    episode_results.append({
        "outcome": outcome,
        "steps": steps,
        "reward": total_reward,
        "agent_territories": agent_territories,
        "enemy_territories": enemy_territories,
    })


def print_simulation_summary(current_steps=0, current_reward=0.0):
    completed = len(episode_results)
    wins = sum(1 for result in episode_results if result["outcome"] == "WIN")
    losses = sum(1 for result in episode_results if result["outcome"] == "LOSS")
    timeouts = sum(1 for result in episode_results if result["outcome"] == "TIMEOUT")
    total_steps = sum(result["steps"] for result in episode_results)
    total_reward = sum(result["reward"] for result in episode_results)

    print(f"\n{'='*55}")
    print("    Simulation Summary")
    print(f"{'='*55}")
    print(f"completed_episodes : {completed}")
    print(f"wins/losses/timeouts: {wins}/{losses}/{timeouts}")

    if completed:
        avg_steps = total_steps / completed
        avg_reward = total_reward / completed
        win_rate = wins / completed
        best_reward = max(result["reward"] for result in episode_results)
        worst_reward = min(result["reward"] for result in episode_results)

        print(f"win_rate           : {win_rate:.1%}")
        print(f"total_steps        : {total_steps}")
        print(f"avg_steps          : {avg_steps:.1f}")
        print(f"total_reward       : {total_reward:+.2f}")
        print(f"avg_reward         : {avg_reward:+.2f}")
        print(f"best/worst_reward  : {best_reward:+.2f}/{worst_reward:+.2f}")
    else:
        print("win_rate           : n/a")
        print("total_steps        : 0")
        print("total_reward       : +0.00")

    if current_steps:
        print(f"unfinished_episode : steps={current_steps}, reward={current_reward:+.2f}")
    print(f"{'='*55}\n")


def shutdown():
    print_simulation_summary(episode_steps, episode_reward)
    env.close()
    sys.exit()

print(f"\n{'='*55}")
print("    WarTime-RL -- Agent Check")
print(f"     Action space : {env.action_space}")
print(f"    OBS shape : {env.observation_space.shape}")
print(f"    Curriculum Level: {env.curriculum_level}")
print(f"\n{'='*55}")

# MAIN LOOP
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            shutdown()
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
            print("[EXIT]")
            shutdown()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            paused = not paused
            print("[PAUSED]" if paused else "[RESUMED]")
        if event.type == pygame.KEYDOWN and event.key in SPEEDS:
            speed_label, render_fps = SPEEDS[event.key]
            print(f"[SPEED] {speed_label} ({render_fps} fps)")

    if paused:
        if pygame.display.get_init():
            pygame.display.flip()
        clock.tick(10)
        continue

    # STEP
    action = env.sample_valid_action()
    obs, reward, terminated, truncated, info = env.step(action)
    action_label = env.describe_action(action)

    episode_reward += reward
    episode_steps += 1
    append_action_log(action_label, info, reward)
    env.render(build_hud(info, action_label, reward))

    # DEBUG STEP_LEVEL
    print(
        f"step={info['step']:3d} \n"
        f"action={action:3d} ({action_label}) \n"
        f"phase={info['turn_phase']} pending={info['pending_reinforcements']} \n"
        f"reward={reward:+.2f} \n"
        f"agent={info['agent_territories']}\n"
        f"enemy={info['enemy_territories']} \n"
        f"event={info['event']}"
    )

    if terminated or truncated:
        outcome = (
            "WIN" if info["enemy_territories"] == 0
            else "LOSS" if info["agent_territories"] == 0
            else "TIMEOUT"
        )
        record_episode(
            outcome,
            episode_steps,
            episode_reward,
            info["agent_territories"],
            info["enemy_territories"],
        )

        print(f"  Episode {episode:3d} | {outcome:7s} | "
              f"steps={episode_steps:3d} | "
              f"total_reward={episode_reward:+7.2f} | "
              f"agent_terr={info['agent_territories']:2d} | "
              f"enemy_terr={info['enemy_territories']:2d}")
        overlay = {
            "outcome": outcome,
            "steps": episode_steps,
            "reward": episode_reward,
            "agent_territories": info["agent_territories"],
            "enemy_territories": info["enemy_territories"],
        }
        env.render(build_hud(info, action_label, reward, overlay))
        pygame.time.delay(OVERLAY_MS)

        if MAX_EPISODES and episode >= MAX_EPISODES:
            print(f"\nReached: {MAX_EPISODES} episodes. See ya later!")
            print_simulation_summary()
            env.close()
            sys.exit()

        episode += 1
        episode_reward = 0.0
        episode_steps = 0
        action_log.clear()
        obs, _ = env.reset()
    clock.tick(render_fps)
