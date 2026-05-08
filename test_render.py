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

#CONFIG
MAX_EPISODES=50
RENDER_FPS=10

#SETUP

env = WartimeEnv(render_mode="human")
obs, _ = env.reset()

env.render()
episode=1
episode_reward=0.0
episode_steps=0
paused=False

print(f"\n{'='*55}")
print("    WarTime-RL -- Agent Check")
print(f"     Action space : {env.action_space}")
print(f"    OBS shape : {env.observation_space.shape}")
print(f"\n{'='*55}")

#MAIN LOOP
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            env.close()
            sys.exit()
        if event.type==pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                print("[EXIT]")
                env.close()
                sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            paused = not paused
            print("[PAUSED]" if paused else "[RESUMED]")

    if paused:
        if pygame.display.get_init():
            pygame.display.flip()
        clock.tick(10)
        continue

    #STEP
    action = env.action_space.sample()
    obs,reward, terminated, truncated, info=env.step(action)
    env.render()

    episode_reward+=reward
    episode_steps+=1

    #DEBUG STEP_LEVEL
    print(f"step={info['step']:3d} \naction={action:3d} \nreward={reward:+.2f} \nagent={info['agent_territories']}\nenemy={info['enemy_territories']} \nevent={info['event']}")
    
    if terminated or truncated:
        outcome="WIN" if info["enemy_territories"]==0 else "LOSS" if info["agent_territories"]==0 else "TIMEOUT"

        print(f"  Episode {episode:3d} | {outcome:7s} | "
            f"steps={episode_steps:3d} | "
            f"total_reward={episode_reward:+7.2f} | "
            f"agent_terr={info['agent_territories']:2d} | "
            f"enemy_terr={info['enemy_territories']:2d}")
        
        if MAX_EPISODES and episode>=MAX_EPISODES:
             print(f"\nReached: {MAX_EPISODES} episodes. See ya later!")
             env.close()
             sys.exit()

        episode+=1
        episode_reward=0.0
        episode_steps=0
        obs,_=env.reset()
    clock.tick(RENDER_FPS)