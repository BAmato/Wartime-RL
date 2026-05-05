from env.wartime_env import WartimeEnv

env = WartimeEnv(render_mode="human")
obs, _ = env.reset()

while True:
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        print(f"Episode ended. Resetting...")
        obs, _ = env.reset()