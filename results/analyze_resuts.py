import pandas as pd
import re
import os
import glob

def get_latest_log():
    files = glob.glob("dqn_training_log_*.csv")
    return max(files, key=os.path.getctime) if files else print("No file")

def parse(file_path='config.py'):
    if not os.path.exists(file_path): return {}
    with open(file_path, 'r') as f:
        content = f.read()
    
    rewards = {}
    reward_block = re.search(r'class RewardConfig:.*?(?=@dataclass|class|$)', content, re.DOTALL)
    if reward_block:
        for line in reward_block.group(0).split('\n'):
            match = re.search(r'(\w+):\s*float\s*=\s*([+-]?\d+\.?\d*)', line)
            if match:
                rewards[match.group(1)] = float(match.group(2))
    return rewards

def analyze(log_path):
    df = pd.read_csv(log_path, comment='#')
    df['outcome'] = df['outcome'].str.strip().str.lower()

    recent_50 = df.tail(100)
    win_rate = (recent_50['outcome'] == 'win').mean()
    avg_reward = recent_50['reward'].mean()
    avg_steps = recent_50['ep_steps'].mean()
    
    print(f"\n--- LOG ANALYSIS: {log_path} ---")
    print(f"Total Episodes: {len(df)}")
    print(f"Outcome Distribution:\n{df['outcome'].value_counts()}")
    print(f"\nRecent Performance (Last 100 Episodes):")
    print(f" - Win Rate:     {win_rate:.2%}")
    print(f" - Avg Reward:   {avg_reward:.2f}")
    print(f" - Avg Steps:    {avg_steps:.1f}")
    print(f" - Current Eps:  {df['epsilon'].iloc[-1]:.4f}")

current_rewards = parse()
latest_log = get_latest_log()

if current_rewards:
    print("--- CURRENT CONFIG REWARDS ---")
    for r, val in current_rewards.items():
        print(f"{r:20}: {val:+.2f}")

if latest_log:
    analyze(latest_log)
else:
    print("\nNo log files found. Please ensure logs are in the same folder.")