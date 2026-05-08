# Wartime-RL
## Artificial Intelligence 4320 Final Project

Wartime-RL is a custom OpenAI Gymnasium environment designed for research into strategic territory control using Deep Q-Networks (DQN). This environment was designed from scratch by the team as an original simulation. The key motivation is the unique presence of stochasticity in combat outcomes: the dice mechanic means the agent must learn to reason under uncertainty, developing risk-tolerant strategies rather than simply memorizing optimal action sequences.

## Team Members
- Bryan Amato
- Kimberly Alicea-De Leon
- Damian Villarreal

## Setup

**With NVIDIA GPU (RTX 5070):**
```bash
conda env create -f environment_gpu.yml
conda activate wartime-rl
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
```

**Without GPU (CPU only):**
```bash
conda env create -f environment_cpu.yml
conda activate wartime-rl
```

**After either setup:**
```bash
mkdir assets
python generate_sprites.py
python test_render.py
```