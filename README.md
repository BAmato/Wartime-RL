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
