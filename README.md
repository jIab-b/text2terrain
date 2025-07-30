# Text2Terrain

Neural procedural terrain generation from natural language descriptions.

## Quick Start

```bash
# Simple setup (interactive)
python scripts/setup.py

# Manual setup with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -e .

# Optional extras
uv pip install -e ".[training]"  # For Fireworks AI training
uv pip install -e ".[dev]"       # Development tools

# Generate synthetic dataset
python -m src.data_gen.generator --n 100000 --output data/raw/demo

# Train locally (small dataset)
python -m src.training.train --config configs/base.yaml

# Train on Fireworks AI
fws submit fireworks.yaml

# Serve inference API
python -m src.inference.api

# Build and run terrain viewer
python scripts/build_all.py
./render/terrain_viewer
```

## Architecture

- **procgen/**: Differentiable procedural terrain engine (JAX)
- **data_gen/**: Synthetic dataset generation pipeline  
- **training/**: LoRA-based training on Fireworks AI
- **inference/**: Lightweight deployment API
- **render/**: Real-time Raylib terrain viewer

## Key Features

- **Compositional generalization**: Novel terrain descriptions work via base model embeddings
- **Seamless tiling**: World-coordinate evaluation ensures infinite scrolling
- **Lightweight deployment**: 8-rank LoRA + 20MB total model size
- **Real-time rendering**: GPU-accelerated heightmap visualization

## Training Data Format

Each sample contains:
- Natural language caption
- Module IDs and parameters  
- World coordinates and seeds
- 256×256 heightmap (PNG)

## License

MIT