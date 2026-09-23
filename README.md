# SCREENHAYSTACK: Finding Blind Zones in GUI Grounding

This repository contains the code for the blind-zone discovery and controlled-relocation experiments described in **SCREENHAYSTACK: Finding Blind Zones in GUI Grounding**. SCREENHAYSTACK evaluates whether a GUI grounding model can localize the same target reliably across different image-plane locations. It places controlled probe icons over GUI screenshots, aggregates model accuracy spatially, and exposes model-specific low-accuracy regions called *blind zones*.



## Scope

This code snapshot contains two parts:

1. **Spatial probing and blind-zone discovery** (`find/`): place controlled targets across GUI backgrounds, run a grounding model, and save per-location predictions, accuracy matrices, and heatmaps.
2. **Controlled spatial relocation** (`move_code/`): apply wrap-around shifts to ScreenSpot-Pro examples and measure the effect of moving targets across or within blind-zone boundaries.

The repository supports the following models:

| Launcher name | Model |
| --- | --- |
| `qwen3-32b` | `Qwen/Qwen3-VL-32B-Instruct` |
| `qwen3-8b` | `Qwen/Qwen3-VL-8B-Instruct` |
| `gta1` | `HelloKKMe/GTA1-7B` |
| `uitars` | `ByteDance-Seed/UI-TARS-1.5-7B` |
| `uivenus` | `inclusionAI/UI-Venus-Ground-7B` |

This snapshot does **not** include datasets, model weights, generated results, or the Click-100k fine-tuning and blind-zone-oriented augmentation code from the mitigation experiments.

## Repository layout

```text
move_code_and_find/
├── LICENSE
├── README.md
├── requirements.txt
├── find/
│   ├── run.py                  # Unified probing launcher
│   ├── run_vllm_offline.sh     # Shell wrapper; uses PyTorch/torchrun
│   ├── common.py               # Shared probing utilities
│   ├── icon.py                 # Probe-icon generator
│   ├── background/             # GUI probing backgrounds and metadata
│   ├── qwen.py                 # Qwen3-VL evaluator
│   ├── gta1.py                 # GTA1 evaluator
│   ├── uitars.py               # UI-TARS evaluator
│   └── uivenus.py              # UI-Venus evaluator
└── move_code/
    ├── runs/                   # Per-model relocation launchers
    └── scripts/                # Relocation and evaluation implementations
```

## Installation

The pinned dependencies reproduce the local `torch311` environment: Python 3.11.14, PyTorch 2.7.1, and CUDA 12.6. Create and activate a Python 3.11 environment, install the matching PyTorch build first, and then install the complete dependency set:

```bash
python -m pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
  --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-build-isolation -r requirements.txt
```

The second command uses `--no-build-isolation` so that `flash-attn==2.8.3` can build against the already installed PyTorch and CUDA environment. A working CUDA 12.6 toolkit and compatible NVIDIA driver are required for this exact setup.

## Data preparation

### Spatial probing

The bundled `find/background/` directory contains the GUI probing backgrounds and their metadata. To use another background set, pass its path with `--bg_dir`.

Prepare:

- a directory containing the probe-icon assets;
- local or Hugging Face access to the selected model weights.

The probing scripts recognize three target types:

- `star`: pure icon;
- `circle_ok`: icon with a text label;
- `clock`: icon containing embedded text.

Pass the background and icon directories explicitly with `--bg_dir` and `--icon_dir`. The expected model-specific assets are:

| Evaluator | `star` | `circle_ok` | `clock` |
| --- | --- | --- | --- |
| Qwen3-VL | `Today.png` | `API.png` | `clock_text_50x35.png` |
| GTA1 | `star_icon_40.png` | `circle_ok_40.png` | `clock_text_60x40.png` |
| UI-TARS | `star_icon_40.png` | `circle_ok_40.png` | `clock_text_40x60.png` |
| UI-Venus | `star_icon_40.png` | `circle_ok_40.png` | `clock_text_60x40.png` |

Run the bundled generator to create the GTA1/UI-Venus 40-pixel probe assets under `find/icons/`:

```bash
cd find
python icon.py
```

Additional Qwen3-VL and UI-TARS asset sizes shown in the table must also be present when those evaluators are used.

### Controlled relocation

The relocation experiments require:

- ScreenSpot-Pro annotations and images;
- a JSONL subset whose targets are outside the model's blind zones;
- a JSONL subset whose targets are inside the model's blind zones;
- the model-specific blind-zone region file.

The per-model paths are configured in `move_code/runs/*.sh`. Update `NOT_IN_SUBSET`, `WORST_SUBSET`, `REGION_BBOX`, and `OUTPUT_DIR` to match your environment before running the experiments. Most relocation scripts accept blind-zone regions as absolute-pixel `bbox` values or relative/absolute `bounds`; `move_out_rel.py` expects region coordinates normalized to `[0, 1]`.

## 1. Run spatial probing

From the `find/` directory:

```bash
python run.py qwen3-8b clock \
  --bg_dir /path/to/backgrounds \
  --icon_dir /path/to/icons
```

Other examples:

```bash
python run.py qwen3-32b circle_ok --bg_dir /path/to/backgrounds --icon_dir /path/to/icons
python run.py gta1 star --bg_dir /path/to/backgrounds --icon_dir /path/to/icons
python run.py uitars clock --bg_dir /path/to/backgrounds --icon_dir /path/to/icons
python run.py uivenus clock --bg_dir /path/to/backgrounds --icon_dir /path/to/icons
```

Useful options:

```text
--gpus 0,1             Visible CUDA devices
--nproc 2              Number of data-parallel workers
--model_path PATH      Override the default model checkpoint
--output_dir PATH      Override the output directory
--batch_size N         Inference batch size
--max_backgrounds N    Limit the number of backgrounds for a smoke test
```

Each worker loads a complete model; `--nproc` provides data parallelism and does not shard one model across GPUs.

A small smoke test can be launched with:

```bash
python run.py qwen3-8b clock \
  --bg_dir /path/to/backgrounds \
  --icon_dir /path/to/icons \
  --batch_size 1 \
  --max_backgrounds 1
```

The probing scripts write JSONL predictions, NumPy accuracy matrices, and PNG heatmaps under `results/<model>/` unless `--output_dir` is supplied. Rank-specific files are merged by rank 0 after distributed inference.

## 2. Run controlled relocation

The relocation code uses a fixed-size wrap-around transformation: pixels crossing one image boundary reappear on the opposite side. The target annotation is shifted by the same offset. This changes the target's absolute position while preserving the full screenshot content.

Available strategies are:

| Strategy | Input subset | Transformation |
| --- | --- | --- |
| `in_region` | Outside blind zones | Move the target into a blind zone |
| `out` | Inside blind zones | Move the target outside blind zones |
| `random_rel` | Outside blind zones | Outside-to-outside relocation control |
| `random_in` | Inside blind zones | Inside-to-inside random-shift control |
| `all` | Both subsets | Run all four strategies sequentially |

From the `move_code/` directory:

```bash
# Inspect the launcher without loading a model.
bash runs/qwen3_8b.sh --help

# Run a small relocation test.
bash runs/qwen3_8b.sh in_region --max_samples 10

# Run one complete strategy.
bash runs/gta_7b.sh out

# Run all strategies for one model.
bash runs/ui_venus_7b.sh all
```

Select GPUs and worker count through environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=29521 \
  bash runs/qwen3_8b.sh in_region
```

Additional arguments placed after the strategy are forwarded to the Python evaluator and override the model preset. Relocation outputs include per-run predictions, transformed annotations, correctness labels, and summary JSON files.

## Output interpretation

A grounding prediction is correct when the parsed point lies inside the target bounding box. UI-Venus natively predicts `[x1, y1, x2, y2]`; its predicted box center is converted to a point so that all models use the same point-in-box criterion.

For blind-zone discovery, aggregate accuracy over probe locations to obtain a spatial accuracy map. The paper uses the bottom 20% of grid cells as a fixed-budget blind-zone mask for downstream transfer and relocation experiments. It also reports a diagnostic threshold of cell accuracy below `0.8` when measuring blind-zone area and severity.

The relocation comparison is designed to separate a blind-zone boundary effect from the generic effect of shifting an image:

- compare outside-to-inside relocation against the original outside-blind-zone baseline;
- compare inside-to-outside relocation against the original inside-blind-zone baseline;
- use outside-to-outside and inside-to-inside shifts as controls.

## Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@article{li2026screenhaystack,
  title   = {SCREENHAYSTACK: Finding Blind Zones in GUI Grounding},
  author  = {Chenyue Li and Xiaoxiao Sun and Yubo Deng and Qinlin Zhao and Serena Yeung-Levy and Yuhui Zhang},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
