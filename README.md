# Grasp-VLA

This repo includes a small `raw_data/` dataset (logged robot ticks + rendered images) and training scripts to fine-tune vision-language-action models (e.g., **OpenVLA**) on it.

## Quickstart

Install deps:

```bash
pip install -r requirements.txt
```

### Repo layout

- `raw_data/`: local dataset folder (ignored by git)
- `dataloaders/`: dataset + collation code (reusable across models)
- `train_scripts/`: runnable training entrypoints (add more scripts here for other models)
- `checkpoints/`: saved finetuned outputs (ignored by git; contains `.gitkeep`)

### OpenVLA

Run a small LoRA fine-tune (requires downloading a large OpenVLA checkpoint):

```bash
python train_scripts/openvla_v1.py --model_id openvla/openvla-v01-7b --max_steps 50
```

If you just want to validate dataset parsing + model loading:

```bash
python train_scripts/openvla_v1.py --model_id openvla/openvla-v01-7b --dry_run
```

Notes:
- The script fine-tunes OpenVLA in its native format (discretized **action tokens**).
- On newer `transformers` versions, OpenVLA may fail to initialize with SDPA; this script defaults to `--attn_implementation eager` for compatibility.
- The script computes normalization stats from `raw_data/` and saves them into the model config under `--norm_key` (default: `grasp_vla_raw_data`).