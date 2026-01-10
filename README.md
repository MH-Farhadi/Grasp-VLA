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
python train_scripts/openvla/openvla_v1.py --model_id openvla/openvla-v01-7b --max_steps 50
```

If you just want to validate dataset parsing + model loading:

```bash
python train_scripts/openvla/openvla_v1.py --model_id openvla/openvla-v01-7b --dry_run
```

Notes:
- The script fine-tunes OpenVLA in its native format (discretized **action tokens**).
- On newer `transformers` versions, OpenVLA may fail to initialize with SDPA; this script defaults to `--attn_implementation eager` for compatibility.
- The script computes normalization stats from `raw_data/` and saves them into the model config under `--norm_key` (default: `grasp_vla_raw_data`).

### Pi-Zero

`train_scripts/pi-zero/pizero-v1.py` is a **template** entrypoint wired to `raw_data/`. It loads a checkpoint from `--model_id` and fine-tunes with a generic image+text → action-text objective (easy to adapt to a Pi-Zero-specific action head/tokenization if needed).

Dry run:

```bash
python train_scripts/pi-zero/pizero-v1.py --model_id <HF_OR_LOCAL_PIZERO_ID> --dry_run
```

### X-VLA

`train_scripts/X-VLA/X_VLA_v1.py` is a template entrypoint wired to `raw_data/`. It loads a checkpoint from `--model_id` and fine-tunes with a generic image+text → action-text objective.

```bash
python train_scripts/X-VLA/X_VLA_v1.py --model_id <HF_OR_LOCAL_XVLA_ID> --dry_run
```

### LeRobot X-VLA (XVLA v1)

If you already converted `raw_data/` into a LeRobotDataset, you can skip rebuilding the dataset and train with a fixed validation holdout (40 episodes):

```bash
python train_scripts/lerobot-xvla/xvla-v1.py \
  --skip_convert \
  --dataset_dir lerobot_datasets/grasp_raw \
  --dataset_repo_id grasp_raw \
  --policy_path lerobot/xvla-base \
  --output_dir outputs/xvla_v1_run \
  --val_episodes 40
```