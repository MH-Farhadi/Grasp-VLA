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

#### Rebuild LeRobot datasets (grasp boosting / grasp-only)

Build the full dataset with grasp-signal boosting + drop idle (“sleeping”) episodes:

```bash
python /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/train_scripts/lerobot-xvla/convert_raw_data_to_lerobot_dataset.py \
  --raw_data_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/raw_data \
  --out_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/lerobot_datasets/grasp_raw \
  --repo_id grasp_raw \
  --fps 5 \
  --overwrite \
  --canonicalize_task \
  --gripper_label_mode hybrid \
  --drop_idle_episodes \
  --idle_min_sum_dpos_m 0.02 \
  --idle_min_sum_drot_rad 0.2 \
  --oversample_gripper_episodes 8
```

Build a grasp-only dataset (drops episodes with no gripper close/open events):

```bash
python /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/train_scripts/lerobot-xvla/convert_raw_data_to_lerobot_dataset.py \
  --raw_data_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/raw_data \
  --out_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/lerobot_datasets/grasp_only \
  --repo_id grasp_only \
  --fps 5 \
  --overwrite \
  --canonicalize_task \
  --gripper_label_mode hybrid \
  --drop_gripperless_episodes \
  --oversample_gripper_episodes 1
```

#### Train (stage1 -> stage2) without overwriting datasets

Train on `grasp_raw` (uses the existing dataset on disk; no rebuild / no overwrite):

```bash
python /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/train_scripts/lerobot-xvla/xvla-v1.py \
  --raw_data_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/raw_data \
  --dataset_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/lerobot_datasets/grasp_raw \
  --dataset_repo_id grasp_raw \
  --skip_convert \
  --device cuda \
  --domain_id 29 \
  --output_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/outputs/xvla_v1_run \
  --stage1_steps 500 \
  --stage2_steps 2500
```

Optional second fine-tune pass on `grasp_only` (start from the previous run's `stage2` directory):

```bash
python /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/train_scripts/lerobot-xvla/xvla-v1.py \
  # IMPORTANT: replace this placeholder with the real stage2 folder from your first run (it must exist on disk)
  --policy_path /PATH/TO/FIRST_RUN/stage2 \
  --raw_data_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/raw_data \
  --dataset_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/lerobot_datasets/grasp_only \
  --dataset_repo_id grasp_only \
  --skip_convert \
  --resume \
  --device cuda \
  --domain_id 29 \
  --output_dir /home/kye/Desktop/Depo/Code/Grasp-VLA/Grasp-VLA/outputs/xvla_v1_grasp_only \
  --stage1_steps 0 \
  --stage2_steps 1500
```