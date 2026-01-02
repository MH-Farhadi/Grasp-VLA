"""
End-to-end helper: convert this repo's `raw_data/` into a LeRobotDataset, then fine-tune LeRobot's X-VLA.

This script is optional convenience glue around:
  1) `convert_raw_data_to_lerobot_dataset.py`
  2) `lerobot-train --policy.path=lerobot/xvla-base ...`

Example:

  python train_scripts/lerobot-xvla/finetune_xvla_on_raw_data.py \
    --raw_data_dir raw_data \
    --dataset_dir lerobot_datasets/grasp_raw \
    --dataset_repo_id grasp_raw \
    --output_dir outputs/xvla_raw_run \
    --steps 200 \
    --batch_size 1 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from convert_raw_data_to_lerobot_dataset import convert


def _unique_out_dir(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.parent / f"{path.name}_{stamp}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw_data -> LeRobotDataset and fine-tune LeRobot X-VLA")

    # Conversion
    p.add_argument("--raw_data_dir", type=str, default="raw_data")
    p.add_argument("--dataset_dir", type=str, default="lerobot_datasets/grasp_raw")
    p.add_argument("--dataset_repo_id", type=str, default="grasp_raw")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument(
        "--camera_key",
        type=str,
        default="image",
        help="Camera key suffix used for the converted dataset (default matches lerobot/xvla-base).",
    )
    p.add_argument("--overwrite_dataset", action="store_true")
    p.add_argument("--skip_convert", action="store_true")

    # Training
    p.add_argument("--policy_path", type=str, default="lerobot/xvla-base")
    p.add_argument(
        "--prefetch_policy",
        action="store_true",
        default=True,
        help="Pre-download the policy repo from the Hub before training (avoids silent hangs).",
    )
    p.add_argument(
        "--no_prefetch_policy",
        action="store_true",
        help="Disable prefetch (useful if policy_path is already a local directory).",
    )
    p.add_argument(
        "--hf_transfer",
        action="store_true",
        default=True,
        help="Use hf_transfer for faster Hub downloads (requires hf-transfer installed).",
    )
    p.add_argument("--no_hf_transfer", action="store_true")
    p.add_argument(
        "--prefetch_dir",
        type=str,
        default=None,
        help="Where to place a local copy of the policy weights (default: outputs/hf_prefetch_<repo>).",
    )
    p.add_argument(
        "--force_prefetch",
        action="store_true",
        help="Delete the prefetch_dir first, then download again (useful if a previous download was interrupted).",
    )
    p.add_argument("--steps", type=int, default=3_000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_freq", type=int, default=10, help="How often to log training metrics (in steps).")
    p.add_argument(
        "--eval_freq",
        type=int,
        default=0,
        help="How often to run eval (0 disables). For real-world datasets (no env), eval is typically off.",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda/cpu (LeRobot will auto-fallback if unavailable)")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--action_mode", type=str, default="auto")
    p.add_argument("--max_action_dim", type=int, default=20)
    p.add_argument("--train_soft_prompts", action="store_true", default=True)
    p.add_argument("--no_train_soft_prompts", action="store_true")
    p.add_argument("--train_policy_transformer", action="store_true", default=True)
    p.add_argument("--no_train_policy_transformer", action="store_true")
    p.add_argument("--freeze_vision_encoder", action="store_true", default=False)
    p.add_argument("--freeze_language_encoder", action="store_true", default=False)
    p.add_argument("--output_dir", type=str, default="outputs/xvla_raw")
    p.add_argument("--resume", action="store_true", help="Resume from an existing run at output_dir")
    p.add_argument("--no_save_checkpoint", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    if not args.skip_convert:
        # If the dataset already exists, default to reusing it unless the user explicitly requests overwrite.
        if dataset_dir.exists() and not args.overwrite_dataset:
            info_path = dataset_dir / "meta" / "info.json"
            if not info_path.exists():
                raise FileExistsError(
                    f"--dataset_dir already exists but does not look like a LeRobotDataset: {dataset_dir}\n"
                    "Either delete it manually, or rerun with --overwrite_dataset."
                )
            info = json.loads(info_path.read_text(encoding="utf-8"))
            features = set((info.get("features") or {}).keys())
            expected_image_key = f"observation.images.{args.camera_key}"
            required = {expected_image_key, "observation.state", "action"}
            if not required.issubset(features):
                raise FileExistsError(
                    f"--dataset_dir already exists but its features don't match this run.\n"
                    f"- dataset_dir: {dataset_dir}\n"
                    f"- missing required keys: {sorted(required - features)}\n"
                    "Fix: rerun with --overwrite_dataset (recommended), or point --dataset_dir to a new folder."
                )

            print(f"Dataset already exists at {dataset_dir}; skipping conversion.")
        else:
            convert(
                raw_data_dir=raw_data_dir,
                out_dir=dataset_dir,
                repo_id=args.dataset_repo_id,
                fps=args.fps,
                camera_key=args.camera_key,
                overwrite=args.overwrite_dataset,
            )

    # Prefetch the X-VLA policy repo so `lerobot-train` doesn't appear to hang while downloading large weights.
    policy_path = args.policy_path
    do_prefetch = bool(args.prefetch_policy) and not bool(args.no_prefetch_policy)
    use_hf_transfer = bool(args.hf_transfer) and not bool(args.no_hf_transfer)
    is_local_dir = Path(policy_path).expanduser().resolve().is_dir()
    if do_prefetch and not is_local_dir:
        try:
            import os

            if args.prefetch_dir is None:
                # Keep a stable default path for the common case.
                if str(policy_path) == "lerobot/xvla-base":
                    prefetch_dir = Path("outputs") / "hf_prefetch_xvla_base"
                else:
                    safe_name = str(policy_path).replace("/", "_")
                    prefetch_dir = Path("outputs") / f"hf_prefetch_{safe_name}"
            else:
                prefetch_dir = Path(args.prefetch_dir)
            prefetch_dir = prefetch_dir.expanduser().resolve()

            if args.force_prefetch and prefetch_dir.exists():
                shutil.rmtree(prefetch_dir)

            model_file = prefetch_dir / "model.safetensors"
            if model_file.exists():
                print(f"Prefetch already present: {model_file}")
            else:
                env = os.environ.copy()
                if use_hf_transfer:
                    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
                env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

                print(f"Prefetching policy repo with progress: {policy_path}")
                print(f"→ local dir: {prefetch_dir}")
                # Use huggingface-cli so progress is visible in the terminal.
                cmd_dl = [
                    "huggingface-cli",
                    "download",
                    policy_path,
                    "--repo-type",
                    "model",
                    "--revision",
                    "main",
                    "--include",
                    "*.safetensors",
                    "config.json",
                    "*.json",
                    "--local-dir",
                    str(prefetch_dir),
                ]
                subprocess.run(cmd_dl, check=True, env=env)

            # Point training at the local folder (XVLAPolicy.from_pretrained expects model.safetensors there).
            if (prefetch_dir / "model.safetensors").exists():
                policy_path = str(prefetch_dir)
            else:
                print(
                    "WARNING: prefetch directory does not contain model.safetensors; "
                    "continuing with the original policy_path."
                )
        except Exception as e:
            print(f"WARNING: prefetch failed ({e}). Continuing; lerobot-train may download during startup.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.resume:
        output_dir = _unique_out_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_soft_prompts = bool(args.train_soft_prompts) and not bool(args.no_train_soft_prompts)
    train_policy_transformer = bool(args.train_policy_transformer) and not bool(
        args.no_train_policy_transformer
    )

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={str(dataset_dir)}",
        f"--output_dir={str(output_dir)}",
        f"--resume={str(bool(args.resume)).lower()}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--eval_freq={args.eval_freq}",
        f"--policy.path={policy_path}",
        "--policy.push_to_hub=false",
        f"--policy.device={args.device}",
        f"--policy.dtype={args.dtype}",
        f"--policy.action_mode={args.action_mode}",
        f"--policy.max_action_dim={args.max_action_dim}",
        f"--policy.train_soft_prompts={str(train_soft_prompts).lower()}",
        f"--policy.train_policy_transformer={str(train_policy_transformer).lower()}",
        f"--policy.freeze_vision_encoder={str(bool(args.freeze_vision_encoder)).lower()}",
        f"--policy.freeze_language_encoder={str(bool(args.freeze_language_encoder)).lower()}",
    ]

    if args.no_save_checkpoint:
        cmd.append("--save_checkpoint=false")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done. Outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


