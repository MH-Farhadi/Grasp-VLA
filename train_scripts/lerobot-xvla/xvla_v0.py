"""
End-to-end helper (v0): convert this repo's `raw_data/` into a LeRobotDataset, then fine-tune LeRobot's X-VLA.

This script is optional convenience glue around:
  1) `convert_raw_data_to_lerobot_dataset.py`
  2) `lerobot-train --policy.path=lerobot/xvla-base ...`

Example:

  python train_scripts/lerobot-xvla/xvla_v0.py \
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
    p.add_argument("--steps", type=int, default=3_000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
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
    if do_prefetch and not Path(policy_path).expanduser().resolve().is_dir():
        try:
            import os

            if use_hf_transfer:
                os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            # Force progress bars on (some environments disable them).
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

            from huggingface_hub import snapshot_download

            print(f"Prefetching policy repo: {policy_path}")
            local_dir = snapshot_download(repo_id=policy_path, repo_type="model", resume_download=True)
            print(f"Policy cached at: {local_dir}")
            policy_path = local_dir
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


