"""
XVLA v1: 2-phase embodiment adaptation via a new soft prompt + small action projection heads.

Phase 1 (alignment / "translation"):
- Create or reuse the local LeRobotDataset converted from this repo's `raw_data/`
- Load the pretrained XVLA backbone
- Initialize a *new random* domain-specific soft prompt (and keep only the prompt + small domain-specific
  action projection heads trainable)
- Freeze the entire pretrained backbone
- Train only: soft prompt + domain-specific action encoder/decoder for the chosen domain_id

Phase 2 (joint fine-tuning):
- Unfreeze the full backbone
- Train everything jointly, while using reduced learning rates for:
  - VLM parameters (handled as 1/10 LR by LeRobot's XVLAAdamW param groups)
  - Soft prompts (via `--stage2_soft_prompt_lr_scale < 1.0`)

Example:

  python train_scripts/lerobot-xvla/xvla-v1.py \
    --raw_data_dir raw_data \
    --dataset_dir lerobot_datasets/grasp_raw \
    --dataset_repo_id grasp_raw \
    --policy_path lerobot/xvla-base \
    --output_dir outputs/xvla_v1_run \
    --domain_id 29 \
    --stage1_steps 500 \
    --stage2_steps 2500 \
    --batch_size 1 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from convert_raw_data_to_lerobot_dataset import convert


def _unique_out_dir(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.parent / f"{path.name}_{stamp}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _reinit_soft_prompt_row(policy: Any, *, domain_id: int, std: float = 0.02) -> None:
    hub = getattr(getattr(getattr(policy, "model", None), "transformer", None), "soft_prompt_hub", None)
    if hub is None:
        raise RuntimeError("Policy transformer has no `soft_prompt_hub` (len_soft_prompts may be 0).")
    if not hasattr(hub, "weight"):
        raise RuntimeError("Unexpected soft_prompt_hub type; expected nn.Embedding-like with `.weight`.")
    if domain_id < 0 or domain_id >= hub.weight.shape[0]:
        raise ValueError(f"domain_id={domain_id} out of range for soft_prompt_hub with {hub.weight.shape[0]} domains")
    with torch.no_grad():
        nn.init.normal_(hub.weight[domain_id], std=std)


def _reinit_domain_aware_linear_row(module: Any, *, domain_id: int) -> None:
    """
    Reinitialize a DomainAwareLinear row (one domain) without touching other domains.
    """
    if not hasattr(module, "fc") or not hasattr(module, "bias"):
        raise TypeError("Expected DomainAwareLinear-like module with `.fc` and `.bias` embeddings.")
    if not hasattr(module, "input_size") or not hasattr(module, "output_size"):
        raise TypeError("Expected DomainAwareLinear-like module with `.input_size` and `.output_size`.")
    input_size = int(module.input_size)
    output_size = int(module.output_size)

    fc_w = module.fc.weight
    b_w = module.bias.weight
    if domain_id < 0 or domain_id >= fc_w.shape[0] or domain_id >= b_w.shape[0]:
        raise ValueError(f"domain_id={domain_id} out of range for DomainAwareLinear with {fc_w.shape[0]} domains")

    with torch.no_grad():
        w = fc_w[domain_id].view(input_size, output_size)
        nn.init.xavier_uniform_(w)
        nn.init.zeros_(b_w[domain_id])


def _mark_trainable(policy: Any, *, trainable_name_substrings: Tuple[str, ...]) -> int:
    """
    Freeze everything, then unfreeze any parameter whose name contains one of the provided substrings.
    Returns the count of trainable parameters.
    """
    for p in policy.parameters():
        p.requires_grad = False
    for name, p in policy.named_parameters():
        if any(s in name for s in trainable_name_substrings):
            p.requires_grad = True
    return sum(int(p.requires_grad) for p in policy.parameters())


def _count_trainable(policy: Any) -> Tuple[int, int]:
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    return n_trainable, n_total


def _save_run_metadata(save_dir: Path, payload: Dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "xvla_v1_run.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _save_policy_and_processors(save_dir: Path, policy: Any, preprocessor: Any, postprocessor: Any) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(policy, "save_pretrained"):
        policy.save_pretrained(save_dir)
    else:
        raise RuntimeError("Policy does not implement save_pretrained().")

    # Save processors with canonical filenames expected by LeRobot.
    if hasattr(preprocessor, "save_pretrained"):
        preprocessor.save_pretrained(save_dir, config_filename="policy_preprocessor.json")
    if hasattr(postprocessor, "save_pretrained"):
        postprocessor.save_pretrained(save_dir, config_filename="policy_postprocessor.json")


def _make_dataloader(dataset: Any, *, batch_size: int, num_workers: int, seed: int, pin_memory: bool) -> torch.utils.data.DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _train_steps(
    *,
    policy: Any,
    preprocessor: Any,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    steps: int,
    grad_accum: int,
    grad_clip_norm: float,
    use_amp: bool,
    log_every: int,
) -> None:
    if steps <= 0:
        return
    if grad_accum <= 0:
        raise ValueError("--grad_accum must be >= 1")

    device = next(policy.parameters()).device
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    policy.train()
    optimizer.zero_grad(set_to_none=True)

    dl_iter = iter(dataloader)
    for step_idx in range(1, steps + 1):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        batch = preprocessor(batch)

        with amp_ctx:
            loss, log_dict = policy.forward(batch)
            loss_to_backprop = loss / grad_accum
        loss_to_backprop.backward()

        if step_idx % grad_accum == 0:
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        if log_every > 0 and step_idx % log_every == 0:
            extra = ""
            if isinstance(log_dict, dict) and "loss" in log_dict:
                extra = f" (sub_losses={{{', '.join([f'{k}:{v:.4f}' for k,v in log_dict.items() if k!='loss'])}}})"
            print(f"[step {step_idx:06d}/{steps}] loss={float(loss):.6f}{extra}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XVLA v1: 2-phase soft-prompt adaptation on Grasp-VLA raw_data")

    # Conversion
    p.add_argument("--raw_data_dir", type=str, default="raw_data")
    p.add_argument("--dataset_dir", type=str, default="lerobot_datasets/grasp_raw")
    p.add_argument("--dataset_repo_id", type=str, default="grasp_raw")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--camera_key", type=str, default="image")
    p.add_argument("--overwrite_dataset", action="store_true")
    p.add_argument("--skip_convert", action="store_true")

    # Model / run
    p.add_argument("--policy_path", type=str, default="lerobot/xvla-base")
    p.add_argument("--prefetch_policy", action="store_true", default=True)
    p.add_argument("--no_prefetch_policy", action="store_true")
    p.add_argument("--hf_transfer", action="store_true", default=True)
    p.add_argument("--no_hf_transfer", action="store_true")

    p.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--action_mode", type=str, default="auto")
    p.add_argument("--max_action_dim", type=int, default=20)

    p.add_argument("--domain_id", type=int, default=None, help="Domain id used for the new prompt/head adaptation")
    p.add_argument("--seed", type=int, default=0)

    # Training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip_norm", type=float, default=10.0)
    p.add_argument("--use_amp", action="store_true", help="Enable autocast on CUDA (in addition to dtype setting)")
    p.add_argument("--log_every", type=int, default=20)

    p.add_argument("--stage1_steps", type=int, default=500)
    p.add_argument("--stage1_lr", type=float, default=1e-4)

    p.add_argument("--stage2_steps", type=int, default=2500)
    p.add_argument("--stage2_lr", type=float, default=1e-4)
    p.add_argument(
        "--stage2_soft_prompt_lr_scale",
        type=float,
        default=0.1,
        help="Soft prompt LR = stage2_lr * scale. (<1.0 means reduced LR during joint training)",
    )

    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer_betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--optimizer_eps", type=float, default=1e-8)

    # Saving
    p.add_argument("--output_dir", type=str, default="outputs/xvla_v1")
    p.add_argument("--resume", action="store_true", help="(reserved) kept for interface symmetry with v0")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _set_seed(int(args.seed))

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    if not args.skip_convert:
        if dataset_dir.exists() and not args.overwrite_dataset:
            print(
                f"Dataset directory already exists: {dataset_dir}\n"
                "Skipping conversion. (Pass --overwrite_dataset to delete & recreate, or --skip_convert to silence this.)"
            )
        else:
            convert(
                raw_data_dir=raw_data_dir,
                out_dir=dataset_dir,
                repo_id=args.dataset_repo_id,
                fps=args.fps,
                camera_key=args.camera_key,
                overwrite=args.overwrite_dataset,
            )

    # Resolve device (and allow auto-fallback to cpu if cuda isn't available).
    device = _resolve_device(args.device)
    pin_memory = device.type == "cuda"

    # Prefetch the policy directory (optional, but avoids long silent downloads).
    policy_path = args.policy_path
    do_prefetch = bool(args.prefetch_policy) and not bool(args.no_prefetch_policy)
    use_hf_transfer = bool(args.hf_transfer) and not bool(args.no_hf_transfer)
    if do_prefetch and not Path(policy_path).expanduser().resolve().is_dir():
        try:
            import os

            if use_hf_transfer:
                os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

            from huggingface_hub import snapshot_download

            print(f"Prefetching policy repo: {policy_path}")
            policy_path = snapshot_download(repo_id=policy_path, repo_type="model", resume_download=True)
            print(f"Policy cached at: {policy_path}")
        except Exception as e:
            print(f"WARNING: prefetch failed ({e}). Continuing; weights may download during startup.")

    # Load dataset (from disk).
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=args.dataset_repo_id, root=dataset_dir, download_videos=False)
    dl = _make_dataloader(
        ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        pin_memory=pin_memory,
    )

    # Load policy config from the pretrained policy and override a few run-time knobs.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    from lerobot.optim.optimizers import XVLAAdamWConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cli_overrides = [
        f"--device={device.type}",
        f"--dtype={args.dtype}",
        f"--action_mode={args.action_mode}",
        f"--max_action_dim={int(args.max_action_dim)}",
        # Ensure prompts exist & are trainable by default (we will still control requires_grad explicitly)
        "--train_soft_prompts=true",
    ]
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
    policy_cfg.pretrained_path = Path(str(policy_path))

    # Populate feature specs from the dataset metadata and load weights.
    policy = make_policy(cfg=policy_cfg, ds_meta=ds.meta, rename_map={})
    policy.to(device)

    # Decide which domain_id we are adapting.
    num_domains = int(getattr(policy.config, "num_domains", 0) or 0)
    if num_domains <= 0:
        raise RuntimeError("Could not determine XVLA num_domains from policy.config")
    domain_id = int(args.domain_id) if args.domain_id is not None else (num_domains - 1)
    if domain_id < 0 or domain_id >= num_domains:
        raise ValueError(f"--domain_id must be in [0, {num_domains-1}] but got {domain_id}")

    # Build processors (loaded from the pretrained policy dir, with overrides for our dataset + domain_id).
    pre_overrides: Dict[str, Any] = {
        "device_processor": {"device": device.type},
        "xvla_add_domain_id": {"domain_id": domain_id},
        "normalizer_processor": {
            "stats": ds.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": {}},
    }
    post_overrides: Dict[str, Any] = {
        "unnormalizer_processor": {
            "stats": ds.meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        }
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(policy_cfg.pretrained_path),
        dataset_stats=ds.meta.stats,
        preprocessor_overrides=pre_overrides,
        postprocessor_overrides=post_overrides,
    )

    # Output directories
    out_root = _unique_out_dir(Path(args.output_dir).expanduser().resolve())
    stage1_dir = out_root / "stage1"
    stage2_dir = out_root / "stage2"

    # -------------------------
    # Phase 1: prompt + heads only
    # -------------------------
    print(f"Phase 1: reinitializing domain={domain_id} soft prompt + training only prompt + action heads (frozen backbone)")

    # Re-init the *new* soft prompt row for our robot/scenario.
    _reinit_soft_prompt_row(policy, domain_id=domain_id, std=0.02)

    # Also (optionally) treat domain-specific action encoder/decoder as new "small action projection heads"
    # by reinitializing only the selected domain row.
    transformer = getattr(getattr(policy, "model", None), "transformer", None)
    if transformer is None:
        raise RuntimeError("Unexpected policy structure: missing policy.model.transformer")
    _reinit_domain_aware_linear_row(transformer.action_encoder, domain_id=domain_id)
    _reinit_domain_aware_linear_row(transformer.action_decoder, domain_id=domain_id)
    if getattr(transformer, "use_hetero_proj", False):
        # These are domain-conditioned projection heads when enabled.
        _reinit_domain_aware_linear_row(transformer.vlm_proj, domain_id=domain_id)
        _reinit_domain_aware_linear_row(transformer.aux_visual_proj, domain_id=domain_id)

    # Freeze everything except the prompt + action heads.
    _mark_trainable(
        policy,
        trainable_name_substrings=(
            "transformer.soft_prompt_hub",
            "transformer.action_encoder",
            "transformer.action_decoder",
            "transformer.vlm_proj",  # only matters if use_hetero_proj=True (DomainAwareLinear)
            "transformer.aux_visual_proj",  # only matters if use_hetero_proj=True (DomainAwareLinear)
        ),
    )
    n_trainable, n_total = _count_trainable(policy)
    print(f"Trainable params (phase 1): {n_trainable:,} / {n_total:,}")

    opt1 = XVLAAdamWConfig(
        lr=float(args.stage1_lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        betas=tuple(float(x) for x in args.optimizer_betas),
        eps=float(args.optimizer_eps),
        soft_prompt_lr_scale=1.0,
        soft_prompt_warmup_lr_scale=None,
    )
    optimizer1 = opt1.build(policy.get_optim_params())
    scheduler1 = None  # Keep phase 1 simple by default.

    _train_steps(
        policy=policy,
        preprocessor=preprocessor,
        dataloader=dl,
        optimizer=optimizer1,
        scheduler=scheduler1,
        steps=int(args.stage1_steps),
        grad_accum=int(args.grad_accum),
        grad_clip_norm=float(args.grad_clip_norm),
        use_amp=bool(args.use_amp),
        log_every=int(args.log_every),
    )

    _save_policy_and_processors(stage1_dir, policy, preprocessor, postprocessor)
    _save_run_metadata(
        stage1_dir,
        {
            "phase": 1,
            "domain_id": domain_id,
            "stage1_steps": int(args.stage1_steps),
            "stage1_lr": float(args.stage1_lr),
            "trainable_params": n_trainable,
            "total_params": n_total,
        },
    )

    # -------------------------
    # Phase 2: unfreeze and finetune jointly
    # -------------------------
    print("Phase 2: unfreezing backbone and training everything jointly (reduced LR for VLM + soft prompts)")

    for p in policy.parameters():
        p.requires_grad = True
    n_trainable2, n_total2 = _count_trainable(policy)
    print(f"Trainable params (phase 2): {n_trainable2:,} / {n_total2:,}")

    opt2 = XVLAAdamWConfig(
        lr=float(args.stage2_lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        betas=tuple(float(x) for x in args.optimizer_betas),
        eps=float(args.optimizer_eps),
        soft_prompt_lr_scale=float(args.stage2_soft_prompt_lr_scale),
        soft_prompt_warmup_lr_scale=None,
    )
    optimizer2 = opt2.build(policy.get_optim_params())

    # Reuse XVLA's cosine decay + warmup preset, but adapt it to our stage2_steps.
    sched_cfg = None
    if hasattr(policy.config, "get_scheduler_preset"):
        try:
            sched_cfg = policy.config.get_scheduler_preset()
        except Exception:
            sched_cfg = None

    scheduler2 = None
    if isinstance(sched_cfg, CosineDecayWithWarmupSchedulerConfig):
        # Ensure scheduler peak_lr aligns with stage2_lr
        sched_cfg = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=int(sched_cfg.num_warmup_steps),
            num_decay_steps=int(sched_cfg.num_decay_steps),
            peak_lr=float(args.stage2_lr),
            decay_lr=float(getattr(sched_cfg, "decay_lr", 2.5e-6)),
        )
        scheduler2 = sched_cfg.build(optimizer2, num_training_steps=int(args.stage2_steps))

    _train_steps(
        policy=policy,
        preprocessor=preprocessor,
        dataloader=dl,
        optimizer=optimizer2,
        scheduler=scheduler2,
        steps=int(args.stage2_steps),
        grad_accum=int(args.grad_accum),
        grad_clip_norm=float(args.grad_clip_norm),
        use_amp=bool(args.use_amp),
        log_every=int(args.log_every),
    )

    _save_policy_and_processors(stage2_dir, policy, preprocessor, postprocessor)
    _save_run_metadata(
        stage2_dir,
        {
            "phase": 2,
            "domain_id": domain_id,
            "stage2_steps": int(args.stage2_steps),
            "stage2_lr": float(args.stage2_lr),
            "stage2_soft_prompt_lr_scale": float(args.stage2_soft_prompt_lr_scale),
            "optimizer2": asdict(opt2),
        },
    )

    print(f"Done. Saved stage1 -> {stage1_dir} | stage2 -> {stage2_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


