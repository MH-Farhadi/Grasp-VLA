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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_episode_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("episode_")


def _is_session_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("session_")


def _list_raw_episode_dirs(raw_data_dir: Path) -> List[Path]:
    """
    Supports:
    - raw_data/episode_*
    - raw_data/session_*/episode_*
    - raw_data being a session dir itself
    """
    raw_data_dir = raw_data_dir.expanduser().resolve()
    episode_dirs: List[Path] = []

    # Episodes directly under the provided directory (old layout, or when passing a session dir)
    episode_dirs.extend(sorted([p for p in raw_data_dir.iterdir() if _is_episode_dir(p)]))

    # Episodes nested under session_* dirs (new multi-session layout)
    for sdir in sorted([p for p in raw_data_dir.iterdir() if _is_session_dir(p)]):
        episode_dirs.extend(sorted([p for p in sdir.iterdir() if _is_episode_dir(p)]))

    # De-dupe while preserving order
    seen: set[Path] = set()
    out: List[Path] = []
    for p in episode_dirs:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _infer_log_rate_hz_from_metadata(raw_data_dir: Path) -> int:
    """
    Infer FPS from raw_data by reading metadata.json["config"]["log_rate_hz"].
    Enforces that log_rate_hz is constant across all episodes discovered.
    """
    episode_dirs = _list_raw_episode_dirs(raw_data_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found under {raw_data_dir}")

    rates: set[int] = set()
    missing = 0
    for ep_dir in episode_dirs:
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            missing += 1
            continue
        try:
            meta = _read_json(meta_path)
            cfg = meta.get("config") or {}
            lr = cfg.get("log_rate_hz")
            if lr is None:
                missing += 1
                continue
            rates.add(int(lr))
        except Exception:
            missing += 1

    if not rates:
        raise RuntimeError(
            f"Could not infer fps from metadata.json (missing/invalid in {missing}/{len(episode_dirs)} episodes)."
        )
    if len(rates) != 1:
        raise RuntimeError(f"Mixed metadata config.log_rate_hz values across dataset: {sorted(rates)}")
    return next(iter(rates))


def _load_policy_contract(policy_dir: Path) -> Dict[str, Any]:
    """
    Minimal "data contract" extraction from a local XVLA policy directory:
    - expected feature keys + shapes (from config.json)
    - expected task key (from policy_preprocessor.json, if present)
    - chunking horizons
    """
    policy_dir = policy_dir.expanduser().resolve()
    cfg_path = policy_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing policy config.json at: {cfg_path}")
    cfg = _read_json(cfg_path)

    input_features = cfg.get("input_features") or {}
    output_features = cfg.get("output_features") or {}
    chunk_size = int(cfg.get("chunk_size") or 1)
    n_action_steps = int(cfg.get("n_action_steps") or 1)

    task_key = "task"
    preproc_path = policy_dir / "policy_preprocessor.json"
    if preproc_path.exists():
        try:
            pre = _read_json(preproc_path)
            for step in pre.get("steps") or []:
                if (step.get("registry_name") or "") == "tokenizer_processor":
                    task_key = str((step.get("config") or {}).get("task_key") or task_key)
        except Exception:
            # Best-effort: contract still mostly comes from config.json
            task_key = "task"

    return {
        "policy_dir": str(policy_dir),
        "task_key": task_key,
        "input_features": input_features,
        "output_features": output_features,
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
    }


def _read_dataset_info(dataset_dir: Path) -> Dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset meta/info.json at: {info_path}")
    return _read_json(info_path)


def _validate_dataset_info_against_contract(
    dataset_info: Dict[str, Any],
    *,
    contract: Dict[str, Any],
    expected_fps: Optional[int],
    action_dim_override: Optional[int],
) -> None:
    ds_features = dataset_info.get("features") or {}
    required = set((contract.get("input_features") or {}).keys()) | set((contract.get("output_features") or {}).keys())

    missing = sorted([k for k in required if k not in ds_features])
    if missing:
        raise RuntimeError(
            "Dataset is missing required feature keys expected by the policy:\n"
            + "\n".join([f"  - {k}" for k in missing])
            + "\n\nFix: re-run with --overwrite_dataset to rebuild the LeRobotDataset with the correct schema."
        )

    if expected_fps is not None:
        ds_fps = dataset_info.get("fps")
        if ds_fps is None or int(ds_fps) != int(expected_fps):
            raise RuntimeError(f"Dataset fps={ds_fps} does not match expected fps={expected_fps}")

    def _expect_chw(key: str, spec: Dict[str, Any]) -> None:
        chw = spec.get("shape") or []
        if len(chw) != 3:
            raise RuntimeError(f"Policy expects {key} with CHW shape, got shape={chw}")
        c, h, w = (int(chw[0]), int(chw[1]), int(chw[2]))
        ds_spec = ds_features.get(key) or {}
        ds_dtype = ds_spec.get("dtype")
        if ds_dtype != "image":
            raise RuntimeError(f"{key}: dataset dtype={ds_dtype} but expected dtype='image'")
        ds_shape = ds_spec.get("shape") or []
        if len(ds_shape) != 3:
            raise RuntimeError(f"Dataset feature {key} has unexpected shape={ds_shape} (expected HWC)")
        dh, dw, dc = (int(ds_shape[0]), int(ds_shape[1]), int(ds_shape[2]))
        if dc != c:
            raise RuntimeError(f"{key}: dataset channels={dc} but policy expects channels={c}")
        if dh != h or dw != w:
            raise RuntimeError(f"{key}: dataset HWC={(dh, dw, dc)} but policy expects CHW={(c, h, w)}")

    def _expect_1d(key: str, spec: Dict[str, Any], *, override_dim: Optional[int] = None) -> None:
        shp = spec.get("shape") or []
        if len(shp) != 1:
            raise RuntimeError(f"Policy expects {key} with 1D shape, got shape={shp}")
        expected_dim = int(override_dim) if override_dim is not None else int(shp[0])
        ds_spec = ds_features.get(key) or {}
        ds_dtype = ds_spec.get("dtype")
        if ds_dtype != "float32":
            raise RuntimeError(f"{key}: dataset dtype={ds_dtype} but expected dtype='float32'")
        ds_shape = ds_spec.get("shape") or []
        if len(ds_shape) != 1 or int(ds_shape[0]) != expected_dim:
            raise RuntimeError(f"{key}: dataset shape={ds_shape} but policy expects ({expected_dim},)")

    # Validate shapes for each required feature.
    for k, spec in (contract.get("input_features") or {}).items():
        ftype = (spec.get("type") or "").upper()
        if ftype == "VISUAL":
            _expect_chw(k, spec)
        elif ftype == "STATE":
            _expect_1d(k, spec)

    for k, spec in (contract.get("output_features") or {}).items():
        ftype = (spec.get("type") or "").upper()
        if ftype == "ACTION":
            _expect_1d(k, spec, override_dim=action_dim_override)


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


def _make_eval_dataloader(
    dataset: Any, *, batch_size: int, num_workers: int, pin_memory: bool
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _split_episode_indices(*, num_episodes: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    if num_episodes <= 0:
        return [], []
    if val_split <= 0:
        return list(range(num_episodes)), []
    if val_split >= 1:
        return [], list(range(num_episodes))

    n_val = int(round(float(num_episodes) * float(val_split)))
    n_val = max(1, min(int(num_episodes) - 1, int(n_val)))

    rng = random.Random(int(seed))
    eps = list(range(int(num_episodes)))
    rng.shuffle(eps)
    val_eps = sorted(eps[:n_val])
    train_eps = sorted(eps[n_val:])
    return train_eps, val_eps


@torch.no_grad()
def _eval_loss(
    *,
    policy: Any,
    preprocessor: Any,
    dataloader: torch.utils.data.DataLoader,
    max_batches: Optional[int],
    use_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    policy.eval()
    device = next(policy.parameters()).device
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    total = 0.0
    n = 0
    sums: Dict[str, float] = {}

    max_batches_i = None if (max_batches is None or int(max_batches) <= 0) else int(max_batches)
    for b_idx, batch in enumerate(dataloader):
        if max_batches_i is not None and b_idx >= max_batches_i:
            break
        batch = preprocessor(batch)
        with amp_ctx:
            loss, log_dict = policy.forward(batch)
        total += float(loss.detach().item())
        n += 1
        if isinstance(log_dict, dict):
            for k, v in log_dict.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                sums[k] = sums.get(k, 0.0) + fv

    if n <= 0:
        return float("nan"), {}
    avg = total / n
    avg_logs = {k: v / n for k, v in sums.items()}
    avg_logs.setdefault("loss", avg)
    return avg, avg_logs


def _train_steps(
    *,
    policy: Any,
    preprocessor: Any,
    dataloader: torch.utils.data.DataLoader,
    eval_dataloader: Optional[torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    steps: int,
    grad_accum: int,
    grad_clip_norm: float,
    use_amp: bool,
    log_every: int,
    eval_every_steps: int,
    eval_every_epochs: int,
    eval_max_batches: Optional[int],
    eval_at_end: bool,
    tag: str,
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

    steps_per_epoch = 0
    try:
        steps_per_epoch = int(len(dataloader))
    except Exception:
        steps_per_epoch = 0

    epoch_idx = 0
    epoch_loss_sum = 0.0
    epoch_loss_n = 0
    epoch_sums: Dict[str, float] = {}
    last_eval_step = 0

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

        # Track running stats inside the current epoch window (epoch := one full pass through the dataloader).
        epoch_loss_sum += float(loss.detach().item())
        epoch_loss_n += 1
        if isinstance(log_dict, dict):
            for k, v in log_dict.items():
                if k == "loss":
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                epoch_sums[k] = epoch_sums.get(k, 0.0) + fv

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

        # Epoch boundary reporting.
        is_epoch_end = bool(steps_per_epoch > 0 and (step_idx % steps_per_epoch == 0))
        if is_epoch_end:
            epoch_idx += 1
            train_epoch_loss = (epoch_loss_sum / max(1, epoch_loss_n)) if epoch_loss_n > 0 else float("nan")
            epoch_extra = ""
            if epoch_sums:
                avg_sub = {k: v / max(1, epoch_loss_n) for k, v in epoch_sums.items()}
                epoch_extra = " (sub_losses={" + ", ".join([f"{k}:{v:.4f}" for k, v in avg_sub.items()]) + "})"
            print(
                f"[{tag}] epoch {epoch_idx} end: train_loss={train_epoch_loss:.6f}{epoch_extra}"
                + (f" (steps_per_epoch={steps_per_epoch})" if steps_per_epoch > 0 else "")
            )
            epoch_loss_sum = 0.0
            epoch_loss_n = 0
            epoch_sums = {}

            if eval_dataloader is not None and int(eval_every_epochs) > 0 and (epoch_idx % int(eval_every_epochs) == 0):
                val_loss, val_logs = _eval_loss(
                    policy=policy,
                    preprocessor=preprocessor,
                    dataloader=eval_dataloader,
                    max_batches=eval_max_batches,
                    use_amp=use_amp,
                )
                sub = {k: v for k, v in val_logs.items() if k != "loss"}
                extra = " (sub_losses={" + ", ".join([f"{k}:{v:.4f}" for k, v in sub.items()]) + "})" if sub else ""
                print(f"[{tag}] epoch {epoch_idx} val_loss={val_loss:.6f}{extra}")
                last_eval_step = step_idx
                policy.train()

        # Step-based evaluation (useful when training for < 1 full epoch).
        if (
            eval_dataloader is not None
            and int(eval_every_steps) > 0
            and (step_idx % int(eval_every_steps) == 0)
            and step_idx != last_eval_step
        ):
            val_loss, val_logs = _eval_loss(
                policy=policy,
                preprocessor=preprocessor,
                dataloader=eval_dataloader,
                max_batches=eval_max_batches,
                use_amp=use_amp,
            )
            sub = {k: v for k, v in val_logs.items() if k != "loss"}
            extra = " (sub_losses={" + ", ".join([f"{k}:{v:.4f}" for k, v in sub.items()]) + "})" if sub else ""
            approx_epoch = (step_idx / steps_per_epoch) if steps_per_epoch > 0 else 0.0
            print(f"[{tag}] step {step_idx}/{steps} (~epoch {approx_epoch:.3f}) val_loss={val_loss:.6f}{extra}")
            last_eval_step = step_idx
            policy.train()

    if steps_per_epoch > 0:
        approx_epochs = steps / steps_per_epoch
        print(f"[{tag}] done: {steps} steps (~{approx_epochs:.3f} epochs)")

    if eval_dataloader is not None and bool(eval_at_end) and last_eval_step != int(steps):
        val_loss, val_logs = _eval_loss(
            policy=policy,
            preprocessor=preprocessor,
            dataloader=eval_dataloader,
            max_batches=eval_max_batches,
            use_amp=use_amp,
        )
        sub = {k: v for k, v in val_logs.items() if k != "loss"}
        extra = " (sub_losses={" + ", ".join([f"{k}:{v:.4f}" for k, v in sub.items()]) + "})" if sub else ""
        print(f"[{tag}] final val_loss={val_loss:.6f}{extra}")
        policy.train()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XVLA v1: 2-phase soft-prompt adaptation on Grasp-VLA raw_data")

    # Conversion
    p.add_argument("--raw_data_dir", type=str, default="raw_data")
    p.add_argument("--dataset_dir", type=str, default="lerobot_datasets/grasp_raw")
    p.add_argument("--dataset_repo_id", type=str, default="grasp_raw")
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Dataset FPS. If omitted, inferred from raw_data metadata.json config.log_rate_hz (recommended).",
    )
    p.add_argument("--camera_key", type=str, default="image", help="Suffix for observation.images.<key> (XVLA expects 'image').")
    p.add_argument("--overwrite_dataset", action="store_true")
    p.add_argument("--skip_convert", action="store_true")
    p.add_argument("--max_episodes", type=int, default=None, help="Limit number of episodes during conversion (debug/overfit).")
    p.add_argument(
        "--min_frames_per_episode",
        type=int,
        default=None,
        help="Drop episodes shorter than this many frames (if omitted, defaults to the policy chunk_size when available).",
    )
    p.add_argument(
        "--max_frames_per_episode", type=int, default=None, help="Limit frames per episode during conversion (debug/overfit)."
    )
    p.add_argument("--raw_quality_gate", action="store_true", default=True, help="Run strict raw log checks before conversion.")
    p.add_argument("--no_raw_quality_gate", action="store_true", help="Disable raw log checks (not recommended).")
    p.add_argument(
        "--drop_truncated",
        action="store_true",
        default=True,
        help="Skip episodes with episode_end.truncated=true (recommended for clean imitation).",
    )
    p.add_argument("--keep_truncated", action="store_true", help="Keep truncated episodes.")
    p.add_argument("--success_only", action="store_true", default=False, help="Only train on episodes with grasp_result.ok=true.")

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
    p.add_argument("--smoke_test", action="store_true", default=True, help="Run a 1-batch contract/preprocessor/policy forward pass check.")
    p.add_argument("--no_smoke_test", action="store_true", help="Disable the initial smoke test.")

    # Evaluation / validation
    p.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Hold out this fraction of episodes for validation (0 disables; split is by episode_index).",
    )
    p.add_argument("--val_seed", type=int, default=None, help="Seed for selecting validation episodes (default: --seed).")
    p.add_argument("--eval_only", action="store_true", help="Only run evaluation (no training / no reinit).")
    p.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for evaluation (defaults to --batch_size).")
    p.add_argument("--eval_every_steps", type=int, default=0, help="Run validation every N training steps (0 disables).")
    p.add_argument(
        "--eval_every_epochs",
        type=int,
        default=1,
        help="Run validation every N epochs (epoch := one full pass through the train dataloader).",
    )
    p.add_argument(
        "--eval_max_batches",
        type=int,
        default=200,
        help="Limit validation to this many batches for speed (0 = full validation set).",
    )
    p.add_argument("--eval_at_end", action="store_true", default=True, help="Run validation at the end of each training phase.")
    p.add_argument("--no_eval_at_end", action="store_true", help="Disable end-of-phase validation.")

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

    device = _resolve_device(args.device)
    pin_memory = device.type == "cuda"

    # Prefetch the policy directory early (so we can validate the "data contract" before conversion).
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

    policy_dir = Path(str(policy_path)).expanduser().resolve()
    policy_contract: Optional[Dict[str, Any]] = None
    if policy_dir.is_dir():
        try:
            policy_contract = _load_policy_contract(policy_dir)
            print(
                "Policy contract:"
                f" chunk_size={policy_contract['chunk_size']}"
                f" n_action_steps={policy_contract['n_action_steps']}"
                f" inputs={sorted(list((policy_contract['input_features'] or {}).keys()))}"
                f" outputs={sorted(list((policy_contract['output_features'] or {}).keys()))}"
            )
        except Exception as e:
            print(f"WARNING: could not load policy contract from {policy_dir} ({e}). Skipping contract checks.")
            policy_contract = None

    # Determine dataset FPS. Do NOT infer from wall-clock tick timestamps (t_ms); use metadata.json config.log_rate_hz.
    fps = int(args.fps) if args.fps is not None else _infer_log_rate_hz_from_metadata(raw_data_dir)
    if args.fps is None:
        print(f"Inferred fps={fps} from raw_data metadata.json config.log_rate_hz")

    # Derive conversion shapes from the policy contract when available (matches xvla-base expectations).
    image_wh: Tuple[int, int] = (256, 256)
    image2_wh: Optional[Tuple[int, int]] = (256, 256)
    empty0_wh: Optional[Tuple[int, int]] = (224, 224)
    state_dim = 8
    if policy_contract is not None:
        in_feats = policy_contract.get("input_features") or {}
        out_feats = policy_contract.get("output_features") or {}

        img_key = f"observation.images.{args.camera_key}"
        if img_key not in in_feats:
            expected_imgs = sorted([k for k in in_feats.keys() if k.startswith("observation.images.")])
            raise RuntimeError(
                f"Policy expects image key '{img_key}' but it is missing from config.json input_features.\n"
                f"Available image keys: {expected_imgs}\n"
                f"Fix by using --camera_key=image (or convert with a matching key + a rename_map)."
            )
        img_shape = (in_feats.get(img_key) or {}).get("shape") or []
        if len(img_shape) != 3:
            raise RuntimeError(f"Unexpected policy image shape for {img_key}: {img_shape} (expected CHW)")
        image_wh = (int(img_shape[2]), int(img_shape[1]))  # (W,H)

        if "observation.images.image2" in in_feats:
            s = (in_feats.get("observation.images.image2") or {}).get("shape") or []
            image2_wh = (int(s[2]), int(s[1])) if len(s) == 3 else None
        else:
            image2_wh = None

        if "observation.images.empty_camera_0" in in_feats:
            s = (in_feats.get("observation.images.empty_camera_0") or {}).get("shape") or []
            empty0_wh = (int(s[2]), int(s[1])) if len(s) == 3 else None
        else:
            empty0_wh = None

        state_shape = (in_feats.get("observation.state") or {}).get("shape") or []
        if len(state_shape) == 1:
            state_dim = int(state_shape[0])

        ckpt_action_shape = (out_feats.get("action") or {}).get("shape") or []
        if len(ckpt_action_shape) == 1 and int(ckpt_action_shape[0]) != int(args.max_action_dim):
            print(
                f"WARNING: policy checkpoint action dim={int(ckpt_action_shape[0])} but --max_action_dim={int(args.max_action_dim)}. "
                "Conversion will follow --max_action_dim."
            )

    raw_quality_gate = bool(args.raw_quality_gate) and not bool(args.no_raw_quality_gate)
    drop_truncated = bool(args.drop_truncated) and not bool(args.keep_truncated)
    min_frames_per_episode = args.min_frames_per_episode
    if min_frames_per_episode is None and policy_contract is not None:
        try:
            cs = int(policy_contract.get("chunk_size") or 0)
            if cs > 0:
                min_frames_per_episode = cs
        except Exception:
            min_frames_per_episode = None

    # Convert raw_data -> LeRobotDataset (optional).
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
                fps=int(fps),
                camera_key=args.camera_key,
                image_wh=image_wh,
                image2_wh=image2_wh,
                empty_camera_0_wh=empty0_wh,
                state_dim=int(state_dim),
                action_dim=int(args.max_action_dim),
                raw_quality_gate=raw_quality_gate,
                drop_truncated=drop_truncated,
                success_only=bool(args.success_only),
                max_episodes=args.max_episodes,
                min_frames_per_episode=min_frames_per_episode,
                max_frames_per_episode=args.max_frames_per_episode,
                overwrite=args.overwrite_dataset,
            )

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir} (did conversion run?)")

    # Fast fail: validate the dataset on disk matches the policy's expected feature keys/shapes.
    if policy_contract is not None:
        ds_info = _read_dataset_info(dataset_dir)
        _validate_dataset_info_against_contract(
            ds_info,
            contract=policy_contract,
            expected_fps=int(fps),
            action_dim_override=int(args.max_action_dim),
        )
        print("Dataset contract check: OK")

    # Load dataset (from disk).
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    val_split = float(args.val_split)
    val_seed = int(args.val_seed) if args.val_seed is not None else int(args.seed)
    ds_val: Optional[LeRobotDataset] = None

    if val_split > 0:
        # Load a tiny slice just to read metadata.total_episodes without mmap'ing the full dataset.
        ds_meta = LeRobotDataset(repo_id=args.dataset_repo_id, root=dataset_dir, episodes=[0], download_videos=False)
        total_eps = int(ds_meta.meta.total_episodes)
        train_eps, val_eps = _split_episode_indices(num_episodes=total_eps, val_split=val_split, seed=val_seed)
        print(f"Episode split: train={len(train_eps)} val={len(val_eps)} (val_split={val_split:g}, seed={val_seed})")
        ds = LeRobotDataset(repo_id=args.dataset_repo_id, root=dataset_dir, episodes=train_eps, download_videos=False)
        ds_val = LeRobotDataset(repo_id=args.dataset_repo_id, root=dataset_dir, episodes=val_eps, download_videos=False)
    else:
        print(f"Episode split disabled (val_split={val_split:g}). Using all episodes for training.")
        ds = LeRobotDataset(repo_id=args.dataset_repo_id, root=dataset_dir, download_videos=False)

    dl = _make_dataloader(
        ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        pin_memory=pin_memory,
    )
    eval_bs = int(args.eval_batch_size) if args.eval_batch_size is not None else int(args.batch_size)
    dl_train_eval = _make_eval_dataloader(
        ds, batch_size=eval_bs, num_workers=int(args.num_workers), pin_memory=pin_memory
    )
    dl_val = (
        _make_eval_dataloader(ds_val, batch_size=eval_bs, num_workers=int(args.num_workers), pin_memory=pin_memory)
        if ds_val is not None
        else None
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

    # -------------------------
    # Contract + pipeline smoke test
    # -------------------------
    do_smoke_test = bool(args.smoke_test) and not bool(args.no_smoke_test)
    if do_smoke_test:
        def _describe_val(v: Any) -> str:
            if isinstance(v, torch.Tensor):
                return f"torch{tuple(v.shape)} {str(v.dtype)}"
            if isinstance(v, (list, tuple)):
                if not v:
                    return f"{type(v).__name__} len=0"
                t0 = type(v[0]).__name__
                return f"{type(v).__name__} len={len(v)} (first={t0})"
            return type(v).__name__

        def _print_key_shapes(tag: str, batch: Any) -> None:
            keys_of_interest = (
                "task",
                "observation.images.image",
                "observation.images.image2",
                "observation.images.empty_camera_0",
                "observation.state",
                "action",
                "domain_id",
                "input_ids",
                "attention_mask",
                "labels",
            )
            print(f"{tag}:")
            if isinstance(batch, dict):
                for k in keys_of_interest:
                    if k in batch:
                        print(f"  - {k}: {_describe_val(batch[k])}")
                extra = [k for k in batch.keys() if k not in keys_of_interest]
                if extra:
                    print(f"  - (other keys): {sorted(extra)[:20]}")
            else:
                print(f"  - (non-dict batch): {_describe_val(batch)}")

        print("Running smoke test: one batch -> preprocessor -> policy.forward()")
        try:
            batch0 = next(iter(dl))
            _print_key_shapes("Raw batch", batch0)
            batch1 = preprocessor(batch0)
            _print_key_shapes("After preprocessor", batch1)
            with torch.no_grad():
                loss, _log_dict = policy.forward(batch1)
            if not torch.isfinite(loss).all():
                raise RuntimeError(f"Non-finite loss: {loss}")
            print(f"Smoke test OK: loss={float(loss):.6f}")
        except Exception as e:
            raise RuntimeError(f"Smoke test failed: {e}") from e

    do_eval_at_end = bool(args.eval_at_end) and not bool(args.no_eval_at_end)
    if bool(args.eval_only):
        print("Eval-only mode: computing loss on train (and val, if configured)")
        train_loss, train_logs = _eval_loss(
            policy=policy,
            preprocessor=preprocessor,
            dataloader=dl_train_eval,
            max_batches=int(args.eval_max_batches),
            use_amp=bool(args.use_amp),
        )
        sub = {k: v for k, v in train_logs.items() if k != "loss"}
        extra = " (sub_losses={" + ", ".join([f"{k}:{v:.4f}" for k, v in sub.items()]) + "})" if sub else ""
        print(f"[eval] train_loss={train_loss:.6f}{extra}")

        if dl_val is not None:
            val_loss, val_logs = _eval_loss(
                policy=policy,
                preprocessor=preprocessor,
                dataloader=dl_val,
                max_batches=int(args.eval_max_batches),
                use_amp=bool(args.use_amp),
            )
            sub = {k: v for k, v in val_logs.items() if k != "loss"}
            extra = " (sub_losses={" + ", ".join([f"{k}:{v:.4f}" for k, v in sub.items()]) + "})" if sub else ""
            print(f"[eval] val_loss={val_loss:.6f}{extra}")
        else:
            print("[eval] No validation split configured (set --val_split > 0 to enable).")
        return 0

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
        eval_dataloader=dl_val,
        optimizer=optimizer1,
        scheduler=scheduler1,
        steps=int(args.stage1_steps),
        grad_accum=int(args.grad_accum),
        grad_clip_norm=float(args.grad_clip_norm),
        use_amp=bool(args.use_amp),
        log_every=int(args.log_every),
        eval_every_steps=int(args.eval_every_steps),
        eval_every_epochs=int(args.eval_every_epochs),
        eval_max_batches=int(args.eval_max_batches),
        eval_at_end=do_eval_at_end,
        tag="stage1",
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
        eval_dataloader=dl_val,
        optimizer=optimizer2,
        scheduler=scheduler2,
        steps=int(args.stage2_steps),
        grad_accum=int(args.grad_accum),
        grad_clip_norm=float(args.grad_clip_norm),
        use_amp=bool(args.use_amp),
        log_every=int(args.log_every),
        eval_every_steps=int(args.eval_every_steps),
        eval_every_epochs=int(args.eval_every_epochs),
        eval_max_batches=int(args.eval_max_batches),
        eval_at_end=do_eval_at_end,
        tag="stage2",
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


