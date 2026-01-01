"""
OpenVLA v1 fine-tuning script for this repo.

This script:
- Loads the local dataset from `raw_data/episode_*/`
- Loads an OpenVLA checkpoint from the Hugging Face Hub (or local path)
- Fine-tunes using OpenVLA's native discretized action-token objective (LoRA by default)
- Saves outputs under `checkpoints/`

Example:
  python train_scripts/openvla_v1.py --model_id openvla/openvla-v01-7b --max_steps 50
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Allow running from anywhere (so `import dataloaders...` works even if cwd != repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.raw_data_vla import (  # noqa: E402
    ActionNormStats,
    RawDataVLADataset,
    VLADataCollator,
    compute_action_norm_stats,
)


def _require(pkg: str, install_hint: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"Missing dependency '{pkg}'. {install_hint}\nOriginal error: {e}"
        ) from e


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_openvla_model_and_processor(
    model_id: str,
    *,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    attn_implementation: str = "eager",
) -> Tuple[Any, Any]:
    """
    Load OpenVLA via Transformers remote code.

    Note: On transformers>=4.57, OpenVLA often fails SDPA probing during __init__. For compatibility,
    we default to `attn_implementation="eager"`.
    """
    _require(
        "transformers",
        "Install with: pip install -r requirements.txt",
    )

    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype: Optional[torch.dtype]
    if torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else None
    elif torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif torch_dtype == "fp16":
        dtype = torch.float16
    elif torch_dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown --torch_dtype: {torch_dtype}")

    kwargs: Dict[str, Any] = dict(
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    if load_in_4bit:
        _require("bitsandbytes", "Install with: pip install bitsandbytes (GPU recommended).")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype or torch.float16,
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    try:
        model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    except AttributeError as e:
        if "_supports_sdpa" in str(e) and attn_implementation != "eager":
            kwargs["attn_implementation"] = "eager"
            model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
        else:
            raise

    return model, processor


def maybe_enable_lora(model: Any, *, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> Any:
    _require("peft", "Install with: pip install peft")
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def train(
    model: Any,
    processor: Any,
    train_ds: Dataset,
    *,
    output_dir: Path,
    action_stats: ActionNormStats,
    batch_size: int = 1,
    lr: float = 2e-4,
    max_steps: int = 50,
    grad_accum: int = 1,
    log_every: int = 5,
    max_length: int = 512,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_action_bins = int(getattr(getattr(model, "config", None), "n_action_bins", 256))
    action_vocab_size = getattr(model, "vocab_size", None)
    if action_vocab_size is None:
        cfg = getattr(model, "config", None)
        if cfg is None or not hasattr(cfg, "text_config") or not hasattr(cfg, "pad_to_multiple_of"):
            raise RuntimeError("Could not determine OpenVLA action vocab size from model.")
        action_vocab_size = int(cfg.text_config.vocab_size) - int(cfg.pad_to_multiple_of)

    collate = VLADataCollator(
        processor,
        action_stats=action_stats,
        action_vocab_size=int(action_vocab_size),
        n_action_bins=n_action_bins,
        max_length=max_length,
    )
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(10_000_000):
        for batch in dl:
            step += 1
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            (loss / grad_accum).backward()

            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                print(f"[step {step:05d}] loss={loss.item():.6f}")

            if step >= max_steps:
                output_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(str(output_dir))
                if hasattr(processor, "save_pretrained"):
                    processor.save_pretrained(str(output_dir))
                print(f"Saved to: {output_dir}")
                return


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune OpenVLA on ./raw_data")
    p.add_argument("--model_id", type=str, required=True, help="HF model id or local path (e.g. openvla/openvla-v01-7b)")
    p.add_argument("--data_dir", type=str, default="raw_data", help="Path to raw_data directory")
    p.add_argument("--output_dir", type=str, default="checkpoints/openvla_v1", help="Where to save finetuned outputs")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples (quick tests)")
    p.add_argument("--max_steps", type=int, default=50, help="Number of optimizer steps to run")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend. `eager` is most compatible for OpenVLA on newer Transformers.",
    )
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--norm_key", type=str, default="grasp_vla_raw_data", help="Key for storing action norm stats")
    p.add_argument("--dry_run", action="store_true", help="Only build dataset and load model; do not train")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Loading dataset from: {data_dir}")
    ds = RawDataVLADataset(data_dir, max_samples=args.max_samples, seed=args.seed)
    print(f"Loaded {len(ds)} samples")
    print(f"Example action_7d: {ds[0].action_7d} | image={ds[0].image_path.name} | instr={ds[0].instruction}")

    actions = torch.tensor([s.action_7d for s in ds.samples], dtype=torch.float32)
    action_stats = compute_action_norm_stats(actions)

    print(f"Loading model: {args.model_id}")
    model, processor = load_openvla_model_and_processor(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
    )

    # Register stats under a dedicated key for future `predict_action(..., unnorm_key=...)` usage.
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None and isinstance(getattr(cfg, "norm_stats", None), dict) and args.norm_key:
            cfg.norm_stats[args.norm_key] = {"action": action_stats.as_openvla_action_dict()}
            if hasattr(model, "norm_stats"):
                model.norm_stats = cfg.norm_stats
            print(f"Registered action norm stats under norm_key='{args.norm_key}'")
    except Exception:
        pass

    if args.dry_run:
        print("Dry run complete (dataset + model loaded).")
        return 0

    if not args.no_lora:
        model = maybe_enable_lora(model)

    train(
        model,
        processor,
        ds,
        output_dir=output_dir,
        action_stats=action_stats,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


