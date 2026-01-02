"""
X-VLA v1 fine-tuning script (template).

This repo's dataset is stored under `raw_data/episode_*/`. This script is wired to that
format via `dataloaders/raw_data_vla.py`, so you can swap different model checkpoints
in and out via `--model_id`.

Default training objective (generic):
- Input: (image, instruction prompt)
- Target: a 7D action rendered as text: "dx dy dz drx dry drz gripper"

If your X-VLA checkpoint uses a different action representation (e.g., discretized
action tokens like OpenVLA, or continuous regression heads), modify the collator/loss
in this file.

Example:
  python train_scripts/X-VLA/X_VLA_v1.py --model_id <HF_OR_LOCAL_XVLA_ID> --dry_run
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Allow running from anywhere (so `import dataloaders...` works even if cwd != repo root)
# This file lives under: train_scripts/X-VLA/X_VLA_v1.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.raw_data_vla import RawDataVLADataset, Sample  # noqa: E402


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


def format_action_text(action_7d: Sequence[float], precision: int = 4) -> str:
    if len(action_7d) != 7:
        raise ValueError(f"Expected 7D action, got len={len(action_7d)}")
    fmt = f"{{:+.{precision}f}}"
    return " ".join(
        [fmt.format(float(v)) for v in action_7d[:6]] + [str(int(round(float(action_7d[6]))))]
    )


@dataclass(frozen=True)
class TextTargetSample:
    image_path: Path
    instruction: str
    target_text: str


class RawDataTextTargetDataset(Dataset):
    """Wrap RawDataVLADataset and convert action_7d to target text."""

    def __init__(self, base: RawDataVLADataset, *, action_precision: int = 4) -> None:
        self.base = base
        self.action_precision = action_precision

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> TextTargetSample:
        s: Sample = self.base[idx]
        return TextTargetSample(
            image_path=s.image_path,
            instruction=s.instruction,
            target_text=format_action_text(s.action_7d, precision=self.action_precision),
        )


class TextVLADataCollator:
    """Generic collator for (image, prompt) -> (action text) causal LM training."""

    def __init__(self, processor: Any, *, prompt_template: str, max_length: int = 512) -> None:
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_length = max_length

        tok = getattr(processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("Processor has no tokenizer; cannot build labels.")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"

    def _prompt(self, instruction: str) -> str:
        return self.prompt_template.format(instruction=instruction)

    def __call__(self, batch: Sequence[TextTargetSample]) -> Dict[str, torch.Tensor]:
        images = [Image.open(s.image_path).convert("RGB") for s in batch]
        prompts = [self._prompt(s.instruction) for s in batch]
        targets = [s.target_text for s in batch]

        proc = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("Processor has no tokenizer; cannot build labels.")

        eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos

        target_tok = tok(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        attn_list: List[torch.Tensor] = []

        for i in range(len(batch)):
            prompt_ids = proc["input_ids"][i]
            prompt_attn = proc["attention_mask"][i]
            tgt_ids = target_tok["input_ids"][i]
            tgt_attn = target_tok["attention_mask"][i]

            prompt_ids = prompt_ids[prompt_attn.bool()]
            tgt_ids = tgt_ids[tgt_attn.bool()]

            full_ids = torch.cat(
                [prompt_ids, tgt_ids, torch.tensor([eos], dtype=prompt_ids.dtype)]
            )
            full_attn = torch.ones_like(full_ids, dtype=prompt_attn.dtype)
            full_labels = torch.cat(
                [
                    torch.full_like(prompt_ids, -100),
                    tgt_ids.to(dtype=prompt_ids.dtype),
                    torch.tensor([eos], dtype=prompt_ids.dtype),
                ]
            )

            input_ids_list.append(full_ids)
            attn_list.append(full_attn)
            labels_list.append(full_labels)

        max_len = max(t.numel() for t in input_ids_list)

        def _pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
            if x.numel() == max_len:
                return x
            return torch.cat(
                [x, torch.full((max_len - x.numel(),), pad_value, dtype=x.dtype)]
            )

        input_ids = torch.stack([_pad_1d(x, pad) for x in input_ids_list], dim=0)
        attention_mask = torch.stack([_pad_1d(x, 0) for x in attn_list], dim=0)
        labels = torch.stack([_pad_1d(x, -100) for x in labels_list], dim=0)

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Preserve vision tensors (pixel_values, etc.)
        for k, v in proc.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if isinstance(v, torch.Tensor):
                out[k] = v

        return out


def load_xvla_model_and_processor(
    model_id: str,
    *,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: str = "eager",
) -> Tuple[Any, Any]:
    """Generic HF loader (works for many multimodal LMs)."""
    _require("transformers", "Install with: pip install -r requirements.txt")
    from transformers import (
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

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

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception:
        model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)

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
    prompt_template: str,
    batch_size: int = 1,
    lr: float = 2e-4,
    max_steps: int = 50,
    grad_accum: int = 1,
    log_every: int = 5,
    max_length: int = 512,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    collate = TextVLADataCollator(processor, prompt_template=prompt_template, max_length=max_length)
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
    p = argparse.ArgumentParser(description="Fine-tune X-VLA on ./raw_data (template)")
    p.add_argument("--model_id", type=str, required=True, help="HF model id or local path for X-VLA checkpoint")
    p.add_argument("--data_dir", type=str, default="raw_data", help="Path to raw_data directory")
    p.add_argument("--output_dir", type=str, default="checkpoints/x_vla_v1", help="Where to save finetuned outputs")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples (quick tests)")
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--action_precision", type=int, default=4)
    p.add_argument(
        "--prompt_template",
        type=str,
        default="<image>\\nInstruction: {instruction}\\nAction:",
        help="Prompt template; must include '{instruction}'. For models like LLaVA, keep the <image> token.",
    )
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Only build dataset and load model; do not train")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Loading dataset from: {data_dir}")
    base = RawDataVLADataset(data_dir, max_samples=args.max_samples, seed=args.seed)
    ds = RawDataTextTargetDataset(base, action_precision=args.action_precision)
    print(f"Loaded {len(ds)} samples")
    print(f"Example target: {ds[0].target_text} | image={ds[0].image_path.name} | instr={ds[0].instruction}")

    print(f"Loading X-VLA model: {args.model_id}")
    model, processor = load_xvla_model_and_processor(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )

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
        prompt_template=args.prompt_template,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
    )

    
    
    

    return 0



    
    

if __name__ == "__main__":
    raise SystemExit(main())


