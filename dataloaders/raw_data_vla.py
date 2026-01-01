"""
Raw-data VLA dataset loader for this repo's `raw_data/episode_*/` format.

Observed structure:
- `raw_data/episode_XXXX/instruction.json` contains `language_command`
- `raw_data/episode_XXXX/ticks.jsonl` contains per-tick records with:
  - `image.path` (relative path to a PNG under the episode dir)
  - `policy.action_from_prev` (Δpos, Δrotvec, gripper_action)
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x)
    raise TypeError(f"Cannot convert to float: {type(x)} ({x})")


def _action_from_prev_to_7d(action_from_prev: Dict[str, Any]) -> Tuple[float, ...]:
    dp = [_to_float(v) for v in action_from_prev["ee_delta_pos_b"]]
    dr = [_to_float(v) for v in action_from_prev["ee_delta_rotvec_b"]]
    g = int(action_from_prev.get("gripper_action", 0))
    if len(dp) != 3 or len(dr) != 3:
        raise ValueError(f"Unexpected action shape: dp={dp}, dr={dr}")
    return (dp[0], dp[1], dp[2], dr[0], dr[1], dr[2], float(g))


@dataclasses.dataclass(frozen=True)
class ActionNormStats:
    # Mirror the structure OpenVLA expects under config.norm_stats[*]["action"]
    mask: Tuple[bool, ...]
    min: Tuple[float, ...]
    max: Tuple[float, ...]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    q01: Tuple[float, ...]
    q99: Tuple[float, ...]

    def as_openvla_action_dict(self) -> Dict[str, Any]:
        return {
            "mask": list(self.mask),
            "min": list(self.min),
            "max": list(self.max),
            "mean": list(self.mean),
            "std": list(self.std),
            "q01": list(self.q01),
            "q99": list(self.q99),
        }


def compute_action_norm_stats(actions: torch.Tensor) -> ActionNormStats:
    """
    Compute per-dimension stats for OpenVLA-style action normalization.

    `actions` is expected to be float tensor of shape [N, 7] with:
      (dx, dy, dz, drx, dry, drz, gripper)
    """
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions shape [N,7], got {tuple(actions.shape)}")

    # By convention in OpenVLA configs: last dim (gripper) is NOT unnormalized.
    mask = (True, True, True, True, True, True, False)

    # Avoid NaNs if N=1
    q01 = torch.quantile(actions, 0.01, dim=0) if actions.shape[0] > 1 else actions[0]
    q99 = torch.quantile(actions, 0.99, dim=0) if actions.shape[0] > 1 else actions[0]

    a_min = actions.min(dim=0).values
    a_max = actions.max(dim=0).values
    mean = actions.mean(dim=0)
    std = actions.std(dim=0, unbiased=False) if actions.shape[0] > 1 else torch.zeros_like(mean)

    def _tolist(x: torch.Tensor) -> List[float]:
        return [float(v) for v in x.detach().cpu().tolist()]

    return ActionNormStats(
        mask=mask,
        min=tuple(_tolist(a_min)),
        max=tuple(_tolist(a_max)),
        mean=tuple(_tolist(mean)),
        std=tuple(_tolist(std)),
        q01=tuple(_tolist(q01)),
        q99=tuple(_tolist(q99)),
    )


OPENVLA_ASSISTANT_POST_COLON_TOKEN_ID = 29871


def openvla_action_to_token_ids(
    action_7d: Sequence[float],
    *,
    stats: ActionNormStats,
    vocab_size: int,
    n_action_bins: int,
) -> List[int]:
    """
    Encode a 7D continuous action into OpenVLA's discretized action token IDs.

    OpenVLA decoding (from `modeling_prismatic.py`) does:
      discretized = vocab_size - token_id
      idx = clip(discretized - 1, 0, len(bin_centers)-1)
      normalized = bin_centers[idx]   # in [-1, 1]
      action = unnormalize(normalized) for mask=True dims

    Here we do the inverse: unnormalized action -> normalized [-1,1] -> nearest bin -> token_id.
    """
    if len(action_7d) != 7:
        raise ValueError(f"Expected 7D action, got len={len(action_7d)}")
    if n_action_bins < 3:
        raise ValueError(f"n_action_bins must be >= 3, got {n_action_bins}")

    # bins = linspace(-1, 1, n_action_bins) => centers count = n_action_bins - 1
    step = 2.0 / float(n_action_bins - 1)
    first_center = -1.0 + 0.5 * step
    max_center_idx = n_action_bins - 2

    token_ids: List[int] = []
    for dim, raw in enumerate(action_7d):
        raw_f = float(raw)

        # Gripper: map {0,1} -> {-1,+1}
        if dim == 6:
            normalized = -1.0 if raw_f <= 0.0 else 1.0
        else:
            if stats.mask[dim]:
                lo = stats.q01[dim]
                hi = stats.q99[dim]
                denom = hi - lo
                if abs(denom) < 1e-12:
                    normalized = 0.0
                else:
                    normalized = 2.0 * (raw_f - lo) / denom - 1.0
            else:
                normalized = raw_f

            if normalized < -1.0:
                normalized = -1.0
            elif normalized > 1.0:
                normalized = 1.0

        # nearest bin center index
        idx = int(round((normalized - first_center) / step))
        if idx < 0:
            idx = 0
        elif idx > max_center_idx:
            idx = max_center_idx

        token_id = int(vocab_size - (idx + 1))
        token_ids.append(token_id)

    return token_ids


@dataclasses.dataclass(frozen=True)
class Sample:
    image_path: Path
    instruction: str
    action_7d: Tuple[float, ...]


class RawDataVLADataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        *,
        max_samples: Optional[int] = None,
        seed: int = 0,
        use_prev_image_for_action: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.samples: List[Sample] = []

        episode_dirs = sorted(
            [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")]
        )
        if not episode_dirs:
            raise FileNotFoundError(
                f"No episode directories found under: {data_dir} (expected episode_*)"
            )

        for ep_dir in episode_dirs:
            instr_path = ep_dir / "instruction.json"
            ticks_path = ep_dir / "ticks.jsonl"
            if not instr_path.exists() or not ticks_path.exists():
                continue

            instruction = _read_json(instr_path).get("language_command", "").strip()
            if not instruction:
                continue

            ticks = list(_iter_jsonl(ticks_path))
            if len(ticks) < 2:
                continue

            for i in range(1, len(ticks)):
                cur = ticks[i]
                afp = (cur.get("policy") or {}).get("action_from_prev")
                if afp is None:
                    continue
                a7 = _action_from_prev_to_7d(afp)

                obs_tick = ticks[i - 1] if use_prev_image_for_action else cur
                rel_img = (obs_tick.get("image") or {}).get("path")
                if not rel_img:
                    continue
                img_path = ep_dir / rel_img
                if not img_path.exists():
                    continue

                self.samples.append(
                    Sample(
                        image_path=img_path,
                        instruction=instruction,
                        action_7d=tuple(a7),
                    )
                )

        if not self.samples:
            raise RuntimeError(
                f"Found 0 training samples under {data_dir}. "
                "Check that episodes contain ticks with policy.action_from_prev and images."
            )

        rng = random.Random(seed)
        rng.shuffle(self.samples)
        if max_samples is not None:
            self.samples = self.samples[: max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


class VLADataCollator:
    def __init__(
        self,
        processor: Any,
        *,
        action_stats: ActionNormStats,
        action_vocab_size: int,
        n_action_bins: int,
        max_length: int = 512,
        assistant_post_colon_token_id: int = OPENVLA_ASSISTANT_POST_COLON_TOKEN_ID,
    ) -> None:
        self.processor = processor
        self.action_stats = action_stats
        self.action_vocab_size = int(action_vocab_size)
        self.n_action_bins = int(n_action_bins)
        self.assistant_post_colon_token_id = int(assistant_post_colon_token_id)

        tok = getattr(processor, "tokenizer", None)
        if tok is not None and tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if tok is not None:
            tok.padding_side = "right"
        self.max_length = max_length

    def _build_prompt(self, instruction: str) -> str:
        # OpenVLA's `predict_action()` assumes the prompt ends with "ASSISTANT:",
        # then appends token id 29871 after the colon.
        return f"USER: {instruction}\nASSISTANT:"

    def __call__(self, batch: Sequence[Sample]) -> Dict[str, torch.Tensor]:
        images = [Image.open(s.image_path).convert("RGB") for s in batch]
        prompts = [self._build_prompt(s.instruction) for s in batch]

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

        pad = tok.pad_token_id
        if pad is None:
            raise RuntimeError("Tokenizer has no pad_token_id; cannot pad batches.")

        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        attn_list: List[torch.Tensor] = []

        for i in range(len(batch)):
            prompt_ids = proc["input_ids"][i]
            prompt_attn = proc["attention_mask"][i]

            # strip padding from prompt
            prompt_ids = prompt_ids[prompt_attn.bool()]

            # Append OpenVLA's post-colon token (see `OpenVLAForActionPrediction.predict_action`).
            post_colon = torch.tensor([self.assistant_post_colon_token_id], dtype=prompt_ids.dtype)
            prompt_ids = torch.cat([prompt_ids, post_colon], dim=0)

            # Encode 7D action into discretized action token IDs.
            action_token_ids = openvla_action_to_token_ids(
                batch[i].action_7d,
                stats=self.action_stats,
                vocab_size=self.action_vocab_size,
                n_action_bins=self.n_action_bins,
            )
            action_ids = torch.tensor(action_token_ids, dtype=prompt_ids.dtype)

            full_ids = torch.cat([prompt_ids, action_ids], dim=0)
            full_attn = torch.ones_like(full_ids, dtype=proc["attention_mask"].dtype)
            full_labels = torch.cat(
                [
                    torch.full_like(prompt_ids, -100),
                    action_ids,
                ]
            )

            input_ids_list.append(full_ids)
            attn_list.append(full_attn)
            labels_list.append(full_labels)

        max_len = max(t.numel() for t in input_ids_list)

        def _pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
            if x.numel() == max_len:
                return x
            return torch.cat([x, torch.full((max_len - x.numel(),), pad_value, dtype=x.dtype)])

        input_ids = torch.stack([_pad_1d(x, pad) for x in input_ids_list], dim=0)
        attention_mask = torch.stack([_pad_1d(x, 0) for x in attn_list], dim=0)
        labels = torch.stack([_pad_1d(x, -100) for x in labels_list], dim=0)

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for k, v in proc.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if isinstance(v, torch.Tensor):
                out[k] = v

        return out


