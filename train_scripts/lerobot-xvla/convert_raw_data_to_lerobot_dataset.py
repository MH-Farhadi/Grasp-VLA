"""
Convert this repo's `raw_data/episode_*/` format into a LeRobotDataset (v3.0) on disk.

This is the bridge needed to fine-tune LeRobot policies (like X-VLA) on your local data using
`lerobot-train`.

Example:

  python train_scripts/lerobot-xvla/convert_raw_data_to_lerobot_dataset.py \
    --raw_data_dir raw_data \
    --out_dir lerobot_datasets/grasp_raw \
    --repo_id grasp_raw \
    --fps 5 \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def _require_lerobot() -> None:
    try:
        import lerobot  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "LeRobot is not installed in this environment.\n"
            "Install it (editable) from the sibling repo:\n"
            "  pip install -e /home/kye/Desktop/Depo/Code/Grasp-VLA/lerobot[xvla]\n"
            f"Original error: {e}"
        ) from e


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


def _gripper_state_to_float(gripper_state: Any) -> float:
    # Common values in this repo's raw_data ticks: "open" / "closed"
    if isinstance(gripper_state, str):
        return 0.0 if gripper_state.strip().lower() == "open" else 1.0
    # If unknown type, default to 0.0
    return 0.0


def _build_state_7d_from_tick(tick: Dict[str, Any]) -> np.ndarray:
    robot = tick.get("robot") or {}
    joints = (robot.get("joints") or {}).get("positions") or []
    joints_f: List[float] = []
    for v in joints:
        try:
            joints_f.append(_to_float(v))
        except Exception:
            continue

    # Expect 6 joints in the sample data; pad/trim defensively.
    joints_f = (joints_f + [0.0] * 6)[:6]
    gripper_state = (robot.get("gripper") or {}).get("state")
    g = _gripper_state_to_float(gripper_state)
    state = np.asarray(joints_f + [g], dtype=np.float32)
    if state.shape != (7,):
        raise ValueError(f"Expected state shape (7,), got {state.shape}")
    return state


def _find_first_image_path(raw_data_dir: Path) -> Path:
    for ep_dir in sorted([p for p in raw_data_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")]):
        ticks_path = ep_dir / "ticks.jsonl"
        if not ticks_path.exists():
            continue
        for tick in _iter_jsonl(ticks_path):
            rel = (tick.get("image") or {}).get("path")
            if not rel:
                continue
            img_path = ep_dir / rel
            if img_path.exists():
                return img_path
    raise FileNotFoundError(f"Could not find any images under: {raw_data_dir}")


def convert(
    *,
    raw_data_dir: Path,
    out_dir: Path,
    repo_id: str,
    fps: int,
    camera_key: str = "image",
    max_episodes: Optional[int] = None,
    max_frames_per_episode: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw_data_dir = raw_data_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"--raw_data_dir not found: {raw_data_dir}")

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"--out_dir already exists: {out_dir}\n"
                "Pass --overwrite to delete and recreate it."
            )

    first_img_path = _find_first_image_path(raw_data_dir)
    with Image.open(first_img_path) as img:
        img = img.convert("RGB")
        w, h = img.size

    img_shape_hwc = (h, w, 3)
    image_feature_key = f"observation.images.{camera_key}"

    features: Dict[str, Dict[str, Any]] = {
        image_feature_key: {
            "dtype": "image",
            "shape": img_shape_hwc,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        },
    }

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        root=out_dir,
        robot_type="Grasp-VLA",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    episode_dirs = sorted([p for p in raw_data_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    if max_episodes is not None:
        episode_dirs = episode_dirs[: max_episodes]

    total_eps = 0
    total_frames = 0
    skipped_eps = 0

    for ep_dir in episode_dirs:
        instr_path = ep_dir / "instruction.json"
        ticks_path = ep_dir / "ticks.jsonl"
        if not instr_path.exists() or not ticks_path.exists():
            skipped_eps += 1
            continue

        instruction = (_read_json(instr_path).get("language_command") or "").strip()
        if not instruction:
            skipped_eps += 1
            continue

        ticks = list(_iter_jsonl(ticks_path))
        if len(ticks) < 2:
            skipped_eps += 1
            continue

        n_added = 0
        for i in range(1, len(ticks)):
            cur = ticks[i]
            afp = (cur.get("policy") or {}).get("action_from_prev")
            if afp is None:
                continue

            obs_tick = ticks[i - 1]
            rel_img = (obs_tick.get("image") or {}).get("path")
            if not rel_img:
                continue
            img_path = ep_dir / rel_img
            if not img_path.exists():
                continue

            # Load image as PIL and hand it to LeRobotDataset, which will re-encode into its own folder.
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                # Safety check: all frames must match the declared feature shape.
                if im.size != (w, h):
                    raise ValueError(
                        f"Image shape mismatch in {img_path}: got (W,H)={im.size}, expected {(w, h)}"
                    )
                image = im.copy()

            action_7d = np.asarray(_action_from_prev_to_7d(afp), dtype=np.float32)
            if action_7d.shape != (7,):
                raise ValueError(f"Expected action shape (7,), got {action_7d.shape}")

            state_7d = _build_state_7d_from_tick(obs_tick)

            frame = {
                "task": instruction,
                image_feature_key: image,
                "observation.state": state_7d,
                "action": action_7d,
            }
            ds.add_frame(frame)
            n_added += 1

            if max_frames_per_episode is not None and n_added >= max_frames_per_episode:
                break

        if n_added > 0:
            ds.save_episode()
            total_eps += 1
            total_frames += n_added
        else:
            skipped_eps += 1

    ds.finalize()
    print(f"Saved LeRobotDataset to: {out_dir}")
    print(f"Converted episodes: {total_eps} | frames: {total_frames} | skipped episodes: {skipped_eps}")
    return out_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Grasp-VLA raw_data -> LeRobotDataset v3.0")
    p.add_argument("--raw_data_dir", type=str, default="raw_data", help="Path to raw_data directory")
    p.add_argument("--out_dir", type=str, required=True, help="Output dataset directory (will be created)")
    p.add_argument("--repo_id", type=str, default="grasp_raw", help="Dataset repo_id stored in metadata")
    p.add_argument("--fps", type=int, default=5, help="Frames per second for the dataset")
    p.add_argument(
        "--camera_key",
        type=str,
        default="image",
        help=(
            "Camera key suffix for observation.images.<key>. "
            "Default 'image' matches lerobot/xvla-base's expected key 'observation.images.image'."
        ),
    )
    p.add_argument("--max_episodes", type=int, default=None, help="Limit number of episodes (debug)")
    p.add_argument("--max_frames_per_episode", type=int, default=None, help="Limit frames per episode (debug)")
    p.add_argument("--overwrite", action="store_true", help="Delete out_dir if it already exists")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    convert(
        raw_data_dir=Path(args.raw_data_dir),
        out_dir=Path(args.out_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        camera_key=args.camera_key,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


