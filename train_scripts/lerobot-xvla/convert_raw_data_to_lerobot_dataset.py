"""
Convert this repo's raw-data format into a LeRobotDataset (v3.0) on disk.

This is the bridge needed to fine-tune LeRobot policies (like X-VLA) on your local data using
`lerobot-train`.

Supported raw_data layouts:
- Old: `raw_data/episode_XXXX/`
- New: `raw_data/session_YYYYMMDD_HHMMSS/episode_XXXX/` (multiple sessions under one root)

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


def _build_state_from_tick(tick: Dict[str, Any], *, state_dim: int) -> np.ndarray:
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

    base = joints_f + [g]  # 7 dims
    if state_dim < len(base):
        raise ValueError(f"state_dim must be >= {len(base)} to fit joints+gripper, got {state_dim}")
    if state_dim > len(base):
        base = base + [0.0] * (state_dim - len(base))

    state = np.asarray(base, dtype=np.float32)
    if state.shape != (int(state_dim),):
        raise ValueError(f"Expected state shape ({state_dim},), got {state.shape}")
    return state


def _is_episode_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("episode_")


def _is_session_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("session_")


def _list_episode_dirs(raw_data_dir: Path) -> List[Path]:
    """
    Return episode directories in a deterministic order.

    Supports:
    - raw_data_dir/episode_*
    - raw_data_dir/session_*/episode_*
    - raw_data_dir itself being a session dir (i.e., it contains episode_*)
    """
    raw_data_dir = raw_data_dir.expanduser().resolve()

    episode_dirs: List[Path] = []

    # 1) Episodes directly under the provided directory (old layout, or when passing a session dir).
    episode_dirs.extend(sorted([p for p in raw_data_dir.iterdir() if _is_episode_dir(p)]))

    # 2) Episodes nested one level down under session_* dirs (new layout).
    session_dirs = sorted([p for p in raw_data_dir.iterdir() if _is_session_dir(p)])
    for sdir in session_dirs:
        episode_dirs.extend(sorted([p for p in sdir.iterdir() if _is_episode_dir(p)]))

    # De-dupe (defensive) while preserving order.
    seen: set[Path] = set()
    unique: List[Path] = []
    for p in episode_dirs:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(rp)
    return unique


def _find_first_image_path(episode_dirs: Sequence[Path]) -> Path:
    for ep_dir in episode_dirs:
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
    raise FileNotFoundError(
        "Could not find any images under the discovered episode directories. "
        "Check that episodes contain ticks.jsonl with image.path entries and that the PNGs exist."
    )


def _resize_center_crop_rgb(im: Image.Image, *, out_wh: Tuple[int, int]) -> Image.Image:
    """
    Deterministic preprocessing:
    - convert to RGB
    - center-crop to square (if needed)
    - resize to out_wh
    """
    im = im.convert("RGB")
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Invalid out_wh={out_wh}")

    w, h = im.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))

    if im.size != (out_w, out_h):
        im = im.resize((out_w, out_h), resample=Image.BILINEAR)
    return im


def _make_empty_rgb_image(*, wh: Tuple[int, int]) -> Image.Image:
    w, h = int(wh[0]), int(wh[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid wh={wh}")
    return Image.new("RGB", (w, h), color=(0, 0, 0))


def _pad_action(action_7d: Sequence[float], *, action_dim: int) -> np.ndarray:
    if len(action_7d) != 7:
        raise ValueError(f"Expected 7D action, got len={len(action_7d)}")
    if action_dim < 7:
        raise ValueError(f"action_dim must be >= 7, got {action_dim}")
    out = np.zeros((int(action_dim),), dtype=np.float32)
    out[:7] = np.asarray(action_7d, dtype=np.float32)
    return out


def _summarize_episode_events(events_path: Path) -> Tuple[bool, bool]:
    """
    Returns (is_truncated, has_success).
    Enforces: exactly one episode_end event.
    """
    episode_end_count = 0
    is_truncated = False
    has_success = False

    for ev in _iter_jsonl(events_path):
        ev_type = (ev.get("type") or "").strip()
        if ev_type == "grasp_result":
            ok = bool((ev.get("data") or {}).get("ok"))
            has_success = has_success or ok
        elif ev_type == "episode_end":
            episode_end_count += 1
            is_truncated = bool((ev.get("data") or {}).get("truncated", False))

    if episode_end_count != 1:
        raise RuntimeError(f"Expected exactly 1 episode_end event, found {episode_end_count} in {events_path}")
    return is_truncated, has_success


def convert(
    *,
    raw_data_dir: Path,
    out_dir: Path,
    repo_id: str,
    fps: int,
    camera_key: str = "image",
    # XVLA contract knobs (defaults match lerobot/xvla-base)
    image_wh: Tuple[int, int] = (256, 256),
    image2_wh: Optional[Tuple[int, int]] = (256, 256),
    empty_camera_0_wh: Optional[Tuple[int, int]] = (224, 224),
    state_dim: int = 8,
    action_dim: int = 20,
    # Raw-data quality/curation
    raw_quality_gate: bool = True,
    drop_truncated: bool = True,
    success_only: bool = False,
    max_episodes: Optional[int] = None,
    min_frames_per_episode: Optional[int] = None,
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

    episode_dirs_all = _list_episode_dirs(raw_data_dir)
    if not episode_dirs_all:
        raise FileNotFoundError(
            f"No episode directories found under: {raw_data_dir}\n"
            "Expected either:\n"
            "  - raw_data/episode_XXXX/\n"
            "  - raw_data/session_*/episode_XXXX/\n"
        )

    # Optional debug: show how many sessions we detected.
    session_dirs = sorted([p for p in raw_data_dir.iterdir() if _is_session_dir(p)])
    if session_dirs:
        print(
            f"Discovered {len(session_dirs)} session dirs and {len(episode_dirs_all)} episode dirs under {raw_data_dir}"
        )
    else:
        print(f"Discovered {len(episode_dirs_all)} episode dirs under {raw_data_dir}")

    episode_dirs = episode_dirs_all
    if max_episodes is not None:
        episode_dirs = episode_dirs[: max_episodes]

    image_feature_key = f"observation.images.{camera_key}"
    img_shape_hwc = (int(image_wh[1]), int(image_wh[0]), 3)

    image2_feature_key = "observation.images.image2" if image2_wh is not None else None
    empty0_feature_key = "observation.images.empty_camera_0" if empty_camera_0_wh is not None else None

    features: Dict[str, Dict[str, Any]] = {
        image_feature_key: {
            "dtype": "image",
            "shape": img_shape_hwc,
            "names": ["height", "width", "channels"],
        },
        **(
            {
                image2_feature_key: {
                    "dtype": "image",
                    "shape": (int(image2_wh[1]), int(image2_wh[0]), 3),
                    "names": ["height", "width", "channels"],
                }
            }
            if (image2_feature_key is not None and image2_wh is not None)
            else {}
        ),
        **(
            {
                empty0_feature_key: {
                    "dtype": "image",
                    "shape": (int(empty_camera_0_wh[1]), int(empty_camera_0_wh[0]), 3),
                    "names": ["height", "width", "channels"],
                }
            }
            if (empty0_feature_key is not None and empty_camera_0_wh is not None)
            else {}
        ),
        "observation.state": {
            "dtype": "float32",
            "shape": (int(state_dim),),
            "names": (
                [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "gripper",
                ]
                + [f"pad_{i}" for i in range(7, int(state_dim))]
            ),
        },
        "action": {
            "dtype": "float32",
            "shape": (int(action_dim),),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
            + [f"pad_{i}" for i in range(7, int(action_dim))],
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

    total_eps = 0
    total_frames = 0
    skipped_eps = 0
    expected_robot_sig: Optional[Tuple[str, str, str]] = None

    for ep_dir in episode_dirs:
        instr_path = ep_dir / "instruction.json"
        ticks_path = ep_dir / "ticks.jsonl"
        events_path = ep_dir / "events.jsonl"
        meta_path = ep_dir / "metadata.json"
        if not instr_path.exists() or not ticks_path.exists():
            skipped_eps += 1
            continue
        if raw_quality_gate:
            # Episode completeness checks.
            if not events_path.exists() or not meta_path.exists() or not (ep_dir / "images").exists():
                skipped_eps += 1
                continue

        instruction = (_read_json(instr_path).get("language_command") or "").strip()
        if not instruction:
            skipped_eps += 1
            continue

        if raw_quality_gate:
            try:
                meta = _read_json(meta_path)
                cfg = meta.get("config") or {}
                log_rate_hz = cfg.get("log_rate_hz")
                if log_rate_hz is None:
                    raise KeyError("metadata.json missing config.log_rate_hz")
                if int(log_rate_hz) != int(fps):
                    raise RuntimeError(
                        f"metadata.json log_rate_hz={int(log_rate_hz)} does not match requested fps={int(fps)} "
                        f"({meta_path})"
                    )

                robot = meta.get("robot") or {}
                robot_name = str(robot.get("name") or "")
                ee_link = str(robot.get("ee_link") or "")
                joint_regex = str(robot.get("arm_joint_regex") or "")
                sig = (robot_name, ee_link, joint_regex)
                if expected_robot_sig is None:
                    expected_robot_sig = sig
                elif sig != expected_robot_sig:
                    raise RuntimeError(
                        "Mixed robot metadata across episodes; refusing to convert a mixed-domain dataset.\n"
                        f"Expected (name,ee_link,arm_joint_regex)={expected_robot_sig} but got {sig} in {meta_path}"
                    )
            except Exception:
                skipped_eps += 1
                continue

        if raw_quality_gate:
            try:
                is_truncated, has_success = _summarize_episode_events(events_path)
            except Exception:
                skipped_eps += 1
                continue
            if drop_truncated and is_truncated:
                skipped_eps += 1
                continue
            if success_only and not has_success:
                skipped_eps += 1
                continue

        n_added = 0
        prev_tick: Optional[Dict[str, Any]] = None
        expected_tick_idx = 0
        malformed = False

        try:
            for cur in _iter_jsonl(ticks_path):
                if raw_quality_gate:
                    tick_idx = cur.get("tick_idx")
                    if tick_idx is None or int(tick_idx) != expected_tick_idx:
                        malformed = True
                        break
                    if expected_tick_idx == 0:
                        afp0 = (cur.get("policy") or {}).get("action_from_prev")
                        if afp0 is not None:
                            malformed = True
                            break
                    expected_tick_idx += 1

                if prev_tick is None:
                    prev_tick = cur
                    continue

                # Alignment rule: observation from prev tick, action label from current tick.
                afp = (cur.get("policy") or {}).get("action_from_prev")
                if afp is None:
                    prev_tick = cur
                    continue

                rel_img = (prev_tick.get("image") or {}).get("path")
                if not rel_img:
                    prev_tick = cur
                    continue
                img_path = ep_dir / rel_img
                if not img_path.exists():
                    prev_tick = cur
                    continue

                with Image.open(img_path) as im:
                    image = _resize_center_crop_rgb(im, out_wh=image_wh)
                if image.size != (int(image_wh[0]), int(image_wh[1])):
                    raise ValueError(
                        f"Post-resize image mismatch in {img_path}: got (W,H)={image.size}, expected {image_wh}"
                    )

                image2 = None
                if image2_feature_key is not None and image2_wh is not None:
                    image2 = image.copy()
                    if image2.size != (int(image2_wh[0]), int(image2_wh[1])):
                        image2 = _resize_center_crop_rgb(image2, out_wh=image2_wh)

                empty0 = None
                if empty0_feature_key is not None and empty_camera_0_wh is not None:
                    empty0 = _make_empty_rgb_image(wh=empty_camera_0_wh)

                a7 = _action_from_prev_to_7d(afp)
                if raw_quality_gate:
                    # Magnitude sanity (catch unit bugs / corruption): these thresholds are intentionally generous.
                    dp = np.asarray(a7[:3], dtype=np.float32)
                    dr = np.asarray(a7[3:6], dtype=np.float32)
                    if np.linalg.norm(dp) > 0.25 or np.linalg.norm(dr) > 3.5:
                        malformed = True
                        break

                action = _pad_action(a7, action_dim=action_dim)
                state = _build_state_from_tick(prev_tick, state_dim=state_dim)

                if raw_quality_gate:
                    if not np.isfinite(action).all() or not np.isfinite(state).all():
                        malformed = True
                        break

                frame: Dict[str, Any] = {
                    "task": instruction,
                    image_feature_key: image,
                    "observation.state": state,
                    "action": action,
                }
                if image2_feature_key is not None and image2 is not None:
                    frame[image2_feature_key] = image2
                if empty0_feature_key is not None and empty0 is not None:
                    frame[empty0_feature_key] = empty0

                ds.add_frame(frame)
                n_added += 1
                prev_tick = cur

                if max_frames_per_episode is not None and n_added >= max_frames_per_episode:
                    break
        except Exception:
            malformed = True

        if malformed or n_added <= 0 or (min_frames_per_episode is not None and n_added < int(min_frames_per_episode)):
            skipped_eps += 1
            continue
        else:
            ds.save_episode()
            total_eps += 1
            total_frames += n_added

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
    p.add_argument(
        "--min_frames_per_episode",
        type=int,
        default=None,
        help="Drop episodes shorter than this many frames (useful for chunked models; e.g. 30 for xvla-base).",
    )
    p.add_argument("--max_frames_per_episode", type=int, default=None, help="Limit frames per episode (debug)")
    p.add_argument("--state_dim", type=int, default=8, help="State vector length (default matches xvla-base)")
    p.add_argument("--action_dim", type=int, default=20, help="Action vector length (default matches xvla-base)")
    p.add_argument(
        "--image_wh",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Resize (center-crop if needed) primary image to (W H). Default matches xvla-base.",
    )
    p.add_argument(
        "--no_image2",
        action="store_true",
        help="Disable emitting observation.images.image2 (xvla-base expects it; keep enabled for XVLA training).",
    )
    p.add_argument(
        "--no_empty_camera_0",
        action="store_true",
        help="Disable emitting observation.images.empty_camera_0 (xvla-base expects it; keep enabled for XVLA training).",
    )
    p.add_argument("--raw_quality_gate", action="store_true", default=True, help="Run strict raw log checks")
    p.add_argument("--no_raw_quality_gate", action="store_true", help="Disable raw log checks (not recommended)")
    p.add_argument(
        "--drop_truncated",
        action="store_true",
        default=True,
        help="Skip episodes with episode_end.truncated=true (recommended for clean imitation).",
    )
    p.add_argument("--keep_truncated", action="store_true", help="Keep truncated episodes")
    p.add_argument("--success_only", action="store_true", default=False, help="Only keep episodes with grasp_result.ok=true")
    p.add_argument("--overwrite", action="store_true", help="Delete out_dir if it already exists")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    raw_quality_gate = bool(args.raw_quality_gate) and not bool(args.no_raw_quality_gate)
    drop_truncated = bool(args.drop_truncated) and not bool(args.keep_truncated)
    image_wh = tuple(int(x) for x in args.image_wh)
    image2_wh = None if bool(args.no_image2) else image_wh
    empty0_wh = None if bool(args.no_empty_camera_0) else (224, 224)
    convert(
        raw_data_dir=Path(args.raw_data_dir),
        out_dir=Path(args.out_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        camera_key=args.camera_key,
        image_wh=image_wh,
        image2_wh=image2_wh,
        empty_camera_0_wh=empty0_wh,
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
        raw_quality_gate=raw_quality_gate,
        drop_truncated=drop_truncated,
        success_only=bool(args.success_only),
        max_episodes=args.max_episodes,
        min_frames_per_episode=args.min_frames_per_episode,
        max_frames_per_episode=args.max_frames_per_episode,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


