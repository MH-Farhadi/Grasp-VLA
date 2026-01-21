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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

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
    # `gripper_action` is sometimes missing or serialized as a string (e.g. "0.0000").
    # Keep parsing robust even though we may override it downstream depending on `gripper_label_mode`.
    g_raw = action_from_prev.get("gripper_action", 0)
    try:
        g = int(round(_to_float(g_raw)))
    except Exception:
        g = 0
    if len(dp) != 3 or len(dr) != 3:
        raise ValueError(f"Unexpected action shape: dp={dp}, dr={dr}")
    return (dp[0], dp[1], dp[2], dr[0], dr[1], dr[2], float(g))


def _bump(counter: Dict[str, int], key: str, n: int = 1) -> None:
    counter[key] = int(counter.get(key, 0)) + int(n)

def _norm_gripper_state(s: Any) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s in ("open", "opened"):
        return "open"
    if s in ("close", "closed"):
        return "close"
    return None


def _infer_gripper_action(
    *,
    prev_tick: Dict[str, Any],
    cur_tick: Dict[str, Any],
    action_from_prev: Optional[Dict[str, Any]],
    mode: Literal["action_from_prev", "state_change", "hybrid"],
) -> int:
    """
    Infer a discrete gripper action in {-1, 0, +1} where:
      -1 => close, +1 => open

    Why: some episodes/sessions have sparse or missing `policy.action_from_prev.gripper_action`.
    We can recover an equivalent impulse label from:
      - gripper state transitions (when available), and
      - joystick gripper commands (common in this repo's logs).
    """
    if mode not in ("action_from_prev", "state_change", "hybrid"):
        mode = "action_from_prev"

    ga = 0
    if action_from_prev is not None and mode in ("action_from_prev", "hybrid"):
        try:
            ga = int(action_from_prev.get("gripper_action", 0))
        except Exception:
            ga = 0
        if ga != 0:
            return ga

    if mode in ("state_change", "hybrid"):
        prev_state = _norm_gripper_state(((prev_tick.get("robot") or {}).get("gripper") or {}).get("state"))
        cur_state = _norm_gripper_state(((cur_tick.get("robot") or {}).get("gripper") or {}).get("state"))
        if prev_state is None or cur_state is None:
            prev_state = None
            cur_state = None
        if prev_state == cur_state:
            prev_state = None
            cur_state = None
        if prev_state is not None and cur_state is not None and prev_state != cur_state:
            return 1 if cur_state == "open" else -1

    # Hybrid fallback: infer an impulse from user joystick gripper commands.
    if mode == "hybrid":
        def _read_user_gripper_cmd(tick: Dict[str, Any]) -> Optional[float]:
            user = tick.get("user") or {}
            js = user.get("joystick") or {}
            if "gripper_cmd" in js:
                try:
                    return _to_float(js.get("gripper_cmd"))
                except Exception:
                    return None
            cmd7 = js.get("cartesian_vel_cmd_7d")
            if isinstance(cmd7, (list, tuple)) and len(cmd7) >= 7:
                try:
                    return _to_float(cmd7[6])
                except Exception:
                    return None
            return None

        def _disc_cmd(x: float, *, thr: float = 0.5) -> int:
            if x >= thr:
                return 1
            if x <= -thr:
                return -1
            return 0

        prev_cmd = _read_user_gripper_cmd(prev_tick)
        cur_cmd = _read_user_gripper_cmd(cur_tick)
        if prev_cmd is not None and cur_cmd is not None:
            prev_d = _disc_cmd(float(prev_cmd))
            cur_d = _disc_cmd(float(cur_cmd))

            # Prefer not to label when the deadman isn't held, but keep best-effort behavior if missing.
            deadman_val = (cur_tick.get("user") or {}).get("deadman", None)
            deadman_ok = True if deadman_val is None else bool(deadman_val)
            if deadman_ok and cur_d != 0 and cur_d != prev_d:
                return cur_d

    return 0


def _canonical_task_from_instruction(instr: Dict[str, Any]) -> Optional[str]:
    meta = instr.get("language_command_meta") or {}
    color = meta.get("color")
    box_idx = meta.get("box_idx")
    if isinstance(color, str) and isinstance(box_idx, (int, float)):
        return f"Pick up the {color.strip().lower()} box {int(box_idx)}."
    # fallback: try target_label like "red box 1"
    target_label = instr.get("target_label")
    if isinstance(target_label, str) and target_label.strip():
        return f"Pick up the {target_label.strip().lower()}."
    return None


def _episode_motion_stats(actions_7d: List[Tuple[float, ...]]) -> Tuple[float, float]:
    """Return (sum_dpos_norm_m, sum_drot_norm_rad) over the episode."""
    if not actions_7d:
        return 0.0, 0.0
    dp = np.asarray([a[:3] for a in actions_7d], dtype=np.float32)
    dr = np.asarray([a[3:6] for a in actions_7d], dtype=np.float32)
    sum_dp = float(np.linalg.norm(dp, axis=1).sum()) if dp.size else 0.0
    sum_dr = float(np.linalg.norm(dr, axis=1).sum()) if dr.size else 0.0
    return sum_dp, sum_dr


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
    # Task/text handling
    canonicalize_task: bool = False,
    # Gripper supervision handling
    gripper_label_mode: Literal["action_from_prev", "state_change", "hybrid"] = "action_from_prev",
    keep_gripperless_episodes: bool = True,
    oversample_gripper_episodes: int = 1,
    # Quality curation (beyond raw_quality_gate)
    drop_idle_episodes: bool = False,
    idle_min_sum_dpos_m: float = 0.02,
    idle_min_sum_drot_rad: float = 0.2,
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
    skipped_reasons: Dict[str, int] = {}
    expected_robot_sig: Optional[Tuple[str, str, str]] = None

    for ep_dir in episode_dirs:
        instr_path = ep_dir / "instruction.json"
        ticks_path = ep_dir / "ticks.jsonl"
        events_path = ep_dir / "events.jsonl"
        meta_path = ep_dir / "metadata.json"
        if not instr_path.exists():
            skipped_eps += 1
            _bump(skipped_reasons, "missing_instruction_json")
            continue
        if not ticks_path.exists():
            skipped_eps += 1
            _bump(skipped_reasons, "missing_ticks_jsonl")
            continue
        if raw_quality_gate:
            # Episode completeness checks.
            if not events_path.exists():
                skipped_eps += 1
                _bump(skipped_reasons, "missing_events_jsonl")
                continue
            if not meta_path.exists():
                skipped_eps += 1
                _bump(skipped_reasons, "missing_metadata_json")
                continue
            if not (ep_dir / "images").exists():
                skipped_eps += 1
                _bump(skipped_reasons, "missing_images_dir")
                continue

        try:
            instr_obj = _read_json(instr_path)
        except Exception:
            skipped_eps += 1
            _bump(skipped_reasons, "invalid_instruction_json")
            continue
        instruction = (instr_obj.get("language_command") or "").strip()
        if canonicalize_task:
            canon = _canonical_task_from_instruction(instr_obj)
            if canon:
                instruction = canon
        if not instruction:
            skipped_eps += 1
            _bump(skipped_reasons, "missing_instruction_text")
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
            except RuntimeError:
                skipped_eps += 1
                _bump(skipped_reasons, "metadata_mismatch")
                continue
            except Exception:
                skipped_eps += 1
                _bump(skipped_reasons, "invalid_metadata_json")
                continue

        if raw_quality_gate:
            try:
                is_truncated, has_success = _summarize_episode_events(events_path)
            except Exception:
                skipped_eps += 1
                _bump(skipped_reasons, "invalid_events_jsonl")
                continue
            if drop_truncated and is_truncated:
                skipped_eps += 1
                _bump(skipped_reasons, "truncated_episode")
                continue
            if success_only and not has_success:
                skipped_eps += 1
                _bump(skipped_reasons, "not_success")
                continue

        # Pass 1: parse ticks into lightweight per-frame specs (no dataset writes yet).
        frame_specs: List[Tuple[str, np.ndarray, np.ndarray]] = []  # (rel_img_path, state, action)
        prev_tick: Optional[Dict[str, Any]] = None
        expected_tick_idx = 0
        malformed_reason: Optional[str] = None
        ep_actions_7d: List[Tuple[float, ...]] = []
        ep_nonzero_gripper = 0
        frame_skip_reasons: Dict[str, int] = {}

        try:
            for cur in _iter_jsonl(ticks_path):
                if raw_quality_gate:
                    tick_idx = cur.get("tick_idx")
                    if tick_idx is None or int(tick_idx) != expected_tick_idx:
                        malformed_reason = "nonsequential_tick_idx"
                        break
                    if expected_tick_idx == 0:
                        afp0 = (cur.get("policy") or {}).get("action_from_prev")
                        if afp0 is not None:
                            malformed_reason = "first_tick_has_action_from_prev"
                            break
                    expected_tick_idx += 1

                if prev_tick is None:
                    prev_tick = cur
                    continue

                # Alignment rule: observation from prev tick, action label from current tick.
                afp = (cur.get("policy") or {}).get("action_from_prev")
                if afp is None:
                    _bump(frame_skip_reasons, "missing_action_from_prev")
                    prev_tick = cur
                    continue

                rel_img = (prev_tick.get("image") or {}).get("path")
                if not rel_img:
                    _bump(frame_skip_reasons, "missing_image_path")
                    prev_tick = cur
                    continue
                img_path = ep_dir / rel_img
                if not img_path.exists():
                    _bump(frame_skip_reasons, "missing_image_file")
                    prev_tick = cur
                    continue

                # Continuous arm action from logs + robust discrete gripper label.
                a7_base = _action_from_prev_to_7d(afp)
                g_disc = _infer_gripper_action(
                    prev_tick=prev_tick, cur_tick=cur, action_from_prev=afp, mode=gripper_label_mode
                )
                a7 = (a7_base[0], a7_base[1], a7_base[2], a7_base[3], a7_base[4], a7_base[5], float(g_disc))
                ep_actions_7d.append(a7)
                if g_disc != 0:
                    ep_nonzero_gripper += 1
                if raw_quality_gate:
                    # Magnitude sanity (catch unit bugs / corruption): these thresholds are intentionally generous.
                    dp = np.asarray(a7[:3], dtype=np.float32)
                    dr = np.asarray(a7[3:6], dtype=np.float32)
                    if np.linalg.norm(dp) > 0.25 or np.linalg.norm(dr) > 3.5:
                        malformed_reason = "action_outlier"
                        break

                action = _pad_action(a7, action_dim=action_dim)
                state = _build_state_from_tick(prev_tick, state_dim=state_dim)

                if raw_quality_gate:
                    if not np.isfinite(action).all() or not np.isfinite(state).all():
                        malformed_reason = "nonfinite_action_or_state"
                        break

                frame_specs.append((rel_img, state, action))
                prev_tick = cur

                if max_frames_per_episode is not None and len(frame_specs) >= int(max_frames_per_episode):
                    break
        except Exception:
            malformed_reason = "ticks_parse_error"

        if malformed_reason is not None:
            skipped_eps += 1
            _bump(skipped_reasons, malformed_reason)
            continue

        n_added = len(frame_specs)
        if n_added <= 0:
            skipped_eps += 1
            _bump(skipped_reasons, "no_frames")
            if frame_skip_reasons:
                top_reason = sorted(frame_skip_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                _bump(skipped_reasons, f"no_frames__{top_reason}")
            continue

        if min_frames_per_episode is not None and n_added < int(min_frames_per_episode):
            skipped_eps += 1
            _bump(skipped_reasons, "too_short")
            continue

        # Extra curation: drop idle episodes (common in "deadman not held"/paused demos)
        if drop_idle_episodes:
            sum_dp, sum_dr = _episode_motion_stats(ep_actions_7d)
            if (
                (sum_dp < float(idle_min_sum_dpos_m))
                and (sum_dr < float(idle_min_sum_drot_rad))
                and (ep_nonzero_gripper == 0)
            ):
                skipped_eps += 1
                _bump(skipped_reasons, "idle_episode")
                continue

        # Optional: skip episodes with no grasp supervision at all
        if (not keep_gripperless_episodes) and (ep_nonzero_gripper == 0):
            skipped_eps += 1
            _bump(skipped_reasons, "no_gripper_events")
            continue

        # Oversample episodes that contain gripper events (increases effective grasp signal)
        rep = 1
        if ep_nonzero_gripper > 0:
            rep = max(1, int(oversample_gripper_episodes))

        # Pass 2: actually write episode(s) to disk.
        for _ in range(rep):
            try:
                for rel_img, state, action in frame_specs:
                    img_path = ep_dir / rel_img
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

                ds.save_episode()
                total_eps += 1
                total_frames += n_added
            except Exception:
                skipped_eps += 1
                _bump(skipped_reasons, "write_episode_error")
                try:
                    ds.clear_episode_buffer(delete_images=True)
                except Exception:
                    pass
                break

    ds.finalize()
    print(f"Saved LeRobotDataset to: {out_dir}")
    print(f"Converted episodes: {total_eps} | frames: {total_frames} | skipped episodes: {skipped_eps}")
    if skipped_reasons:
        print("Skip reasons (counts):")
        for k in sorted(skipped_reasons.keys()):
            print(f"  - {k}: {int(skipped_reasons[k])}")
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
    # Task/text handling
    p.add_argument(
        "--canonicalize_task",
        action="store_true",
        default=False,
        help="Rewrite task strings to a canonical template using instruction.json language_command_meta (reduces prompt entropy).",
    )
    # Gripper supervision handling
    p.add_argument(
        "--gripper_label_mode",
        type=str,
        default="action_from_prev",
        choices=["action_from_prev", "state_change", "hybrid"],
        help="How to label the gripper action: from logs, from state transitions, or hybrid (prefer logs, else state_change).",
    )
    p.add_argument(
        "--keep_gripperless_episodes",
        action="store_true",
        default=True,
        help="Keep episodes with no non-zero gripper events (recommended if you still want reaching skill).",
    )
    p.add_argument(
        "--drop_gripperless_episodes",
        action="store_true",
        help="Drop episodes with no non-zero gripper events (useful to build a grasp-focused dataset).",
    )
    p.add_argument(
        "--oversample_gripper_episodes",
        type=int,
        default=1,
        help="Duplicate episodes that contain any gripper close/open events by this factor (boost grasp signal).",
    )
    # Quality curation
    p.add_argument(
        "--drop_idle_episodes",
        action="store_true",
        default=False,
        help="Drop episodes with near-zero total motion and no gripper events (filters 'sleeping' demos).",
    )
    p.add_argument("--idle_min_sum_dpos_m", type=float, default=0.02, help="Idle filter: min total |dpos| sum (m)")
    p.add_argument("--idle_min_sum_drot_rad", type=float, default=0.2, help="Idle filter: min total |drot| sum (rad)")
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
        canonicalize_task=bool(args.canonicalize_task),
        gripper_label_mode=str(getattr(args, "gripper_label_mode", "action_from_prev")),
        keep_gripperless_episodes=bool(getattr(args, "keep_gripperless_episodes", True))
        and not bool(getattr(args, "drop_gripperless_episodes", False)),
        oversample_gripper_episodes=int(getattr(args, "oversample_gripper_episodes", 1)),
        drop_idle_episodes=bool(getattr(args, "drop_idle_episodes", False)),
        idle_min_sum_dpos_m=float(getattr(args, "idle_min_sum_dpos_m", 0.02)),
        idle_min_sum_drot_rad=float(getattr(args, "idle_min_sum_drot_rad", 0.2)),
        max_episodes=args.max_episodes,
        min_frames_per_episode=args.min_frames_per_episode,
        max_frames_per_episode=args.max_frames_per_episode,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


