#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import types

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
ANYPOS_ROOT = REPO_ROOT / "third_parties" / "AnyPos"
LIBERO_ROOT = REPO_ROOT / "third_parties" / "LIBERO"
LIBERO_PACKAGE_ROOT = LIBERO_ROOT / "libero"
LIBERO_LOCAL_CONFIG_ROOT = REPO_ROOT / ".libero_eval"

LIBERO_LOCAL_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_LOCAL_CONFIG_ROOT)

_libero_local_config = {
    "benchmark_root": str((LIBERO_PACKAGE_ROOT / "libero").resolve()),
    "bddl_files": str((LIBERO_PACKAGE_ROOT / "libero" / "bddl_files").resolve()),
    "init_states": str((LIBERO_PACKAGE_ROOT / "libero" / "init_files").resolve()),
    "datasets": str((REPO_ROOT / "data" / "LIBERO-datasets").resolve()),
    "assets": str((LIBERO_PACKAGE_ROOT / "libero" / "assets").resolve()),
}
with open(LIBERO_LOCAL_CONFIG_ROOT / "config.yaml", "w", encoding="utf-8") as _handle:
    yaml.safe_dump(_libero_local_config, _handle)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ANYPOS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYPOS_ROOT))


def _bootstrap_local_libero():
    source_root = str(LIBERO_PACKAGE_ROOT.resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    package_init = LIBERO_PACKAGE_ROOT / "libero" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "libero",
        package_init,
        submodule_search_locations=[str((LIBERO_PACKAGE_ROOT / "libero").resolve())],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["libero"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # LIBERO source imports both `libero.*` and `libero.libero.*`.
    # Create a package alias instead of aliasing the same module object,
    # otherwise `libero.envs` can re-enter itself through `libero.libero.envs`.
    compat_pkg = types.ModuleType("libero.libero")
    compat_pkg.__file__ = module.__file__
    compat_pkg.__path__ = list(module.__path__)
    compat_pkg.__package__ = "libero.libero"
    for key, value in module.__dict__.items():
        if key.startswith("__") and key not in {"__doc__"}:
            continue
        setattr(compat_pkg, key, value)
    sys.modules["libero.libero"] = compat_pkg
    setattr(module, "libero", compat_pkg)


_bootstrap_local_libero()

from idm.idm import IDM  # noqa: E402
from idm.preprocessor import DinoPreprocessor  # noqa: E402
from libero import get_libero_path  # noqa: E402
from libero.benchmark import get_benchmark  # noqa: E402
from lightewm.dataset.operators import ImageCropAndResize  # noqa: E402
from lightewm.runner.runner_util.wan_runtime import build_wan_i2v_pipeline_from_params  # noqa: E402
from lightewm.utils.data import save_video  # noqa: E402


DEFAULT_SUITES = ("libero_object", "libero_goal", "libero_spatial", "libero_10")
DEFAULT_HORIZONS = {
    "libero_object": 240,
    "libero_goal": 320,
    "libero_spatial": 240,
    "libero_10": 512,
}
DEFAULT_VIDEO_CKPT = "checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors"
DEFAULT_IDM_CKPT = "checkpoints/LIBERO-IDM/100000.pt"
DEFAULT_PROMPT_METADATA = "data/libero_i2v_train/metadata_dense_prompt.csv"
DEFAULT_INFER_CONFIG = "examples/LIBERO/infer_ti2v_5b.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LIBERO with Wan video generation + AnyPos IDM rollout."
    )
    parser.add_argument(
        "--suites",
        type=str,
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated suites to evaluate.",
    )
    parser.add_argument("--task-order-index", type=int, default=0, help="LIBERO benchmark task order index.")
    parser.add_argument("--tasks-per-suite", type=int, default=10, help="Evaluate the first N tasks per suite.")
    parser.add_argument("--trials-per-task", type=int, default=10, help="Trials per task.")
    parser.add_argument(
        "--video-ckpt",
        type=str,
        default=DEFAULT_VIDEO_CKPT,
        help="Wan LIBERO finetuned DiT checkpoint.",
    )
    parser.add_argument(
        "--idm-ckpt",
        type=str,
        default=DEFAULT_IDM_CKPT,
        help="IDM checkpoint trained on LIBERO absolute ee+gripper metadata.",
    )
    parser.add_argument(
        "--prompt-metadata-path",
        type=str,
        default=DEFAULT_PROMPT_METADATA,
        help="Dense-prompt CSV used to look up task prompts. Falls back to task language if absent.",
    )
    parser.add_argument(
        "--infer-config",
        type=str,
        default=DEFAULT_INFER_CONFIG,
        help="Base Wan inference YAML used for default infer kwargs.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/libero_video_idm_eval", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--idm-model-name", type=str, default="direction_aware", help="IDM model name.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--camera-height", type=int, default=128, help="LIBERO camera height.")
    parser.add_argument("--camera-width", type=int, default=128, help="LIBERO camera width.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Zero-action physics warmup steps after reset.")
    parser.add_argument(
        "--replan-every",
        type=int,
        default=49,
        help="Re-generate a video every N environment steps. Default 1 means every step.",
    )
    parser.add_argument(
        "--idm-frame-index",
        type=int,
        default=1,
        help="Frame index from the generated video used for IDM. 1 means the first predicted future frame.",
    )
    parser.add_argument("--position-action-limit", type=float, default=0.05, help="Max absolute delta xyz per step.")
    parser.add_argument("--orientation-action-limit", type=float, default=0.5, help="Max absolute delta axis-angle per step.")
    parser.add_argument(
        "--gripper-mode",
        type=str,
        default="binary",
        choices=["binary", "delta"],
        help="How to convert predicted absolute gripper state to env action.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.002,
        help="Threshold for binary gripper action.",
    )
    parser.add_argument(
        "--gripper-delta-scale",
        type=float,
        default=0.005,
        help="Scale for continuous gripper delta mode.",
    )
    parser.add_argument(
        "--save-debug-media",
        action="store_true",
        default=False,
        help="Save conditioning image and generated video for each trial.",
    )
    parser.add_argument(
        "--debug-media-max-per-task",
        type=int,
        default=1,
        help="Maximum number of trials with saved debug media per task.",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _normalize_task_key(name: str) -> str:
    text = Path(str(name)).stem
    if text.endswith("_demo"):
        text = text[: -len("_demo")]
    return text


def _load_wan_defaults(infer_config_path: str):
    with open(infer_config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = dict(config["model"]["params"])
    model["pipeline_class_path"] = config["model"]["class_path"]
    runner_params = dict(config["runner"]["params"])
    infer_kwargs = dict(runner_params.get("infer_kwargs", {}))
    return model, runner_params, infer_kwargs


def _load_dense_prompt_lookup(metadata_path: str):
    prompt_counts = defaultdict(Counter)
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        return {}

    with metadata_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("camera_key", "")).strip() not in ("", "agentview_rgb"):
                continue
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            source_file = str(row.get("source_file", "")).strip()
            if source_file:
                prompt_counts[_normalize_task_key(source_file)][prompt] += 1
            video_field = str(row.get("video", "")).strip()
            if video_field:
                prompt_counts[_normalize_task_key(video_field)][prompt] += 1

    lookup = {}
    for key, counter in prompt_counts.items():
        lookup[key] = counter.most_common(1)[0][0]
    return lookup


def _apply_orientation_fix(image: Image.Image) -> Image.Image:
    return ImageOps.flip(ImageOps.mirror(image))


def _obs_agentview_to_pil(obs) -> Image.Image:
    if "agentview_image" not in obs:
        raise KeyError("LIBERO observation missing 'agentview_image'.")
    image = Image.fromarray(np.asarray(obs["agentview_image"]).astype(np.uint8)).convert("RGB")
    return _apply_orientation_fix(image)


def _load_robosuite_transform_utils():
    try:
        import robosuite.utils.transform_utils as T

        return T
    except Exception:
        return None


def _quat_to_axis_angle(quat, transform_utils=None):
    quat = np.asarray(quat, dtype=np.float32)
    if transform_utils is not None:
        return np.asarray(transform_utils.quat2axisangle(quat), dtype=np.float32)

    # Fallback assumes robosuite quaternion order (x, y, z, w).
    xyz = quat[:3]
    w = float(quat[3])
    norm_xyz = float(np.linalg.norm(xyz))
    if norm_xyz < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / norm_xyz
    angle = 2.0 * np.arctan2(norm_xyz, w)
    return (axis * angle).astype(np.float32)


def _get_current_abs_state(obs, transform_utils=None):
    if "ee_states" in obs and "gripper_states" in obs:
        ee_state = np.asarray(obs["ee_states"], dtype=np.float32).reshape(-1)
        gripper = float(np.asarray(obs["gripper_states"], dtype=np.float32).reshape(-1)[0])
        return np.concatenate([ee_state[:6], np.array([gripper], dtype=np.float32)], axis=0)

    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_axis_angle = _quat_to_axis_angle(obs["robot0_eef_quat"], transform_utils=transform_utils)
    gripper = float(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[0])
    return np.concatenate([eef_pos[:3], eef_axis_angle[:3], np.array([gripper], dtype=np.float32)], axis=0)


def _absolute_state_to_env_action(
    predicted_abs_state,
    current_abs_state,
    position_action_limit,
    orientation_action_limit,
    gripper_mode,
    gripper_threshold,
    gripper_delta_scale,
):
    predicted_abs_state = np.asarray(predicted_abs_state, dtype=np.float32).reshape(-1)
    current_abs_state = np.asarray(current_abs_state, dtype=np.float32).reshape(-1)
    pos_delta = np.clip(
        predicted_abs_state[:3] - current_abs_state[:3],
        -float(position_action_limit),
        float(position_action_limit),
    )
    ori_delta = np.clip(
        predicted_abs_state[3:6] - current_abs_state[3:6],
        -float(orientation_action_limit),
        float(orientation_action_limit),
    )
    gripper_delta = float(predicted_abs_state[6] - current_abs_state[6])
    if gripper_mode == "binary":
        if gripper_delta > gripper_threshold:
            gripper_action = 1.0
        elif gripper_delta < -gripper_threshold:
            gripper_action = -1.0
        else:
            gripper_action = 0.0
    else:
        gripper_action = float(np.clip(gripper_delta / max(gripper_delta_scale, 1e-6), -1.0, 1.0))
    return np.concatenate([pos_delta, ori_delta, np.array([gripper_action], dtype=np.float32)], axis=0)


def _load_idm(idm_ckpt_path: str, model_name: str, device: str):
    try:
        checkpoint = torch.load(idm_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(idm_ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    train_mean = state_dict["train_mean"]
    output_dim = int(train_mean.numel())
    net = IDM(
        model_name=model_name,
        dinov2_name="facebook/dinov2-with-registers-base",
        output_dim=output_dim,
        train_mean=state_dict.get("train_mean"),
        train_std=state_dict.get("train_std"),
    )
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net.to(device)
    return net


def _run_idm_on_frame(idm, preprocessor, frame, device: str):
    if isinstance(frame, Image.Image):
        frame = np.asarray(frame.convert("RGB"))
    batch = preprocessor.process_batch([frame]).to(device)
    with torch.no_grad():
        prediction = idm(batch)[0]
    return prediction.detach().cpu().numpy().astype(np.float32)


def _select_generated_frame(video_frames, frame_index: int):
    if len(video_frames) == 0:
        raise RuntimeError("Video model returned zero frames.")
    clamped = max(0, min(int(frame_index), len(video_frames) - 1))
    frame = video_frames[clamped]
    if isinstance(frame, Image.Image):
        return frame.convert("RGB"), clamped
    return Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB"), clamped


def _make_env(task, camera_height: int, camera_width: int):
    try:
        from libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import LIBERO environment dependencies. "
            "Install the LIBERO / robosuite runtime before running this evaluator."
        ) from exc

    env_args = {
        "bddl_file_name": os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        ),
        "camera_heights": int(camera_height),
        "camera_widths": int(camera_width),
    }
    return OffScreenRenderEnv(**env_args)


def _load_libero_init_states(task):
    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    try:
        return torch.load(str(init_states_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(init_states_path), map_location="cpu")


def _build_trial_seed(base_seed: int, suite_idx: int, task_idx: int, trial_idx: int, step_idx: int):
    return int(base_seed + suite_idx * 1_000_000 + task_idx * 10_000 + trial_idx * 100 + step_idx)


def main():
    args = parse_args()

    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    if len(suites) == 0:
        raise ValueError("No suites requested.")

    output_dir = Path(_resolve_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    video_ckpt_path = _resolve_path(args.video_ckpt)
    idm_ckpt_path = _resolve_path(args.idm_ckpt)
    infer_config_path = _resolve_path(args.infer_config)
    prompt_metadata_path = _resolve_path(args.prompt_metadata_path)

    model_params, infer_runner_params, infer_kwargs = _load_wan_defaults(infer_config_path)
    model_paths = list(model_params["model_paths"])
    model_paths[0] = video_ckpt_path
    model_params["model_paths"] = model_paths
    model_params["device"] = args.device
    model_params["torch_dtype"] = "bfloat16"

    dense_prompt_lookup = _load_dense_prompt_lookup(prompt_metadata_path)
    transform_utils = _load_robosuite_transform_utils()

    print(
        {
            "suites": suites,
            "tasks_per_suite": args.tasks_per_suite,
            "trials_per_task": args.trials_per_task,
            "video_ckpt": video_ckpt_path,
            "idm_ckpt": idm_ckpt_path,
            "prompt_metadata_path": prompt_metadata_path,
            "device": args.device,
            "infer_kwargs": infer_kwargs,
            "note": "Evaluation uses current agentview -> Wan future video -> IDM absolute ee+gripper -> OSC_POSE delta action.",
        }
    )

    wan_pipe = build_wan_i2v_pipeline_from_params(model_params)
    idm = _load_idm(idm_ckpt_path, model_name=args.idm_model_name, device=args.device)
    preprocessor = DinoPreprocessor(SimpleNamespace(use_transform=False))
    input_image_resizer = ImageCropAndResize(
        height=int(infer_kwargs["height"]),
        width=int(infer_kwargs["width"]),
        max_pixels=None,
        height_division_factor=16,
        width_division_factor=16,
        resize_mode=str(infer_runner_params.get("input_image_resize_mode", "letterbox")),
    )

    summary = {
        "video_ckpt": video_ckpt_path,
        "idm_ckpt": idm_ckpt_path,
        "prompt_metadata_path": prompt_metadata_path,
        "suites": {},
    }

    total_successes = 0
    total_trials = 0

    for suite_idx, suite_name in enumerate(suites):
        benchmark = get_benchmark(suite_name)(args.task_order_index)
        horizon = DEFAULT_HORIZONS[suite_name]
        suite_record = {
            "horizon": int(horizon),
            "num_tasks": min(int(args.tasks_per_suite), benchmark.get_num_tasks()),
            "tasks": [],
        }
        suite_successes = 0
        suite_trials = 0

        task_iterator = range(min(int(args.tasks_per_suite), benchmark.get_num_tasks()))
        for task_idx in task_iterator:
            task = benchmark.get_task(task_idx)
            init_states = _load_libero_init_states(task)
            prompt = dense_prompt_lookup.get(_normalize_task_key(task.name), task.language)

            task_record = {
                "task_index": int(task_idx),
                "task_name": task.name,
                "language": task.language,
                "prompt": prompt,
                "trials": [],
            }
            task_successes = 0
            debug_saved = 0

            trial_bar = tqdm(
                range(int(args.trials_per_task)),
                desc=f"{suite_name}:{task_idx:02d}",
                leave=False,
            )
            for trial_idx in trial_bar:
                init_state = init_states[trial_idx % len(init_states)]
                if hasattr(init_state, "cpu"):
                    init_state = init_state.cpu().numpy()
                else:
                    init_state = np.asarray(init_state)

                env = _make_env(task, args.camera_height, args.camera_width)
                env.reset()
                env.seed(args.seed + suite_idx * 1000 + task_idx * 100 + trial_idx)
                obs = env.set_init_state(init_state)

                for _ in range(int(args.warmup_steps)):
                    obs, _, done, _ = env.step(np.zeros(7, dtype=np.float32))
                    if bool(done):
                        break

                success = bool(done) if "done" in locals() else False
                step_count = 0
                replan_count = 0
                should_save_debug = bool(args.save_debug_media and debug_saved < int(args.debug_media_max_per_task))
                gt_rollout_frames = []
                replan_video_records = []
                cached_prediction = None
                cached_generated_video = None
                cached_selected_frame_index = None
                cached_seed = None

                if should_save_debug:
                    gt_rollout_frames.append(_obs_agentview_to_pil(obs))

                step_bar = tqdm(
                    total=int(horizon),
                    desc=f"{suite_name}:{task_idx:02d}:trial{trial_idx:02d}:step",
                    leave=False,
                )
                try:
                    while step_count < horizon and not success:
                        if step_count % int(args.replan_every) == 0 or cached_prediction is None:
                            conditioned_image = _obs_agentview_to_pil(obs)
                            input_image = input_image_resizer(conditioned_image)
                            seed = _build_trial_seed(args.seed, suite_idx, task_idx, trial_idx, step_count)
                            generated_video = wan_pipe(
                                prompt=prompt,
                                input_image=input_image,
                                seed=seed,
                                **infer_kwargs,
                            )
                            idm_frame, selected_frame_index = _select_generated_frame(
                                generated_video,
                                args.idm_frame_index,
                            )
                            cached_prediction = _run_idm_on_frame(
                                idm=idm,
                                preprocessor=preprocessor,
                                frame=idm_frame,
                                device=args.device,
                            )
                            cached_generated_video = generated_video
                            cached_selected_frame_index = selected_frame_index
                            cached_seed = int(seed)
                            if should_save_debug:
                                replan_video_records.append(
                                    {
                                        "step": int(step_count),
                                        "seed": int(seed),
                                        "selected_generated_frame_index": int(selected_frame_index),
                                        "video": generated_video,
                                    }
                                )
                            replan_count += 1

                        current_abs_state = _get_current_abs_state(obs, transform_utils=transform_utils)
                        action = _absolute_state_to_env_action(
                            predicted_abs_state=cached_prediction,
                            current_abs_state=current_abs_state,
                            position_action_limit=args.position_action_limit,
                            orientation_action_limit=args.orientation_action_limit,
                            gripper_mode=args.gripper_mode,
                            gripper_threshold=args.gripper_threshold,
                            gripper_delta_scale=args.gripper_delta_scale,
                        )

                        obs, _, done, _ = env.step(action.astype(np.float32))
                        success = bool(done)
                        step_count += 1
                        if should_save_debug:
                            gt_rollout_frames.append(_obs_agentview_to_pil(obs))
                        step_bar.update(1)
                        step_bar.set_postfix(replans=int(replan_count), success=bool(success))
                finally:
                    step_bar.close()

                if should_save_debug:
                    debug_dir = output_dir / "debug_media" / suite_name / f"task_{task_idx:02d}" / f"trial_{trial_idx:02d}"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    _obs_agentview_to_pil(obs).save(debug_dir / "final_obs.png")

                    if len(gt_rollout_frames) > 0:
                        save_video(
                            gt_rollout_frames,
                            str(debug_dir / "gt_rollout.mp4"),
                            fps=int(infer_runner_params.get("fps", 10)),
                            quality=int(infer_runner_params.get("quality", 5)),
                        )

                    replan_saved = []
                    for replan_idx, replan_record in enumerate(replan_video_records):
                        video_file = f"generated_replan_{replan_idx:03d}_step_{int(replan_record['step']):04d}.mp4"
                        save_video(
                            replan_record["video"],
                            str(debug_dir / video_file),
                            fps=int(infer_runner_params.get("fps", 10)),
                            quality=int(infer_runner_params.get("quality", 5)),
                        )
                        replan_saved.append(
                            {
                                "replan_index": int(replan_idx),
                                "step": int(replan_record["step"]),
                                "seed": int(replan_record["seed"]),
                                "selected_generated_frame_index": int(
                                    replan_record["selected_generated_frame_index"]
                                ),
                                "video_file": video_file,
                            }
                        )

                    meta = {
                        "prompt": prompt,
                        "success": bool(success),
                        "steps": int(step_count),
                        "last_selected_generated_frame_index": (
                            None if cached_selected_frame_index is None else int(cached_selected_frame_index)
                        ),
                        "last_seed": (None if cached_seed is None else int(cached_seed)),
                        "num_replans": int(replan_count),
                        "replan_videos": replan_saved,
                        "gt_rollout_video_file": ("gt_rollout.mp4" if len(gt_rollout_frames) > 0 else None),
                        "last_predicted_abs_action": (
                            None if cached_prediction is None else np.asarray(cached_prediction).tolist()
                        ),
                    }
                    with open(debug_dir / "meta.json", "w", encoding="utf-8") as handle:
                        json.dump(meta, handle, indent=2)
                    debug_saved += 1

                env.close()

                task_successes += int(success)
                suite_successes += int(success)
                total_successes += int(success)
                suite_trials += 1
                total_trials += 1
                task_record["trials"].append(
                    {
                        "trial_index": int(trial_idx),
                        "success": bool(success),
                        "steps": int(step_count),
                    }
                )
                task_record["success_rate"] = float(task_successes / max(1, len(task_record["trials"])))
                trial_bar.set_postfix(success_rate=f"{task_record['success_rate']:.3f}")

            task_record["num_successes"] = int(task_successes)
            task_record["num_trials"] = int(args.trials_per_task)
            task_record["success_rate"] = float(task_successes / max(1, int(args.trials_per_task)))
            suite_record["tasks"].append(task_record)

        suite_record["num_successes"] = int(suite_successes)
        suite_record["num_trials"] = int(suite_trials)
        suite_record["success_rate"] = float(suite_successes / max(1, suite_trials))
        summary["suites"][suite_name] = suite_record
        print(
            f"[Suite] {suite_name}: "
            f"{suite_successes}/{suite_trials} success, rate={suite_record['success_rate']:.4f}"
        )

    summary["overall_num_successes"] = int(total_successes)
    summary["overall_num_trials"] = int(total_trials)
    summary["overall_success_rate"] = float(total_successes / max(1, total_trials))

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[Overall] {total_successes}/{total_trials} success, rate={summary['overall_success_rate']:.4f}")
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
