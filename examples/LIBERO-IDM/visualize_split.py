import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_parties.AnyPos.idm.libero_abs_ee_dataset import (
    DEFAULT_SUITES,
    LiberoAbsoluteEEDataset,
    extract_official_split_regions,
)


def draw_split_lines(image: Image.Image, arm_boxes: dict):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for x in [int(arm_boxes["left_split"]), int(arm_boxes["right_split"]), int(arm_boxes["gripper_split"])]:
        draw.line([(x, 0), (x, h)], fill="red", width=2)
    y = int(arm_boxes["arm_gripper_split"])
    draw.line([(0, y), (w, y)], fill="blue", width=2)
    return image


def make_canvas(images):
    pad = 12
    panel_w = max(image.width for image in images)
    panel_h = max(image.height for image in images)
    canvas = Image.new("RGB", (panel_w * len(images) + pad * (len(images) + 1), panel_h + pad * 2), (255, 255, 255))
    for idx, image in enumerate(images):
        x = pad + idx * (panel_w + pad)
        canvas.paste(image, (x, pad))
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize official AnyPos split on LIBERO frames.")
    parser.add_argument("--libero-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--suites", type=str, default=",".join(DEFAULT_SUITES))
    parser.add_argument("--camera-key", type=str, default="agentview_rgb")
    parser.add_argument("--target-key", type=str, default="ee_states")
    parser.add_argument("--frame-stride", type=int, default=16)
    parser.add_argument("--min-episode-len", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    dataset = LiberoAbsoluteEEDataset(
        libero_root=args.libero_root,
        split=args.split,
        suites=suites,
        camera_key=args.camera_key,
        target_key=args.target_key,
        frame_stride=args.frame_stride,
        min_episode_len=args.min_episode_len,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = min(args.num_samples, len(dataset))
    for idx in range(count):
        _, image_np = dataset[idx]
        meta = dataset.get_metadata(idx)
        flipped_image, arm_boxes, regions = extract_official_split_regions(image_np)
        overview = draw_split_lines(flipped_image.copy(), arm_boxes)
        canvas = make_canvas([overview] + regions)
        name = (
            f"{idx:04d}__{meta['suite']}__{meta['task_name']}__"
            f"{meta['demo_key']}__frame{meta['frame_index']:04d}.png"
        )
        canvas.save(output_dir / name)

    print(f"Saved {count} visualization(s) to {output_dir}")


if __name__ == "__main__":
    main()
