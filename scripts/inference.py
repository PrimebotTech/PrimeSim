#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACTION_DIM = 31
NUM_VIEWS = 3
HEIGHT = 192
WIDTH = 256
MODEL_FPS = 10
OUTPUT_FPS = 30
OPENING_FRAMES = 33
CONTINUATION_FRAMES = 37
CONSISTENCY_STEPS = (1000, 750, 500, 250)
LOCAL_ATTN_SIZE = 21
SINK_SIZE = 3
SEED = 42
PROMPT = "Complete the bimanual manipulation task."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--text-encoder", required=True, type=Path)
    parser.add_argument("--vae", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--demo-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_file(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_dir(path):
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path


def load_demo(demo_dir):
    import numpy as np
    import torch
    from PIL import Image

    image = np.asarray(Image.open(demo_dir / "first_frame.png").convert("RGB"))
    expected_shape = (NUM_VIEWS * HEIGHT, WIDTH, 3)
    if image.shape != expected_shape:
        raise ValueError(
            f"Expected first_frame.png shape {expected_shape}, got {image.shape}"
        )
    image = image.reshape(NUM_VIEWS, HEIGHT, WIDTH, 3)
    first_frame = torch.from_numpy(image.copy()).permute(0, 3, 1, 2).float()
    first_frame = first_frame.div(127.5).sub(1.0).unsqueeze(0)

    actions = np.load(demo_dir / "actions.npy", allow_pickle=False)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"Expected actions.npy shape [T, {ACTION_DIM}], got {actions.shape}"
        )
    if not len(actions) or not np.isfinite(actions).all():
        raise ValueError("actions.npy is empty or contains NaN/Inf")

    model_frames = len(actions) + 1
    if model_frames <= OPENING_FRAMES:
        segments = 1
    else:
        continuation = CONTINUATION_FRAMES - 1
        segments = (
            1 + (model_frames - OPENING_FRAMES + continuation - 1) // continuation
        )

    opening_actions = OPENING_FRAMES - 1
    continuation_actions = CONTINUATION_FRAMES - 1
    lengths = [opening_actions] + [continuation_actions] * (segments - 1)
    padded_actions = sum(lengths)
    if len(actions) < padded_actions:
        actions = np.concatenate(
            [
                actions,
                np.repeat(actions[-1:], padded_actions - len(actions), axis=0),
            ],
            axis=0,
        )
    actions = torch.from_numpy(actions[:padded_actions].astype(np.float32, copy=False))
    return {
        "actions": list(actions.split(lengths)),
        "first_frame": first_frame,
        "model_frames": model_frames,
    }


def build_pipeline(args):
    import torch
    from primesim.pipeline import PrimeSimPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pipe = PrimeSimPipeline.from_pretrained(
        text_encoder_path=str(args.text_encoder),
        vae_path=str(args.vae),
        tokenizer_path=str(args.tokenizer),
        action_dim=ACTION_DIM,
        checkpoint_path=str(args.checkpoint),
        causal=True,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    pipe.consistency_steps = CONSISTENCY_STEPS
    return pipe


def generate(pipe, demo):
    import torch
    from einops import rearrange

    first_action = (
        demo["actions"][0]
        .unsqueeze(0)
        .to(
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
    )
    prepared = {
        "prompt": [PROMPT],
        "batch_size": 1,
        "input_images": demo["first_frame"],
        "input_videos": None,
        "seed": SEED,
        "rand_device": "cpu",
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": OPENING_FRAMES,
        "action": first_action,
    }
    pipe.units(prepared)
    context = prepared["context"].detach()
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    latents = []
    kv_cache = None
    sequence_start_frame = 0
    for index, action in enumerate(demo["actions"]):
        if index == 0:
            noise = prepared["latents"]
            first_frame = prepared["first_frame_latents"]
        else:
            shape = (
                1,
                prepared["latents"].shape[1],
                prepared["latents"].shape[2],
                (CONTINUATION_FRAMES - 1) // 4,
                prepared["latents"].shape[4],
                prepared["latents"].shape[5],
            )
            noise = pipe.generate_noise(
                shape,
                seed=SEED + index,
                rand_device="cpu",
            )
            first_frame = None
        segment, kv_cache, _ = pipe.self_forcing_segment(
            pipe.dit,
            noise,
            context,
            action.unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype),
            first_frame,
            CONSISTENCY_STEPS,
            with_grad=False,
            kv_cache=kv_cache,
            sequence_start_frame=sequence_start_frame,
            action_target_offset=1 if index == 0 else 0,
            sigma_shift=5.0,
            local_attn_size=LOCAL_ATTN_SIZE,
            sink_size=SINK_SIZE,
            last_step_only=True,
        )
        latents.append(segment)
        sequence_start_frame += int(noise.shape[3])
        print(f"Generated segment {index + 1}/{len(demo['actions'])}", flush=True)

    latents = torch.cat(latents, dim=3)
    pipe.dit.to("cpu")
    torch.cuda.empty_cache()
    batch_size, num_views = latents.shape[:2]
    decoded = pipe.vae.decode(
        rearrange(latents, "b v c t h w -> (b v) c t h w"),
        device=pipe.device,
    )
    decoded = rearrange(
        decoded,
        "(b v) c t h w -> b c t (v h) w",
        b=batch_size,
        v=num_views,
    )
    return pipe.vae_output_to_video(decoded)[: demo["model_frames"]]


def save_video(frames, output):
    import imageio.v2 as imageio
    import numpy as np

    repeat = OUTPUT_FPS // MODEL_FPS
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        output,
        [np.asarray(frame) for frame in frames for _ in range(repeat)],
        fps=OUTPUT_FPS,
        codec="libx264",
        macro_block_size=None,
    )


def main():
    args = parse_args()
    args.checkpoint = require_file(args.checkpoint)
    args.text_encoder = require_file(args.text_encoder)
    args.vae = require_file(args.vae)
    args.tokenizer = require_dir(args.tokenizer)
    args.demo_dir = require_dir(args.demo_dir)
    args.output = args.output.expanduser().resolve()
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")

    demo = load_demo(args.demo_dir)
    print(
        json.dumps(
            {
                "action_dim": ACTION_DIM,
                "model_frames": demo["model_frames"],
                "model_fps": MODEL_FPS,
                "output_frames": demo["model_frames"] * OUTPUT_FPS // MODEL_FPS,
                "output_fps": OUTPUT_FPS,
            },
            indent=2,
        ),
        flush=True,
    )

    import torch

    pipe = build_pipeline(args)
    with torch.inference_mode():
        frames = generate(pipe, demo)
    if len(frames) != demo["model_frames"]:
        raise RuntimeError(
            f"Decoded {len(frames)} frames, expected {demo['model_frames']}"
        )
    save_video(frames, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
