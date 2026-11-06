import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from einops import rearrange
from PIL import Image

from primesim.modules import NUM_VIEWS
from primesim.utils.image_metrics import eval_images

if TYPE_CHECKING:  # runner imports this module, so only for the annotation
    from primesim.runner.logger import ModelLogger


def get_pipeline_inputs(shared_inputs):
    batch_size = shared_inputs["input_images"].shape[0]
    random_idx = random.randint(0, batch_size - 1)
    inputs = {
        "prompt": [shared_inputs["prompt"][random_idx]],
        "first_frame": shared_inputs["input_images"][random_idx][None],
        "action": shared_inputs["action"][random_idx][None]
        if "action" in shared_inputs
        else None,
        "height": shared_inputs["height"],
        "width": shared_inputs["width"],
        "num_frames": shared_inputs["num_frames"],
    }
    ground_truth = shared_inputs["input_videos"][random_idx][None]
    ground_truth = rearrange(ground_truth, "b v c t h w -> b c t (v h) w")
    return inputs, ground_truth


def generate_video_from_pipeline(
    pipe,
    first_frame: Image.Image,
    prompt: str,
    action: Optional[torch.Tensor] = None,
    height: int = 480,
    width: int = 640,
    num_frames: int = 9,
    num_inference_steps: int = 50,
    seed: int = 42,
) -> List[Image.Image]:
    """
    Generate video using existing training pipeline.

    Args:
        pipe: WanVideoPipeline (already loaded in training)
        first_frame: First frame PIL image
        action: Action tensor [horizon, action_dim]
        prompt: Text prompt (optional for I2V)
        height, width: Output video dimensions
        num_frames: Number of frames to generate
        num_inference_steps: Diffusion steps
        seed: Random seed

    Returns:
        List of PIL images (generated frames)
    """
    kwargs = dict(
        prompt=prompt,
        input_images=first_frame,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        num_inference_steps=num_inference_steps,
        output_type="floatpoint",
    )

    if action is not None:
        # action may already be [B, T, D] (from get_pipeline_inputs) or [T, D] (raw)
        act = action if action.dim() == 3 else action.unsqueeze(0)
        kwargs["action"] = act.to(dtype=pipe.torch_dtype, device=pipe.device)

    video = pipe(**kwargs)
    return video


def concat_frames_side_by_side(
    gt_frames: List[Image.Image],
    gen_frames: List[Image.Image],
    height: int,
    width: int,
) -> List[Image.Image]:
    """Concatenate GT and generated frames horizontally."""
    result = []
    n = min(len(gt_frames), len(gen_frames))
    for i in range(n):
        gt = (
            gt_frames[i].resize((width, NUM_VIEWS * height), Image.LANCZOS)
            if gt_frames[i].size != (width, NUM_VIEWS * height)
            else gt_frames[i]
        )
        gen = (
            gen_frames[i].resize((width, NUM_VIEWS * height), Image.LANCZOS)
            if gen_frames[i].size != (width, NUM_VIEWS * height)
            else gen_frames[i]
        )
        merged = Image.new("RGB", (width * 2, NUM_VIEWS * height))
        merged.paste(gt, (0, 0))
        merged.paste(gen, (width, 0))
        result.append(merged)

    return result


def pil_frames_to_wandb_video(
    frames: List[Image.Image],
    fps: int = 5,
    caption: str = "gen_frames_vs_gt_frames",
) -> np.ndarray:
    frames_array = np.stack([np.array(frame) for frame in frames], axis=0)
    frames_array = frames_array.transpose(0, 3, 1, 2)
    video = wandb.Video(
        frames_array,
        fps=fps,
        format="mp4",
        caption=caption,
    )

    return video


def generate_and_save_comparison_video(
    pipe,
    data: Dict[str, Any],
    model_logger: "ModelLogger",
    num_inference_steps: int = 50,
    seed: int = 42,
    fps: int = 5,
    wandb_active: bool = False,
    step: Optional[int] = None,
    metric_prefix: str = "train/img_recon_",
    caption: str = "gen_frames_vs_gt_frames",
) -> str:
    """
    Generate video from a dataset sample and save side-by-side comparison.

    Args:
        pipe: WanVideoPipeline (from training)
        sample: Dataset sample dict (must contain "video", optionally "action",
            "prompt")
        height, width: Video dimensions
        num_frames: Number of frames
        num_inference_steps: Diffusion steps
        seed: Random seed
        fps: Output video FPS
        metric_prefix: wandb key prefix for the image metrics, so a train-batch
            comparison and a val-pass comparison don't collide in wandb.
        caption: wandb caption for the exported video (e.g.
            "train/gen_frames_vs_gt_frames").

    Returns:
        dict with keys "metrics" (wandb-ready metric dict, prefixed by
        ``metric_prefix``)
        and "video" (a ``wandb.Video`` of the side-by-side comparison).
    """
    if step is None:
        step = model_logger.num_steps if model_logger is not None else 0

    inputs, gt_frames = get_pipeline_inputs(data)
    gt_frames = pipe.vae_output_to_video(gt_frames)
    # Generate video
    gen_frames = generate_video_from_pipeline(
        pipe=pipe,
        **inputs,
        num_inference_steps=num_inference_steps,
    )
    gen_video = pipe.vae_output_to_video(gen_frames)
    metrics = eval_images(gen_video, gt_frames, prefix=metric_prefix)

    # Concatenate side-by-side
    merged = concat_frames_side_by_side(
        gt_frames, gen_video, inputs["height"], inputs["width"]
    )

    # Log to wandb
    wandb_video = pil_frames_to_wandb_video(merged, fps=fps, caption=caption)
    if wandb_active and wandb.run is not None:
        wandb.log(
            {
                caption: wandb_video,
                **metrics,
            },
            step=step,
        )

    return {"metrics": metrics, "video": wandb_video}
