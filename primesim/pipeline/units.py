"""The pipeline's preprocessing units. Like the data transforms, each one is a
callable that takes the inputs dict, adds or updates entries, and returns it;
PrimeSimPipeline chains them with mmengine's Compose. Each unit holds the pipeline
for its models (VAE, text encoder, tokenizer) and helpers."""

from typing import TYPE_CHECKING

from einops import rearrange

from primesim.modules import NUM_VIEWS
from primesim.modules.primesim_dit import _dbg

if TYPE_CHECKING:
    from .primesim_pipeline import PrimeSimPipeline


class CheckShape:
    """Round height / width / num_frames up to sizes the VAE and DiT can tile."""

    def __init__(self, pipe: "PrimeSimPipeline"):
        self.pipe = pipe

    def __call__(self, data):
        data["height"], data["width"], data["num_frames"] = (
            self.pipe.check_resize_height_width(
                data["height"], data["width"], data["num_frames"]
            )
        )
        return data


class InitNoise:
    """Gaussian noise of the latent shape, [B, V, C, T_latent, H/16, W/16]."""

    def __init__(self, pipe: "PrimeSimPipeline"):
        self.pipe = pipe

    def __call__(self, data):
        pipe = self.pipe
        length = (data["num_frames"] - 1) // 4 + 1
        shape = (
            data["batch_size"],
            NUM_VIEWS,
            pipe.vae.model.z_dim,
            length,
            data["height"] // pipe.vae.upsampling_factor,
            data["width"] // pipe.vae.upsampling_factor,
        )
        data["noise"] = pipe.generate_noise(
            shape, seed=data.get("seed"), rand_device=data["rand_device"]
        )
        return data


class EncodePrompt:
    """T5-encode the prompt into the DiT's text context, zeroed past each prompt."""

    def __init__(self, pipe: "PrimeSimPipeline"):
        self.pipe = pipe

    def __call__(self, data):
        data["context"] = self.pipe.encode_prompt(data["prompt"])
        return data


class EncodeVideo:
    """Start the latents from the noise and, in training, VAE-encode the target
    video into input_latents."""

    def __init__(self, pipe: "PrimeSimPipeline"):
        self.pipe = pipe

    def __call__(self, data):
        data["latents"] = data["noise"]
        input_videos = data.get("input_videos")
        if input_videos is None:
            return data
        _dbg("\n=== WAN SHAPE DEBUG [VAE Input] ===")
        _dbg(
            f"  [InputVideoEmbedder] input_video (after preprocess): "
            f"List[{tuple(input_videos[0].shape)}] * {len(input_videos)}  # [B, C, T, "
            f"H, W] raw video before VAE encode"
        )
        B, V, C, T, H, W = input_videos.shape
        input_videos = rearrange(input_videos, "b v c t h w -> (b v) c t h w")
        input_latents = self.pipe.vae.encode(input_videos, device=self.pipe.device)
        input_latents = rearrange(input_latents, "(b v) c t h w -> b v c t h w", b=B)
        _dbg(
            f"  [InputVideoEmbedder] input_latents (after VAE): "
            f"{tuple(input_latents.shape)}  # [B, C_vae, T_latent, H_vae, W_vae]"
        )
        _dbg(f"  [InputVideoEmbedder] noise: {tuple(data['noise'].shape)}")
        data["input_latents"] = input_latents
        return data


class EncodeFirstFrame:
    """VAE-encode each view's first frame into latent frame 0 (Wan2.2-TI2V)."""

    def __init__(self, pipe: "PrimeSimPipeline"):
        self.pipe = pipe

    def __call__(self, data):
        pipe = self.pipe
        input_images = data.get("input_images")
        if input_images is None:
            raise ValueError(
                "the world model is first-frame-to-video: input_images, the first "
                "frame of each view [B, V, C, H, W], is required."
            )
        input_images = input_images.unsqueeze(dim=3)
        B, V, C, T, H, W = input_images.shape
        input_images = rearrange(input_images, "b v c t h w -> (b v) c t h w").to(
            dtype=next(pipe.vae.parameters()).dtype, device=pipe.device
        )
        z = pipe.vae.encode(input_images, device=pipe.device)
        z = rearrange(z, "(b v) c t h w -> b v c t h w", b=B)
        data["latents"][:, :, :, 0:1] = z
        data["first_frame_latents"] = z
        return data
