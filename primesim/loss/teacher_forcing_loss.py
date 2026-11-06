import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from primesim.loss import LOSS
from primesim.modules import CAUSAL_CHUNK_SIZE
from primesim.pipeline import PrimeSimPipeline


@LOSS.register_module()
class TeacherForcingLoss(nn.Module):
    """Stage 1 loss for the causal world model: diffusion forcing with teacher
    forcing. Every CAUSAL_CHUNK_SIZE-frame chunk is noised to its own random
    timestep, frame 0 stays the clean first frame, and the model also gets the clean
    video so each chunk is predicted from the clean frames of earlier chunks. The
    velocity (noise - x0) is regressed on frames 1.., weighted per frame by
    scheduler.training_weight."""

    def forward(self, pipe: PrimeSimPipeline, **inputs):
        if not pipe.dit.causal:
            raise ValueError("TeacherForcingLoss needs the causal model.")
        clean = inputs["input_latents"]
        batch_size, latent_frames = clean.shape[0], clean.shape[3]
        if latent_frames % CAUSAL_CHUNK_SIZE != 0:
            raise ValueError(
                f"{latent_frames} latent frames are not whole "
                f"{CAUSAL_CHUNK_SIZE}-frame chunks."
            )
        num_chunks = latent_frames // CAUSAL_CHUNK_SIZE
        chunk_timestep_id = torch.randint(
            0,
            len(pipe.scheduler.timesteps),
            (batch_size, num_chunks),
            device=pipe.scheduler.timesteps.device,
        )
        timestep_id = chunk_timestep_id.repeat_interleave(CAUSAL_CHUNK_SIZE, dim=1)
        timesteps = pipe.scheduler.timesteps[timestep_id].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        timesteps[:, 0] = 0  # the clean first frame

        noise = torch.randn_like(clean)
        latents = pipe.scheduler.add_noise(clean, noise, timesteps)
        latents[:, :, :, 0:1] = inputs["first_frame_latents"]
        target = pipe.scheduler.training_target(clean, noise, timesteps)

        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        pred = pipe.model_fn(
            **models,
            **{**inputs, "latents": latents, "clean_latents": clean},
            timestep=timesteps,
        )
        loss = F.mse_loss(
            pred[:, :, :, 1:].float(), target[:, :, :, 1:].float(), reduction="none"
        )
        weights = pipe.scheduler.training_weight(timesteps[:, 1:]).to(loss.device)
        loss = loss * weights[:, None, None, :, None, None]  # [B, V, C, T-1, H, W]
        return reduce(loss, "b ... -> b", "mean").mean()
