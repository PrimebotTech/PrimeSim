import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from primesim.loss import LOSS
from primesim.pipeline import PrimeSimPipeline


@LOSS.register_module()
class FlowMatchSFTLoss(nn.Module):
    """Flow-matching loss for the TI2V world model: noise every latent frame to a
    random timestep, keep frame 0 the clean first frame, and regress the velocity
    (noise - x0) on frames 1.. ."""

    def forward(self, pipe: PrimeSimPipeline, **inputs):
        clean = inputs["input_latents"]
        batch_size = clean.shape[0]
        timestep_id = torch.randint(0, len(pipe.scheduler.timesteps), (batch_size,))
        timesteps = pipe.scheduler.timesteps[timestep_id].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )

        noise = torch.randn_like(clean)
        latents = pipe.scheduler.add_noise(clean, noise, timesteps)
        latents[:, :, :, 0:1] = inputs["first_frame_latents"]
        target = pipe.scheduler.training_target(clean, noise, timesteps)

        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        pred = pipe.model_fn(
            **models, **{**inputs, "latents": latents}, timestep=timesteps
        )
        loss = F.mse_loss(
            pred[:, :, :, 1:].float(), target[:, :, :, 1:].float(), reduction="none"
        )
        return reduce(loss, "b ... -> b", "mean").mean()
