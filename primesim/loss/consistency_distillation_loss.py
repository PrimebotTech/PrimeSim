import torch
import torch.nn as nn
import torch.nn.functional as F

from primesim.diffusion import FlowMatchScheduler
from primesim.loss import LOSS
from primesim.pipeline import PrimeSimPipeline


@LOSS.register_module()
class ConsistencyDistillationLoss(nn.Module):
    """Stage 2 loss: causal consistency distillation into a few-step student.

    On a num_steps-step schedule, noise the clean video to a random step t (frame 0
    stays the clean first frame). The frozen teacher takes one Euler step t -> t+1
    with classifier-free guidance; the EMA target predicts the clean video from
    there, and the student, from t, is trained to predict the same. All three are
    teacher-forced on the clean video, like stage 1.
    """

    def __init__(self, num_steps: int = 50, cfg_scale: float = 3.0, sigma_shift=5.0):
        super().__init__()
        self.cfg_scale = float(cfg_scale)
        self.scheduler = FlowMatchScheduler()
        self.scheduler.set_timesteps(num_steps, shift=sigma_shift)

    def predict_clean(self, latents, flow, timesteps):
        sigma = self.scheduler._lookup_sigma(timesteps, device=latents.device)
        sigma = self.scheduler._expand_sigma_for_sample(sigma, latents)
        return latents - sigma.to(dtype=latents.dtype) * flow

    def forward(
        self,
        pipe: PrimeSimPipeline,
        teacher_dit: nn.Module,
        ema_dit: nn.Module,
        negative_context: torch.Tensor,
        **inputs,
    ):
        clean = inputs["input_latents"].to(device=pipe.device, dtype=pipe.torch_dtype)
        first_frame = inputs["first_frame_latents"]
        batch_size, latent_frames = clean.shape[0], clean.shape[3]

        # A pair of adjacent steps (t, t_next), one per sample, frame 0 at t=0.
        schedule = self.scheduler.timesteps.to(device=clean.device)
        index = torch.randint(0, len(schedule) - 1, (batch_size,), device=clean.device)
        t, t_next = (
            schedule[i][:, None].expand(batch_size, latent_frames).clone()
            for i in (index, index + 1)
        )
        t[:, 0] = t_next[:, 0] = 0
        t, t_next = t.to(pipe.torch_dtype), t_next.to(pipe.torch_dtype)

        latents_t = self.scheduler.add_noise(clean, torch.randn_like(clean), t)
        latents_t[:, :, :, 0:1] = first_frame

        def call(dit, latents, timesteps, context=None, checkpointing=False):
            model_inputs = {**inputs, "latents": latents, "clean_latents": clean}
            if context is not None:
                model_inputs["context"] = context
            model_inputs["use_gradient_checkpointing"] = checkpointing
            return pipe.model_fn(dit=dit, **model_inputs, timestep=timesteps)

        with torch.no_grad():
            # Teacher: one guided Euler step t -> t_next.
            v_cond = call(teacher_dit, latents_t, t)
            v_uncond = call(teacher_dit, latents_t, t, negative_context)
            v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            dt = (t - t_next).to(latents_t.dtype) / self.scheduler.num_train_timesteps
            dt = self.scheduler._expand_sigma_for_sample(dt, latents_t)
            latents_next = latents_t - dt * v
            latents_next[:, :, :, 0:1] = first_frame
            del v_cond, v_uncond, v
            # EMA target: the clean video predicted from t_next.
            target = self.predict_clean(
                latents_next, call(ema_dit, latents_next, t_next), t_next
            )
        # Student: the clean video predicted from t; the only pass with a graph.
        flow = call(
            pipe.dit, latents_t, t, checkpointing=inputs["use_gradient_checkpointing"]
        )
        pred = self.predict_clean(latents_t, flow, t)
        return F.mse_loss(pred[:, :, :, 1:].float(), target[:, :, :, 1:].float())
