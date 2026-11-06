"""Stage 3: asymmetric distribution matching distillation (DMD) of the few-step
causal generator."""

from collections import OrderedDict

import torch
from accelerate import Accelerator
from mmengine.registry import MODELS

from primesim.diffusion import FlowMatchScheduler
from primesim.pipeline import PrimeSimPipeline, load_dit

from .primesim_distill_module import merged_state_dict
from .training_module import BaseTrainModule

# minWM's negative prompt for the real score's guidance.
DMD_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@MODELS.register_module()
class PrimeSimDMDModule(BaseTrainModule):
    """Stage 3 training module.

    The generator is the stage 2 few-step causal model with LoRA. The real score is
    the bidirectional pretrained model, frozen; the fake score (critic) starts from
    the same weights, with LoRA. Every optimizer step trains the critic on a fresh
    generator rollout; every update_ratio-th step first trains the generator, through
    its self-forcing rollout, on the scores' DMD gradient. Training never uses the
    data's video, only its first frame, prompt and actions. Checkpoints are the
    generator with LoRA merged, i.e. plain causal PrimeSimModel weights.
    """

    def __init__(
        self,
        action_dim,
        model,
        score_checkpoint_path,
        lora_rank=32,
        lora_target_modules="",
        update_ratio=5,
        real_guidance_scale=3.0,
        negative_prompt=DMD_NEGATIVE_PROMPT,
        consistency_steps=(1000, 750, 500, 250),
        score_min_step=20,
        score_max_step=980,
        local_attn_size=9,
        device="cpu",
        loss=None,
        accelerator=Accelerator,
        **kwargs,
    ):
        super().__init__()
        if device == "gpu":
            device = accelerator.device
        dtype = torch.bfloat16
        self.pipe = PrimeSimPipeline.from_pretrained(
            **model,
            action_dim=action_dim,
            causal=True,
            torch_dtype=dtype,
            device=device,
        )
        self.pipe.requires_grad_(False)
        # The scores' 1000-step schedule (sampling resets the pipeline's).
        self.sigma_shift = 5.0
        self.scheduler = FlowMatchScheduler()
        self.scheduler.set_timesteps(1000, shift=self.sigma_shift)
        self.real_score_dit = load_dit(
            score_checkpoint_path, action_dim, False, dtype, device
        )
        self.fake_score_dit = load_dit(
            score_checkpoint_path, action_dim, False, dtype, device
        )
        self.real_score_dit.requires_grad_(False)
        self.fake_score_dit.requires_grad_(False)
        # LoRA on the critic, then on the generator: their only trainable weights.
        self.fake_score_dit = self.add_lora_to_model(
            self.fake_score_dit,
            target_modules=self.parse_lora_target_modules(
                self.fake_score_dit, lora_target_modules
            ),
            lora_rank=lora_rank,
            upcast_dtype=dtype,
        )
        self.pipe.dit = self.add_lora_to_model(
            self.pipe.dit,
            target_modules=self.parse_lora_target_modules(
                self.pipe.dit, lora_target_modules
            ),
            lora_rank=lora_rank,
            upcast_dtype=dtype,
        )
        for name, dit in (
            ("generator", self.pipe.dit),
            ("critic", self.fake_score_dit),
        ):
            n = sum(p.numel() for p in dit.parameters() if p.requires_grad)
            print(f"[stage 3] {name} LoRA rank {lora_rank}: {n / 1e6:.1f}M trainable")
        self.pipe.consistency_steps = tuple(consistency_steps)
        self.loss = loss
        self.update_ratio = update_ratio
        self.real_guidance_scale = real_guidance_scale
        self.negative_prompt = negative_prompt
        self.score_min_step = score_min_step
        self.score_max_step = score_max_step
        self.local_attn_size = local_attn_size
        self.optimizer_steps = 0
        self.train()

    def train(self, mode: bool = True):
        # Every network stays in eval mode; only their LoRA weights train.
        super().train(mode)
        for module in (self.pipe, self.real_score_dit, self.fake_score_dit):
            module.eval()
        return self

    def training_phases(self):
        """The critic every step; the generator first every update_ratio-th."""
        if self.optimizer_steps % self.update_ratio == 0:
            return ("generator", "critic")
        return ("critic",)

    def on_after_optimizer_step(self):
        self.optimizer_steps += 1

    def state_dict_for_save(self, accelerator):
        if not accelerator.is_main_process:
            return OrderedDict()
        return merged_state_dict(self.pipe.dit)

    def score_timesteps(self, generated):
        """One shifted timestep per sample in [score_min_step, score_max_step], for
        every frame but frame 0 (the clean first frame)."""
        batch_size, latent_frames = generated.shape[0], generated.shape[3]
        raw = torch.randint(0, 1000, (batch_size,), device=generated.device)
        sigma = raw.to(generated.dtype) / 1000.0
        shift = self.sigma_shift
        t = (shift * sigma / (1.0 + (shift - 1.0) * sigma) * 1000.0).clamp(
            self.score_min_step, self.score_max_step
        )
        t = t[:, None].expand(batch_size, latent_frames).clone()
        t[:, 0] = 0
        return t.to(self.pipe.torch_dtype)

    def score(self, dit, noisy, t, context, inputs, checkpointing=False):
        return self.pipe.model_fn(
            dit=dit,
            latents=noisy,
            timestep=t,
            context=context,
            action=inputs["action"],
            use_gradient_checkpointing=checkpointing,
        )

    def predict_clean(self, noisy, flow, t):
        scheduler = self.scheduler
        sigma = scheduler._expand_sigma_for_sample(
            scheduler._lookup_sigma(t, device=noisy.device), noisy
        )
        return noisy - sigma.to(noisy.dtype) * flow

    def forward(self, inputs, phase="critic"):
        if "context" not in inputs:  # the first phase of a step prepares the inputs
            inputs["input_videos"] = None  # DMD never uses the data's video
            self.pipe.units(inputs)
            inputs["negative_context"] = self.pipe.encode_prompt(
                [self.negative_prompt] * inputs["batch_size"]
            )
        generator_phase = phase == "generator"
        generated = self.pipe.self_forcing_rollout(
            self.pipe.dit,
            torch.randn_like(inputs["latents"]),
            inputs["context"],
            inputs["action"],
            inputs["first_frame_latents"],
            self.pipe.consistency_steps,
            with_grad=generator_phase,
            local_attn_size=self.local_attn_size,
        )
        t = self.score_timesteps(generated)
        noise = torch.randn_like(generated)
        context, negative = inputs["context"], inputs["negative_context"]
        if generator_phase:
            with torch.no_grad():
                noisy = self.scheduler.add_noise(generated.detach(), noise, t)
                fake = self.score(self.fake_score_dit, noisy, t, context, inputs)
                fake_x0 = self.predict_clean(noisy, fake, t)
                cond = self.score(self.real_score_dit, noisy, t, context, inputs)
                uncond = self.score(self.real_score_dit, noisy, t, negative, inputs)
                real_flow = cond + self.real_guidance_scale * (cond - uncond)
                real_x0 = self.predict_clean(noisy, real_flow, t)
            return self.loss.generator_loss(generated, fake_x0, real_x0)
        generated = generated.detach()
        noisy = self.scheduler.add_noise(generated, noise, t)
        flow = self.score(
            self.fake_score_dit, noisy, t, context, inputs, checkpointing=True
        )
        return self.loss.critic_loss(flow, noise, generated)
