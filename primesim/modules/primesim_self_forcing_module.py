"""Stage 4: cross-window Self-Forcing++ on episode-aligned pretraining data."""

import torch
import torch.distributed as dist
from accelerate import Accelerator
from mmengine.registry import MODELS

from primesim.data.pipeline.transforms import ToRelativeAction

from .primesim_dmd_module import PrimeSimDMDModule


def _uniform_batch_int(data, key):
    values = torch.as_tensor(data[key]).detach().cpu().reshape(-1)
    if not values.numel() or not bool((values == values[0]).all()):
        raise ValueError(f"Every stage-4 batch row must share {key!r}.")
    return int(values[0].item())


def _shared_random_index(upper_bound, device):
    if upper_bound <= 0:
        raise ValueError("A stage-4 chain must contain at least one segment.")
    if not (dist.is_available() and dist.is_initialized()):
        return int(torch.randint(upper_bound, (1,)).item())
    bounds = torch.tensor([upper_bound, upper_bound], device=device, dtype=torch.long)
    dist.all_reduce(bounds[:1], op=dist.ReduceOp.MIN)
    dist.all_reduce(bounds[1:], op=dist.ReduceOp.MAX)
    if bounds[0] != bounds[1]:
        raise RuntimeError("Stage-4 chain lengths differ across distributed ranks.")
    value = (
        torch.randint(upper_bound, (1,), device=device)
        if dist.get_rank() == 0
        else torch.zeros(1, dtype=torch.long, device=device)
    )
    dist.broadcast(value, src=0)
    return int(value.item())


@MODELS.register_module()
class PrimeSimSelfForcingModule(PrimeSimDMDModule):
    """Self-Forcing++ with persistent KV history over a sampled episode subchain.

    One generator-DMD window is sampled per chain. Its gradient is retained until
    the final segment, while the fake score (critic) is updated on every segment.
    The generator weights therefore stay fixed for the lifetime of their KV cache.
    """

    stage4_chain_training = True

    def parse_lora_target_modules(self, model, lora_target_modules):
        targets = super().parse_lora_target_modules(model, lora_target_modules)
        for root_name in ("action_mlp1", "frame_action_adapter"):
            root = getattr(model, root_name, None)
            if root is None:
                continue
            for child_name, child in root.named_modules():
                if isinstance(child, torch.nn.Linear):
                    targets.append(
                        root_name if not child_name else f"{root_name}.{child_name}"
                    )
        return list(dict.fromkeys(targets))

    def __init__(
        self,
        opening_num_frames=33,
        continuation_num_frames=37,
        score_window_mode="random",
        generator_learning_rate=2.0e-6,
        critic_learning_rate=4.0e-7,
        generator_betas=(0.0, 0.999),
        critic_betas=(0.0, 0.999),
        sink_size=3,
        last_step_only=True,
        device="cpu",
        accelerator=Accelerator,
        **kwargs,
    ):
        kwargs.pop("update_ratio", None)
        super().__init__(
            device=device, accelerator=accelerator, update_ratio=1, **kwargs
        )
        self.opening_num_frames = int(opening_num_frames)
        self.continuation_num_frames = int(continuation_num_frames)
        if (self.opening_num_frames - 1) % 4 or (self.continuation_num_frames - 1) % 4:
            raise ValueError("Stage-4 RGB windows must be 1 plus a multiple of four.")
        self.score_window_mode = str(score_window_mode).lower()
        if self.score_window_mode not in {"random", "head", "tail"}:
            raise ValueError("score_window_mode must be random, head, or tail.")
        self.generator_learning_rate = float(generator_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.generator_betas = tuple(float(value) for value in generator_betas)
        self.critic_betas = tuple(float(value) for value in critic_betas)
        self.sink_size = int(sink_size)
        self.last_step_only = bool(last_step_only)
        self._relative_action = ToRelativeAction(key_mapper=[])
        self._chain_action_anchor = None
        self._chain_actions_are_relative = None
        self._kv_cache = None
        self._sequence_start_frame = 0
        self._chain_episode_ids = None
        self._selected_segment = None
        self._window_opening = True
        self._window_end = False
        self._phase_inputs = None
        self._phase_generated = None
        self._phase_score_inputs = None

    @staticmethod
    def _split_params(module):
        decay, no_decay = [], []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            target = (
                no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay
            )
            target.append(parameter)
        return decay, no_decay

    def optimizer_param_groups(self, weight_decay):
        groups, stats = [], []
        for phase, module, learning_rate, betas in (
            (
                "generator",
                self.pipe.dit,
                self.generator_learning_rate,
                self.generator_betas,
            ),
            (
                "critic",
                self.fake_score_dit,
                self.critic_learning_rate,
                self.critic_betas,
            ),
        ):
            decay, no_decay = self._split_params(module)
            for label, params, group_decay in (
                ("decay", decay, float(weight_decay)),
                ("no_decay", no_decay, 0.0),
            ):
                if not params:
                    continue
                groups.append(
                    {
                        "params": params,
                        "phase": phase,
                        "lr": learning_rate,
                        "betas": betas,
                        "weight_decay": group_decay,
                    }
                )
                stats.append(
                    (
                        f"{phase}/{label}",
                        group_decay,
                        len(params),
                        sum(parameter.numel() for parameter in params),
                    )
                )
        return groups, stats

    def phase_parameters(self, phase):
        module = self.pipe.dit if phase == "generator" else self.fake_score_dit
        return [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]

    def reset_rollout_history(self):
        self._kv_cache = None
        self._sequence_start_frame = 0
        self._chain_action_anchor = None
        self._chain_actions_are_relative = None

    def training_phases(self, data=None):
        if data is None:
            raise ValueError("Stage 4 needs episode metadata for every window.")
        opening = bool(torch.as_tensor(data["episode_start"]).any())
        ending = bool(torch.as_tensor(data["episode_end"]).any())
        episode_ids = torch.as_tensor(data["amortized_episode_id"]).cpu().reshape(-1)
        segment = _uniform_batch_int(data, "amortized_segment_index")
        num_segments = _uniform_batch_int(data, "amortized_episode_num_segments")
        if opening:
            if segment != 0:
                raise ValueError("An opening stage-4 window must have segment index 0.")
            self.reset_rollout_history()
            self._selected_segment = _shared_random_index(
                num_segments, self.pipe.device
            )
        else:
            if self._chain_episode_ids is None or not torch.equal(
                episode_ids, self._chain_episode_ids
            ):
                raise RuntimeError("Stage-4 sampler did not preserve episode order.")
        self._chain_episode_ids = episode_ids.clone()
        self._window_opening = opening
        self._window_end = ending
        self._phase_inputs = None
        self._phase_generated = None
        self._phase_score_inputs = None
        if segment == self._selected_segment:
            return "generator", "critic"
        return ("critic",)

    def chain_boundary(self):
        return self._window_end

    def _prepare_inputs(self, inputs):
        opening = self._window_opening
        videos = inputs["input_videos"]
        state = inputs.get("state")
        action_mask = inputs.get("action_mask")
        if state is None or action_mask is None:
            raise ValueError("Stage 4 needs state and action_mask from the dataset.")
        actions_are_relative = bool(_uniform_batch_int(inputs, "actions_are_relative"))
        if opening:
            self._chain_action_anchor = state[:, :1].clone()
            self._chain_actions_are_relative = actions_are_relative
        elif self._chain_actions_are_relative != actions_are_relative:
            raise RuntimeError("Stage-4 action semantics changed inside one chain.")

        visual_frames = (
            self.opening_num_frames if opening else self.continuation_num_frames
        )
        action_frames = visual_frames - 1
        if videos.shape[3] < visual_frames or inputs["action"].shape[1] < action_frames:
            raise ValueError("The data query is shorter than the stage-4 window.")

        raw_action = inputs["action"][:, :action_frames]
        raw_mask = action_mask[:, :action_frames]
        if actions_are_relative:
            generator_action = raw_action * raw_mask.to(raw_action.dtype)
        else:
            generator_action = self._relative_action.relative_to_reference(
                raw_action, self._chain_action_anchor, raw_mask
            )

        score_offset = 0
        if not opening:
            if self.score_window_mode == "tail":
                score_offset = 4
            elif self.score_window_mode == "random":
                score_offset = 4 * int(torch.randint(0, 2, (1,)).item())
        score_raw = inputs["action"][:, score_offset : score_offset + 32]
        score_mask = action_mask[:, score_offset : score_offset + 32]
        if actions_are_relative and score_offset:
            score_action = self._relative_action.relative_to_reference(
                score_raw,
                inputs["action"][:, score_offset - 1 : score_offset],
                score_mask,
            )
        elif actions_are_relative:
            score_action = score_raw * score_mask.to(score_raw.dtype)
        else:
            score_action = self._relative_action.relative_to_reference(
                score_raw,
                state[:, score_offset : score_offset + 1],
                score_mask,
            )

        prepared = dict(inputs)
        prepared["input_videos"] = None
        # Continuation score windows may use the head (GT frame 0 + g0..g7) or
        # tail (GT frame 4 + g1..g8). The generator drops this encoded boundary
        # latent and continues exclusively from its persistent KV history.
        prepared["input_images"] = videos[:, :, :, score_offset]
        prepared["num_frames"] = visual_frames
        prepared["action"] = generator_action.to(
            device=self.pipe.device, dtype=self.pipe.torch_dtype
        )
        self.pipe.units(prepared)
        prepared["score_action"] = score_action.to(
            device=self.pipe.device, dtype=self.pipe.torch_dtype
        )
        prepared["score_condition"] = prepared["first_frame_latents"]
        prepared["score_offset"] = score_offset // 4
        if self.real_guidance_scale:
            prepared["negative_context"] = self.pipe.encode_prompt(
                [self.negative_prompt] * prepared["batch_size"]
            )
        return prepared

    def _rollout(self, inputs, with_grad):
        opening = self._window_opening
        noise = inputs["latents"] if opening else inputs["latents"][:, :, :, 1:]
        first_frame = inputs["first_frame_latents"] if opening else None
        generated, self._kv_cache, exit_timestep = self.pipe.self_forcing_segment(
            self.pipe.dit,
            noise,
            inputs["context"],
            inputs["action"],
            first_frame,
            self.pipe.consistency_steps,
            with_grad=with_grad,
            kv_cache=self._kv_cache,
            sequence_start_frame=self._sequence_start_frame,
            action_target_offset=1 if opening else 0,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            last_step_only=self.last_step_only,
        )
        self._sequence_start_frame += int(noise.shape[3])
        if opening:
            score_sample = generated
        else:
            offset = int(inputs["score_offset"])
            score_sample = torch.cat(
                [
                    inputs["score_condition"].detach(),
                    generated[:, :, :, offset : offset + 8],
                ],
                dim=3,
            )
        score_inputs = dict(inputs)
        score_inputs["action"] = inputs["score_action"]
        score_inputs["first_frame_latents"] = inputs["score_condition"]
        score_inputs["rollout_exit_timestep"] = exit_timestep
        return score_sample, score_inputs

    def _generator_loss(self, generated, inputs):
        t = self.score_timesteps(generated)
        noise = torch.randn_like(generated)
        with torch.no_grad():
            noisy = self.scheduler.add_noise(generated.detach(), noise, t)
            fake = self.score(self.fake_score_dit, noisy, t, inputs["context"], inputs)
            fake_x0 = self.predict_clean(noisy, fake, t)
            real_cond = self.score(
                self.real_score_dit, noisy, t, inputs["context"], inputs
            )
            if self.real_guidance_scale:
                real_uncond = self.score(
                    self.real_score_dit,
                    noisy,
                    t,
                    inputs["negative_context"],
                    inputs,
                )
                real_cond = real_cond + self.real_guidance_scale * (
                    real_cond - real_uncond
                )
            real_x0 = self.predict_clean(noisy, real_cond, t)
        return self.loss.generator_loss(generated, fake_x0, real_x0)

    def _critic_loss(self, generated, inputs):
        generated = generated.detach()
        t = self.score_timesteps(generated)
        noise = torch.randn_like(generated)
        noisy = self.scheduler.add_noise(generated, noise, t)
        flow = self.score(
            self.fake_score_dit,
            noisy,
            t,
            inputs["context"],
            inputs,
            checkpointing=True,
        )
        return self.loss.critic_loss(flow, noise, generated)

    def forward(self, inputs, phase="critic"):
        if phase not in {"generator", "critic"}:
            raise ValueError("Stage 4 phase must be generator or critic.")
        if phase == "critic" and self._phase_inputs is not None:
            prepared = self._phase_inputs
        else:
            prepared = self._prepare_inputs(inputs)
            if phase == "generator":
                self._phase_inputs = prepared

        if phase == "generator":
            generated, score_inputs = self._rollout(prepared, with_grad=True)
            self._phase_generated = generated.detach()
            self._phase_score_inputs = score_inputs
            return self._generator_loss(generated, score_inputs)

        if self._phase_generated is None:
            generated, score_inputs = self._rollout(prepared, with_grad=False)
        else:
            generated, score_inputs = self._phase_generated, self._phase_score_inputs
        loss = self._critic_loss(generated, score_inputs)
        self._phase_inputs = None
        self._phase_generated = None
        self._phase_score_inputs = None
        return loss

    def on_after_optimizer_step(self):
        super().on_after_optimizer_step()
        self.reset_rollout_history()
        self._chain_episode_ids = None
        self._selected_segment = None
