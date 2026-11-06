from typing import Optional, Union

import torch
from einops import rearrange, reduce
from mmengine.dataset import Compose
from PIL import Image
from tqdm import tqdm
from typing_extensions import Literal

from primesim.diffusion import FlowMatchScheduler
from primesim.modules import (
    CAUSAL_CHUNK_SIZE,
    HuggingfaceTokenizer,
    PrimeSimModel,
    T5TextEncoder,
    WanVideoVAE38,
)
from primesim.utils.initialization import skip_model_initialization
from primesim.utils.loader import load_state_dict
from primesim.utils.lora import GeneralLoRALoader

from .units import CheckShape, EncodeFirstFrame, EncodePrompt, EncodeVideo, InitNoise

# Wan2.2-TI2V-5B: the one base model, and the files it ships as.
WAN22_TI2V_5B_DIT = dict(
    dim=3072,
    in_dim=48,
    ffn_dim=14336,
    out_dim=48,
    text_dim=4096,
    freq_dim=256,
    eps=1e-6,
    patch_size=[1, 2, 2],
    num_heads=24,
    num_layers=30,
)


def _load_model(
    model_class,
    path,
    torch_dtype,
    device,
    config=None,
    key_prefix="",
):
    """Build model_class without random init, load `path` into it strictly, cast,
    eval. `key_prefix` is prepended to every checkpoint key: the VAE file stores the
    inner VideoVAE38_'s keys, which sit under `model.` in WanVideoVAE38.
    """
    with skip_model_initialization():
        model = model_class(**(config or {}))
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device=device)
    if key_prefix:
        state_dict = {key_prefix + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, assign=True)
    return model.to(dtype=torch_dtype, device=device).eval()


def load_dit(
    checkpoint_path, action_dim, causal=False, torch_dtype=torch.bfloat16, device="cuda"
):
    """A PrimeSimModel loaded strictly from a world-model checkpoint (midtrain,
    stage 1, 2 or 3: all share the layout)."""
    config = dict(WAN22_TI2V_5B_DIT, action_dim=action_dim, causal=causal)
    dit = _load_model(
        PrimeSimModel, checkpoint_path, torch_dtype, device, config=config
    )
    print(f"Loaded world-model checkpoint {checkpoint_path} (causal={causal})")
    return dit


class PrimeSimPipeline(torch.nn.Module):
    """Wan2.2-TI2V-5B world model: multiview first-frame-to-video, conditioned on text
    and an action trajectory. One DiT, one text encoder, one VAE."""

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__()
        # device and torch_dtype are where intermediate tensors live, not the models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The VAE downsamples 16x and the DiT patchifies 2x more, so pixel sizes must be
        # multiples of 32 to give whole latent / token grids; the VAE keeps the first
        # frame and compresses time 4x after it.
        self.height_division_factor = 32
        self.width_division_factor = 32
        self.time_division_factor = 4
        self.time_division_remainder = 1
        self.scheduler = FlowMatchScheduler()
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: T5TextEncoder = None
        self.dit: PrimeSimModel = None
        self.vae: WanVideoVAE38 = None
        self.in_iteration_models = ("dit",)
        # Inputs dict -> model inputs, like the data pipeline's transforms.
        self.units = Compose(
            [
                CheckShape(self),
                InitNoise(self),
                EncodePrompt(self),
                EncodeVideo(self),
                EncodeFirstFrame(self),
            ]
        )
        self.model_fn = model_fn_wan_video
        # Set for a consistency-distilled causal model (stage 2): sample with these
        # few steps per chunk and a KV cache instead of the full schedule.
        self.consistency_steps: Optional[tuple[int, ...]] = None

    @staticmethod
    def from_pretrained(
        text_encoder_path: str,
        vae_path: str,
        tokenizer_path: str,
        action_dim: int,
        checkpoint_path: str,
        causal: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
    ):
        """Build the world model (chunk-causal attention if `causal`, stages 1-4).

        text_encoder_path, vae_path: the Wan2.2-TI2V-5B umt5-xxl encoder and VAE
        weights files; tokenizer_path: the google/umt5-xxl tokenizer folder;
        checkpoint_path: the PrimeSim DiT weights (midtrain or stage 1-4), loaded
        strictly.
        """
        pipe = PrimeSimPipeline(device=device, torch_dtype=torch_dtype)
        pipe.dit = load_dit(checkpoint_path, action_dim, causal, torch_dtype, device)
        pipe.text_encoder = _load_model(
            T5TextEncoder, text_encoder_path, torch_dtype, device
        )
        pipe.vae = _load_model(
            WanVideoVAE38, vae_path, torch_dtype, device, key_prefix="model."
        )
        pipe.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean="whitespace"
        )
        return pipe

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1)
                // self.height_division_factor
                * self.height_division_factor
            )
            print(
                f"height % {self.height_division_factor} != 0. We round it up to "
                f"{height}."
            )
        if width % self.width_division_factor != 0:
            width = (
                (width + self.width_division_factor - 1)
                // self.width_division_factor
                * self.width_division_factor
            )
            print(
                f"width % {self.width_division_factor} != 0. We round it up to {width}."
            )
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (
                    (num_frames + self.time_division_factor - 1)
                    // self.time_division_factor
                    * self.time_division_factor
                    + self.time_division_remainder
                )
                print(
                    f"num_frames % {self.time_division_factor} != "
                    f"{self.time_division_remainder}. We round it up to {num_frames}."
                )
            return height, width, num_frames

    def vae_output_to_image(
        self, vae_output, pattern="B C H W", min_value=-1, max_value=1
    ):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(
            0, 255
        )
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

    def vae_output_to_video(
        self, vae_output, pattern="B C T H W", min_value=-1, max_value=1
    ):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [
            self.vae_output_to_image(
                image, pattern="H W C", min_value=min_value, max_value=max_value
            )
            for image in vae_output
        ]
        return video

    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_torch_dtype=torch.float32,
        device=None,
        torch_dtype=None,
    ):
        # Initialize Gaussian noise
        generator = (
            None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        )
        noise = torch.randn(
            shape, generator=generator, device=rand_device, dtype=rand_torch_dtype
        )
        noise = noise.to(
            dtype=torch_dtype or self.torch_dtype, device=device or self.device
        )
        return noise

    def get_module(self, model, name):
        if "." in name:
            name, suffix = name[: name.index(".")], name[name.index(".") + 1 :]
            if name.isdigit():
                return self.get_module(model[int(name)], suffix)
            else:
                return self.get_module(getattr(model, name), suffix)
        else:
            return getattr(model, name)

    def freeze_except(self, model_names):
        self.eval()
        self.requires_grad_(False)
        for name in model_names:
            module = self.get_module(self, name)
            if module is None:
                print(
                    f"No {name} models in the pipeline. "
                    f"We cannot enable training on the model. "
                    f"If this occurs during the data processing stage, it is normal."
                )
                continue
            module.train()
            module.requires_grad_(True)

    def load_lora(
        self, module: torch.nn.Module, lora_path: str = None, alpha=1, state_dict=None
    ):
        """Fuse a LoRA (file or state dict) into `module`'s weights."""
        lora = (
            state_dict
            if state_dict is not None
            else load_state_dict(
                lora_path, torch_dtype=self.torch_dtype, device=self.device
            )
        )
        lora_loader = GeneralLoRALoader(
            torch_dtype=self.torch_dtype, device=self.device
        )
        lora_loader.fuse_lora_to_base_model(
            module, lora_loader.convert_state_dict(lora), alpha=alpha
        )

    def encode_prompt(self, prompt):
        """T5 text context [B, 512, 4096] for a list of prompts, zeroed past each
        prompt's tokens."""
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(self.device), mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def denoise_autoregressive(self, models, inputs, progress_bar_cmd=tqdm):
        """Causal model: denoise one CAUSAL_CHUNK_SIZE-frame chunk at a time, each
        over the full schedule, with the finished chunks fed back as clean (t=0)
        context and the actions cut to the frames generated so far. The prefix is
        recomputed each step rather than cached."""
        latents, first_frame = inputs["latents"], inputs["first_frame_latents"]
        batch_size, latent_frames = latents.shape[0], latents.shape[3]
        if latent_frames % CAUSAL_CHUNK_SIZE != 0:
            raise ValueError(
                f"{latent_frames} latent frames are not whole "
                f"{CAUSAL_CHUNK_SIZE}-frame chunks."
            )
        action = inputs.get("action")
        output = torch.empty_like(latents)
        for start in progress_bar_cmd(range(0, latent_frames, CAUSAL_CHUNK_SIZE)):
            end = start + CAUSAL_CHUNK_SIZE
            chunk = latents[:, :, :, start:end].clone()
            if start == 0:
                chunk[:, :, :, :1] = first_frame
            # Four actions per latent frame, up to the end of this chunk.
            chunk_action = (
                None if action is None else action[:, : min(4 * end, action.shape[1])]
            )
            for progress_id, timestep in enumerate(self.scheduler.timesteps):
                prefix = torch.cat([output[:, :, :, :start], chunk], dim=3)
                frame_timesteps = torch.zeros(
                    batch_size, end, dtype=self.torch_dtype, device=self.device
                )
                frame_timesteps[:, start:end] = timestep
                if start == 0:
                    frame_timesteps[:, 0] = 0  # the clean first frame
                flow = self.model_fn(
                    **models,
                    **{**inputs, "latents": prefix, "action": chunk_action},
                    timestep=frame_timesteps,
                )[:, :, :, start:end]
                chunk = self.scheduler.step(flow, timestep, chunk)
                if start == 0:
                    chunk[:, :, :, :1] = first_frame
            output[:, :, :, start:end] = chunk
        return output

    def denoise_consistency(
        self,
        models,
        inputs,
        consistency_steps,
        progress_bar_cmd=tqdm,
        sigma_shift=5.0,
        local_attn_size=20,
    ):
        """Consistency-distilled causal model (stage 2): one CAUSAL_CHUNK_SIZE-frame
        chunk at a time with a KV cache. Per chunk, each step predicts the clean
        chunk in one call and re-noises it to the next step; a final pass at t=0
        writes the clean chunk into the cache for the chunks after it.

        consistency_steps are 1000-step training-grid positions, e.g. (1000, 750,
        500, 250), warped onto the shifted schedule (~1000, 937.5, 833.3, 625).
        """
        latents, first_frame = inputs["latents"], inputs["first_frame_latents"]
        batch_size, latent_frames = latents.shape[0], latents.shape[3]
        if latent_frames % CAUSAL_CHUNK_SIZE != 0:
            raise ValueError(
                f"{latent_frames} latent frames are not whole "
                f"{CAUSAL_CHUNK_SIZE}-frame chunks."
            )
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(1000, shift=sigma_shift)
        grid = torch.cat([scheduler.timesteps, torch.zeros(1)])
        steps = grid[1000 - torch.as_tensor(consistency_steps, dtype=torch.long)]

        dit = models["dit"]
        kv_cache = dit.new_kv_cache(latents, local_attn_size)
        action = inputs.get("action")
        output = torch.empty_like(latents)

        def model_call(chunk, timestep, start):
            frame_timesteps = torch.full(
                (batch_size, chunk.shape[3]),
                float(timestep),
                dtype=torch.float32,
                device=latents.device,
            )
            if start == 0:
                frame_timesteps[:, 0] = 0  # the clean first frame
            # The actions from frame 0 up to the end of this chunk.
            end = min(start + CAUSAL_CHUNK_SIZE, latent_frames - 1)
            chunk_action = None if action is None else action[:, : 4 * end]
            flow = self.model_fn(
                **models,
                **{**inputs, "latents": chunk, "action": chunk_action},
                timestep=frame_timesteps,
                kv_cache=kv_cache,
                start_frame=start,
            )
            return flow, frame_timesteps

        for start in progress_bar_cmd(range(0, latent_frames, CAUSAL_CHUNK_SIZE)):
            chunk = latents[:, :, :, start : start + CAUSAL_CHUNK_SIZE].clone()
            for step_id, timestep in enumerate(steps):
                if start == 0:
                    chunk[:, :, :, :1] = first_frame
                flow, frame_timesteps = model_call(chunk, timestep, start)
                sigma = scheduler._expand_sigma_for_sample(
                    scheduler._lookup_sigma(frame_timesteps, device=chunk.device), chunk
                )
                chunk = (chunk - sigma * flow).to(chunk.dtype)  # predicted clean chunk
                if start == 0:
                    chunk[:, :, :, :1] = first_frame
                if step_id + 1 < len(steps):
                    next_timesteps = torch.full_like(
                        frame_timesteps, steps[step_id + 1]
                    )
                    if start == 0:
                        next_timesteps[:, 0] = 0
                    chunk = scheduler.add_noise(
                        chunk, torch.randn_like(chunk), next_timesteps
                    )
            output[:, :, :, start : start + CAUSAL_CHUNK_SIZE] = chunk
            model_call(chunk, 0.0, start)  # commit the clean chunk to the cache
        return output

    def self_forcing_rollout(
        self,
        dit,
        noise,
        context,
        action,
        first_frame,
        consistency_steps,
        with_grad,
        sigma_shift=5.0,
        local_attn_size=9,
    ):
        """Stage 3 training rollout of the few-step causal generator: the KV-cache
        consistency sampler, but each chunk stops at one exit step drawn at random
        (the same for every chunk and every rank) and returns that step's clean
        prediction. With with_grad only those exit calls keep a graph; the earlier
        steps, re-noising and the cache commits are detached.
        """
        batch_size, latent_frames = noise.shape[0], noise.shape[3]
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(1000, shift=sigma_shift)
        grid = torch.cat([scheduler.timesteps, torch.zeros(1)])
        steps = grid[1000 - torch.as_tensor(consistency_steps, dtype=torch.long)]
        num_chunks = latent_frames // CAUSAL_CHUNK_SIZE
        exit_ids = torch.randint(0, len(steps), (num_chunks,), device=noise.device)
        exit_ids.fill_(exit_ids[0])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(exit_ids, src=0)
        exit_id = int(exit_ids[0])

        kv_cache = dit.new_kv_cache(noise, local_attn_size)

        def call(chunk, timestep, start):
            frame_timesteps = torch.full(
                (batch_size, chunk.shape[3]),
                float(timestep),
                dtype=torch.float32,
                device=noise.device,
            )
            if start == 0:
                frame_timesteps[:, 0] = 0  # the clean first frame
            end = min(start + CAUSAL_CHUNK_SIZE, latent_frames - 1)
            flow = self.model_fn(
                dit=dit,
                latents=chunk,
                timestep=frame_timesteps,
                context=context,
                action=None if action is None else action[:, : 4 * end],
                kv_cache=kv_cache,
                start_frame=start,
            )
            return flow, frame_timesteps

        def keep_first_frame(chunk, start):
            if start != 0:
                return chunk
            return torch.cat([first_frame, chunk[:, :, :, 1:]], dim=3)

        outputs = []
        for start in range(0, latent_frames, CAUSAL_CHUNK_SIZE):
            chunk = keep_first_frame(
                noise[:, :, :, start : start + CAUSAL_CHUNK_SIZE].clone(), start
            )
            for step_id, timestep in enumerate(steps):
                with torch.set_grad_enabled(with_grad and step_id == exit_id):
                    flow, frame_timesteps = call(chunk, timestep, start)
                sigma = scheduler._expand_sigma_for_sample(
                    scheduler._lookup_sigma(frame_timesteps, device=chunk.device), chunk
                )
                denoised = keep_first_frame(
                    (chunk - sigma * flow).to(chunk.dtype), start
                )
                if step_id == exit_id:
                    break
                next_timesteps = torch.full_like(frame_timesteps, steps[step_id + 1])
                if start == 0:
                    next_timesteps[:, 0] = 0
                with torch.no_grad():
                    chunk = scheduler.add_noise(
                        denoised, torch.randn_like(denoised), next_timesteps
                    )
                    chunk = keep_first_frame(chunk, start)
            outputs.append(denoised)
            with torch.no_grad():  # commit the clean chunk to the cache
                call(denoised.detach(), 0.0, start)
        return torch.cat(outputs, dim=3)

    def self_forcing_segment(
        self,
        dit,
        noise,
        context,
        action,
        first_frame,
        consistency_steps,
        *,
        with_grad,
        kv_cache=None,
        sequence_start_frame=0,
        action_target_offset=1,
        sigma_shift=5.0,
        local_attn_size=21,
        sink_size=3,
        last_step_only=True,
    ):
        """Roll out one stage-4 segment while preserving self-attention history.

        The opening segment contains a fixed first latent and therefore uses
        ``action_target_offset=1``. Continuations contain only new latents, use
        offset zero, and append to the same bounded KV cache at the global RoPE
        position ``sequence_start_frame``.
        """
        batch_size, latent_frames = noise.shape[0], noise.shape[3]
        if latent_frames % CAUSAL_CHUNK_SIZE:
            raise ValueError(
                f"Stage-4 segments need whole {CAUSAL_CHUNK_SIZE}-latent chunks."
            )
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(1000, shift=sigma_shift)
        grid = torch.cat([scheduler.timesteps, torch.zeros(1)])
        steps = grid[1000 - torch.as_tensor(consistency_steps, dtype=torch.long)]
        if last_step_only:
            exit_id = len(steps) - 1
        else:
            exit_tensor = torch.randint(0, len(steps), (1,), device=noise.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(exit_tensor, src=0)
            exit_id = int(exit_tensor.item())

        if kv_cache is None:
            kv_cache = dit.new_kv_cache(
                noise,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
            )
        action_start_frame = int(sequence_start_frame)

        def call(chunk, timestep, local_start):
            global_start = int(sequence_start_frame) + local_start
            frame_timesteps = torch.full(
                (batch_size, chunk.shape[3]),
                float(timestep),
                dtype=torch.float32,
                device=noise.device,
            )
            if global_start == 0 and first_frame is not None:
                frame_timesteps[:, 0] = 0
            flow = self.model_fn(
                dit=dit,
                latents=chunk,
                timestep=frame_timesteps,
                context=context,
                action=action,
                kv_cache=kv_cache,
                start_frame=global_start,
                action_start_frame=action_start_frame,
                action_target_offset=action_target_offset,
            )
            return flow, frame_timesteps

        def keep_first(chunk, local_start):
            if local_start != 0 or first_frame is None:
                return chunk
            return torch.cat([first_frame, chunk[:, :, :, 1:]], dim=3)

        outputs = []
        for local_start in range(0, latent_frames, CAUSAL_CHUNK_SIZE):
            chunk = keep_first(
                noise[:, :, :, local_start : local_start + CAUSAL_CHUNK_SIZE].clone(),
                local_start,
            )
            for step_id, timestep in enumerate(steps):
                with torch.set_grad_enabled(with_grad and step_id == exit_id):
                    flow, frame_timesteps = call(chunk, timestep, local_start)
                sigma = scheduler._expand_sigma_for_sample(
                    scheduler._lookup_sigma(frame_timesteps, device=chunk.device),
                    chunk,
                )
                denoised = keep_first(
                    (chunk - sigma * flow).to(chunk.dtype), local_start
                )
                if step_id == exit_id:
                    break
                next_timesteps = torch.full_like(frame_timesteps, steps[step_id + 1])
                if (
                    int(sequence_start_frame) + local_start == 0
                    and first_frame is not None
                ):
                    next_timesteps[:, 0] = 0
                with torch.no_grad():
                    chunk = scheduler.add_noise(
                        denoised, torch.randn_like(denoised), next_timesteps
                    )
                    chunk = keep_first(chunk, local_start)
            outputs.append(denoised)
            with torch.no_grad():
                call(denoised.detach(), 0.0, local_start)
        return torch.cat(outputs, dim=3), kv_cache, float(steps[exit_id])

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        # First frame of each view, [B, V, C, H, W] -- required (raised if missing)
        input_images: Optional[torch.Tensor] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        batch_size: Optional[int] = 1,
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Action conditioning
        action: Optional[torch.Tensor] = None,
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        # Inputs
        inputs = {
            "prompt": prompt,
            "batch_size": batch_size,
            "input_images": input_images,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "sigma_shift": sigma_shift,
            "action": action,
        }
        inputs = self.units(inputs)

        # Denoise
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        if self.dit.causal and self.consistency_steps is not None:
            inputs["latents"] = self.denoise_consistency(
                models, inputs, self.consistency_steps, progress_bar_cmd
            )
        elif self.dit.causal:
            inputs["latents"] = self.denoise_autoregressive(
                models, inputs, progress_bar_cmd
            )
        else:
            for progress_id, timestep in enumerate(
                progress_bar_cmd(self.scheduler.timesteps)
            ):
                timestep = timestep.unsqueeze(0).to(
                    dtype=self.torch_dtype, device=self.device
                )
                noise_pred = self.model_fn(**models, **inputs, timestep=timestep)

                # Scheduler
                inputs["latents"] = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    inputs["latents"],
                )
                inputs["latents"][:, :, :, 0:1] = inputs["first_frame_latents"]

        # Decode
        B, V, C, T, H, W = inputs["latents"].shape
        video = rearrange(inputs["latents"], "b v c t h w -> (b v) c t h w")
        video = self.vae.decode(video, device=self.device)
        video = rearrange(video, "(b v) c t h w -> b c t (v h) w", v=V)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass

        return video


def model_fn_wan_video(
    dit: PrimeSimModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    action: Optional[torch.Tensor] = None,
    clean_latents: Optional[torch.Tensor] = None,
    kv_cache: Optional[list] = None,
    start_frame: int = 0,
    action_start_frame: int = 0,
    action_target_offset: int = 1,
    use_gradient_checkpointing: bool = False,
    **kwargs,
):
    """The pipeline's model call. The loss and __call__ hand it every pipeline input;
    the DiT takes only these. Everything world-model-specific lives in
    PrimeSimModel.forward."""
    return dit(
        latents,
        timestep,
        context,
        action=action,
        clean_latents=clean_latents,
        kv_cache=kv_cache,
        start_frame=start_frame,
        action_start_frame=action_start_frame,
        action_target_offset=action_target_offset,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
