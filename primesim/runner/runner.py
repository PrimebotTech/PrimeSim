import copy

import torch
from accelerate import Accelerator
from tqdm import tqdm

from primesim.modules import BaseTrainModule
from primesim.utils.video_utils import generate_and_save_comparison_video

from .logger import ModelLogger


def to_device(data, device, torch_float_dtype=None):
    """Move every tensor in a (nested) dict / list / tuple to `device`, casting
    floating-point tensors to `torch_float_dtype`."""
    if isinstance(data, torch.Tensor):
        data = data.to(device)
        if torch_float_dtype is not None and data.dtype in (
            torch.float,
            torch.float16,
            torch.bfloat16,
        ):
            data = data.to(torch_float_dtype)
        return data
    if isinstance(data, (tuple, list)):
        return type(data)(to_device(x, device, torch_float_dtype) for x in data)
    if isinstance(data, dict):
        return {k: to_device(v, device, torch_float_dtype) for k, v in data.items()}
    return data


def pipeline_inputs(pipe, data):
    """A dataloader batch -> the pipeline's inputs dict (not yet on the device)."""
    videos = data["videos"]  # [B, V, C, T, H, W]
    inputs = {
        "prompt": data["prompt"],
        "input_videos": videos,
        "input_images": videos[:, :, :, 0],  # each view's first frame
        "action": data["action"],
        "batch_size": videos.shape[0],
        "height": videos.shape[-2],
        "width": videos.shape[-1],
        "num_frames": videos.shape[-3],
        "rand_device": pipe.device,
        "use_gradient_checkpointing": True,
    }
    # Stage 4 rebases absolute actions after collation and consumes exact sampler
    # boundary metadata. Other stages simply do not provide these keys.
    for key in (
        "state",
        "action_mask",
        "actions_are_relative",
        "episode_start",
        "episode_end",
        "amortized_episode_id",
        "amortized_segment_index",
        "amortized_episode_num_segments",
    ):
        if key in data:
            inputs[key] = data[key]
    return inputs


def compute_loss(accelerator, model, data):
    """The training module's loss on one dataloader batch."""
    pipe = accelerator.unwrap_model(model).pipe
    inputs = to_device(pipeline_inputs(pipe, data), pipe.device, pipe.torch_dtype)
    return model(inputs)


def log_wandb(metrics: dict, step: int):
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def gen_seeded(seed, fn):
    """Run ``fn`` under a fixed torch RNG seed, then restore the prior state.

    The loss samples a random timestep and random noise per call, and video generation
    samples noise too, so an unseeded val pass measures mostly sampling variance rather
    than model change. Resetting the RNG to the same seed for every pass makes
    successive
    passes comparable; it does mean each pass is a fixed-noise estimate, not an
    expectation over all timesteps/noise draws.
    """
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    torch.manual_seed(seed)
    try:
        return fn()
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def set_eval_mode(model, accelerator):
    """Switch the model to eval, returning a restore closure.

    ``generate_and_save_comparison_video`` leaves the scheduler in inference mode, so
    the
    restore resets the scheduler's training timesteps on the way back out -- the loss
    path
    needs them, and it is harmless if they were never changed.
    """
    unwrapped = accelerator.unwrap_model(model)
    was_training = model.training
    model.eval()

    def restore():
        if was_training:
            model.train()
        unwrapped.pipe.scheduler.set_timesteps(1000)

    return restore


@torch.no_grad()
def validate(accelerator, model, val_dataloader, max_batches, seed):
    """Mean validation loss, averaged over processes.

    Returns None when there is nothing to validate (empty split, or the dataloader is
    None),
    so the caller can skip logging rather than record a fake zero.
    """
    if val_dataloader is None:
        return None

    def run_pass():
        total, count = 0.0, 0
        for i, data in enumerate(val_dataloader):
            if max_batches and i >= max_batches:
                break
            loss = compute_loss(accelerator, model, data)
            total += loss.detach().float()
            count += 1
        return total, count

    restore = set_eval_mode(model, accelerator)
    try:
        total, count = gen_seeded(seed, run_pass)
    finally:
        restore()
    if count == 0:
        return None
    local_mean = (total / count).to(accelerator.device)
    # average across ranks so every process logs the same number
    gathered = accelerator.gather(local_mean.reshape(1))
    return float(gathered.mean().item())


@torch.no_grad()
def validate_video_metrics(
    accelerator,
    model,
    val_dataloader,
    max_batches,
    seed,
    *,
    fps,
    num_inference_steps,
):
    """Video/image-reconstruction metrics on generated val samples.

    Iterates up to ``max_batches`` val batches, generates a video from one sample per
    batch
    and computes ``eval_images`` against the ground truth. Returns mean metrics over the
    generated samples, averaged across processes, or None when the loader is empty.
    """
    if val_dataloader is None:
        return None, None

    unwrapped = accelerator.unwrap_model(model)
    representative_video = None

    def run_pass():
        nonlocal representative_video
        totals, count = {}, 0
        m = 0
        for data in val_dataloader:
            if max_batches and m >= max_batches:
                break
            m += 1
            inputs = pipeline_inputs(unwrapped.pipe, data)
            gen_output = generate_and_save_comparison_video(
                pipe=unwrapped.pipe,
                data=inputs,
                model_logger=None,
                fps=fps,
                wandb_active=False,
                num_inference_steps=num_inference_steps,
                metric_prefix="val/img_recon_",
                caption="val/gen_frames_vs_gt_frames",
            )
            representative_video = gen_output.get("video")
            for k, v in gen_output.get("metrics", {}).items():
                totals[k] = totals.get(k, 0.0) + float(v)
            count += 1
        return totals, count

    restore = set_eval_mode(model, accelerator)
    try:
        with unwrapped.sampling():
            total_metrics, count = gen_seeded(seed, run_pass)
    finally:
        restore()
    torch.cuda.empty_cache()
    if count == 0:
        return None, None
    per = {
        k: (torch.tensor(v / count)).to(accelerator.device)
        for k, v in total_metrics.items()
    }
    if not per:
        return None, representative_video
    keys = list(per.keys())
    gathered = accelerator.gather_for_metrics(torch.stack([per[k] for k in keys]))
    metrics = {keys[i]: float(gathered[i].mean().item()) for i in range(len(keys))}
    return metrics, representative_video


def total_grad_norm(model):
    """Compute total L2 gradient norm of trainable parameters."""
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total += p.grad.data.float().norm(2).item() ** 2
    return total**0.5


def is_due(step: int, every: int) -> bool:
    """True on every `every`-th step; never when `every` is 0 / None."""
    return bool(every) and step > 0 and step % every == 0


def train_step(
    accelerator,
    model,
    optimizer,
    scheduler,
    model_logger,
    data,
    *,
    save_steps,
    compute_grad_norm,
):
    """One optimisation step: forward, backward, clip, update, checkpoint, LR step.

    Returns the loss and the clipped gradient norm (0.0 unless `compute_grad_norm`).
    """
    unwrapped = accelerator.unwrap_model(model)
    if getattr(unwrapped, "stage4_chain_training", False):
        return train_stage4_step(
            accelerator,
            model,
            optimizer,
            scheduler,
            model_logger,
            data,
            save_steps=save_steps,
            compute_grad_norm=compute_grad_norm,
        )

    optimizer.zero_grad()
    phases = unwrapped.training_phases()
    if phases is None:
        loss = compute_loss(accelerator, model, data)
        accelerator.backward(loss)
    else:
        # Several losses per step (stage 3: generator, critic), each with its own
        # backward, on the same prepared inputs; one optimizer step for all.
        pipe = unwrapped.pipe
        inputs = to_device(pipeline_inputs(pipe, data), pipe.device, pipe.torch_dtype)
        loss = 0.0
        for phase in phases:
            phase_loss = model(inputs, phase=phase)
            accelerator.backward(phase_loss)
            loss = loss + phase_loss.detach()
    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
    grad_norm = total_grad_norm(model) if compute_grad_norm else 0.0
    optimizer.step()
    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
        accelerator.unwrap_model(model).on_after_optimizer_step()
    model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
    scheduler.step()
    return loss, grad_norm, True


def _base_optimizer(optimizer):
    candidate = optimizer
    while hasattr(candidate, "optimizer"):
        candidate = candidate.optimizer
    return candidate


def _clear_grads(parameters):
    for parameter in parameters:
        parameter.grad = None


def train_stage4_step(
    accelerator,
    model,
    optimizer,
    scheduler,
    model_logger,
    data,
    *,
    save_steps,
    compute_grad_norm,
):
    """One cross-window stage-4 segment.

    Critic AdamW steps after every segment. Generator gradients are computed on
    one sampled segment, retained without changing generator weights, and applied
    only at the sampler-delimited chain boundary.
    """
    unwrapped = accelerator.unwrap_model(model)
    phases = unwrapped.training_phases(data)
    opening = bool(torch.as_tensor(data["episode_start"]).any())
    if opening:
        optimizer.zero_grad()

    phase_optimizer = _base_optimizer(optimizer)
    if not getattr(phase_optimizer, "supports_phase_steps", False):
        raise TypeError("Stage 4 requires the PhaseAdamW optimizer.")
    inputs = pipeline_inputs(unwrapped.pipe, data)
    loss = torch.zeros((), device=accelerator.device)
    for phase in phases:
        parameters = unwrapped.phase_parameters(phase)
        if phase == "critic":
            _clear_grads(parameters)
        phase_loss = model(inputs, phase=phase)
        accelerator.backward(phase_loss)
        loss = loss + phase_loss.detach()
        if phase == "critic":
            accelerator.clip_grad_norm_(parameters, max_norm=1.0)
            phase_optimizer.set_active_phase("critic")
            optimizer.step()
            _clear_grads(parameters)

    chain_end = unwrapped.chain_boundary()
    grad_norm = total_grad_norm(model) if compute_grad_norm else 0.0
    if chain_end:
        generator_parameters = unwrapped.phase_parameters("generator")
        accelerator.clip_grad_norm_(generator_parameters, max_norm=1.0)
        phase_optimizer.set_active_phase("generator")
        optimizer.step()
        _clear_grads(generator_parameters)
        if not accelerator.optimizer_step_was_skipped:
            unwrapped.on_after_optimizer_step()
        scheduler.step()
    optimizer_stepped = chain_end and not accelerator.optimizer_step_was_skipped
    model_logger.on_step_end(
        accelerator,
        model,
        save_steps,
        loss=loss,
        step_happened=optimizer_stepped,
    )
    return loss, grad_norm, optimizer_stepped


class MetricWindow:
    """Loss / grad-norm values since the last train-metrics log."""

    def __init__(self):
        self.losses, self.grad_norms = [], []

    def add(self, loss: float, grad_norm: float):
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)

    def flush(self):
        """Means since the last flush, then start a new window."""
        means = (
            sum(self.losses) / len(self.losses),
            sum(self.grad_norms) / len(self.grad_norms),
        )
        self.losses, self.grad_norms = [], []
        return means


def log_sample_video(
    accelerator,
    model,
    model_logger,
    data,
    *,
    step,
    fps,
    wandb_active,
):
    """Generate a video from this training batch's first frame and log it beside the
    ground truth, with its image metrics."""
    video_data = copy.deepcopy(data)
    unwrapped = accelerator.unwrap_model(model)
    inputs = pipeline_inputs(unwrapped.pipe, video_data)
    model.eval()
    with torch.no_grad(), unwrapped.sampling():
        generate_and_save_comparison_video(
            pipe=unwrapped.pipe,
            data=inputs,
            model_logger=model_logger,
            wandb_active=wandb_active,
            fps=fps,
            step=step,
            metric_prefix="train/img_recon_",
            caption="train/gen_frames_vs_gt_frames",
        )
    model.train()
    unwrapped.pipe.scheduler.set_timesteps(1000)
    # video_data is a deepcopy of a whole video batch and inputs holds the encoded
    # latents; drop them before validation, which fires on this same step whenever
    # val_steps and save_frame_steps coincide.
    del video_data, inputs
    torch.cuda.empty_cache()


def log_step(
    accelerator,
    model,
    optimizer,
    model_logger,
    data,
    window,
    *,
    loss,
    grad_norm,
    step,
    epoch,
    log_steps,
    save_frame_steps,
    fps,
    wandb_active,
    optimizer_stepped=True,
):
    """Everything logged after a training step: the sample video every
    save_frame_steps, and the windowed train loss / grad norm / LR every log_steps."""
    if optimizer_stepped and is_due(step, save_frame_steps):
        log_sample_video(
            accelerator,
            model,
            model_logger,
            data,
            step=step,
            fps=fps,
            wandb_active=wandb_active,
        )
    window.add(loss.item(), grad_norm)
    if optimizer_stepped and is_due(step, log_steps):
        avg_loss, avg_grad_norm = window.flush()
        if wandb_active:
            log_wandb(
                {
                    "train/loss": avg_loss,
                    "train/grad_norm": avg_grad_norm,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                },
                step=step,
            )


def run_validation(
    accelerator,
    model,
    val_dataloader,
    *,
    step,
    val_steps,
    val_batches,
    val_img_steps,
    val_img_batches,
    val_seed,
    fps,
    wandb_active,
):
    """Validation due at this step: the val loss every val_steps, and generated-video
    metrics every val_img_steps."""
    if is_due(step, val_steps):
        val_loss = validate(
            accelerator, model, val_dataloader, max_batches=val_batches, seed=val_seed
        )
        if val_loss is not None:
            if accelerator.is_main_process:
                print(f"[val] step {step}: val/loss={val_loss:.5f}")
            if wandb_active:
                log_wandb({"val/loss": val_loss}, step=step)
    if is_due(step, val_img_steps):
        val_metrics, val_video = validate_video_metrics(
            accelerator,
            model,
            val_dataloader,
            max_batches=val_img_batches,
            seed=val_seed,
            fps=fps,
            num_inference_steps=50,
        )
        if val_metrics is not None:
            if accelerator.is_main_process:
                line = " ".join(f"{k}={v:.5f}" for k, v in val_metrics.items())
                print(f"[val-img] step {step}: {line}")
            if wandb_active:
                log_wandb(
                    {**val_metrics, "val/gen_frames_vs_gt_frames": val_video},
                    step=step,
                )


def launch_training_task(
    accelerator: Accelerator,
    dataloader: torch.utils.data.DataLoader,
    model: BaseTrainModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_logger: ModelLogger,
    fps: int,
    log_steps: int,
    num_epochs: int,
    num_workers: int = 1,
    save_steps: int = None,
    max_train_steps: int = 500_000,
    save_frame_steps: int = 500,
    val_steps: int = 0,
    val_batches: int = 0,
    val_seed: int = 12345,
    val_img_steps: int = 0,
    val_img_batches: int = 0,
    val_dataloader: torch.utils.data.DataLoader = None,
    wandb_active: bool = False,
    **kwargs,
):
    """Train `model` (already passed through accelerator.prepare, with `optimizer`,
    `scheduler` and the dataloaders) for up to `max_train_steps` steps."""

    reached_max_steps = False
    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id}", disable=not wandb_active)
        window = MetricWindow()
        for data in pbar:
            with accelerator.accumulate(model):
                loss, grad_norm, optimizer_stepped = train_step(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    model_logger,
                    data,
                    save_steps=save_steps,
                    compute_grad_norm=wandb_active,
                )
                current_step = model_logger.num_steps
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                log_step(
                    accelerator,
                    model,
                    optimizer,
                    model_logger,
                    data,
                    window,
                    loss=loss,
                    grad_norm=grad_norm,
                    step=current_step,
                    epoch=epoch_id,
                    log_steps=log_steps,
                    save_frame_steps=save_frame_steps,
                    fps=fps,
                    wandb_active=wandb_active,
                    optimizer_stepped=optimizer_stepped,
                )
                if optimizer_stepped:
                    run_validation(
                        accelerator,
                        model,
                        val_dataloader,
                        step=current_step,
                        val_steps=val_steps,
                        val_batches=val_batches,
                        val_img_steps=val_img_steps,
                        val_img_batches=val_img_batches,
                        val_seed=val_seed,
                        fps=fps,
                        wandb_active=wandb_active,
                    )
            if max_train_steps is not None and current_step >= max_train_steps:
                if accelerator.is_main_process:
                    print(f"[lr] reached max_train_steps={max_train_steps}, stopping.")
                reached_max_steps = True
                break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        if reached_max_steps:
            break
    model_logger.on_training_end(accelerator, model, save_steps)
