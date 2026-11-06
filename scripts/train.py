import math
import os

# Set before torch initializes the CUDA caching allocator. Expandable segments let the
# allocator grow existing segments instead of stranding memory in wrong-sized blocks,
# which matters most right after the sample-video pass, where several GB of
# reserved-but-unallocated memory used to sit next to the validation forward.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datetime import timedelta
from typing import Any, Dict

import accelerate
import hydra
import mmengine.optim  # noqa: F401  -- registers torch's optimizers in OPTIMIZERS
import torch
import torch.distributed as dist
from mmengine import DATASETS, Config
from mmengine.registry import MODELS, OPTIMIZERS
from omegaconf import OmegaConf
from primesim.loss import LOSS

# imported for their registrations in MODELS (fine-tuning and stage 1, stage 2,
# stage 3, stage 4)
from primesim.modules import (  # noqa: F401
    PrimeSimDistillModule,
    PrimeSimDMDModule,
    PrimeSimSelfForcingModule,
    PrimeSimTrainModule,
)
from primesim.runner import LOGGERS, launch_training_task

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def verify_dist():
    is_initialized = dist.is_initialized()
    print(f"   is_initialized: {is_initialized}")
    if not is_initialized:
        return
    print(f"🔍 [Rank {os.environ.get('RANK', '?')}]")
    print(f"   process_index: {dist.get_rank()}")
    print(f"   local_rank: {dist.get_rank() % torch.cuda.device_count()}")
    print(f"   world_size: {dist.get_world_size()}")
    print(f"   is_main_process: {dist.get_rank() == 0}")


def _param_groups(model, weight_decay: float):
    """Split trainable params into decay / no-decay groups.

    Weight decay on LayerNorm gains and biases attenuates the signal flowing through a
    block rather than regularising it, so those are excluded. ``ndim <= 1`` catches norm
    weights and biases together, which is the usual shorthand -- it matters more than
    usual for a DiT, where adaptive-LayerNorm modulation parameters *are* the
    conditioning mechanism.

    Returns (groups, stats) where stats is a list of (label, n_tensors, n_params).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups, stats = [], []
    for label, params, wd in (
        ("decay", decay, weight_decay),
        ("no_decay(norm/bias)", no_decay, 0.0),
    ):
        if not params:
            continue
        groups.append({"params": params, "weight_decay": wd})
        stats.append((label, wd, len(params), sum(p.numel() for p in params)))
    return groups, stats


def _build_lr_scheduler(
    optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
    num_processes: int,
):
    """Linear warmup then cosine decay to ``min_lr_ratio`` x base LR.

    ``accelerator.prepare()`` wraps this in an AcceleratedScheduler, which advances
    the wrapped scheduler once per process per optimizer step. The horizon therefore
    has to be expressed in those inner units -- hence the ``num_processes`` scaling.
    Without it the schedule would run ``num_processes`` times too fast and sit at the
    floor almost immediately.
    """
    warmup = max(1, warmup_steps * num_processes)
    total = max(warmup + 1, total_steps * num_processes)

    def lr_lambda(inner_step: int) -> float:
        if inner_step < warmup:
            return (inner_step + 1) / warmup
        progress = min(1.0, (inner_step - warmup) / (total - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model, optimizer_cfg, trainer, accelerator):
    """The configured optimizer over the trainable parameters (no weight decay on
    norms and biases), with linear warmup then cosine decay to lr_min_ratio x lr."""
    optimizer_cfg = dict(optimizer_cfg)
    # OmegaConf hands this over as a list; torch optimizers expect a 2-tuple.
    optimizer_cfg["betas"] = tuple(float(b) for b in optimizer_cfg["betas"])
    weight_decay = optimizer_cfg["weight_decay"]
    if optimizer_cfg["type"] == "ZeroRedundancyOptimizer":
        # Shards the optimizer state across DDP ranks; one process has nothing to
        # shard, so it uses the wrapped optimizer directly.
        if accelerator.num_processes == 1:
            optimizer_cfg = dict(
                type=optimizer_cfg["optimizer_type"],
                lr=optimizer_cfg["lr"],
                betas=optimizer_cfg["betas"],
                weight_decay=weight_decay,
            )
    if hasattr(model, "optimizer_param_groups"):
        param_groups, pg_stats = model.optimizer_param_groups(weight_decay)
    else:
        param_groups, pg_stats = _param_groups(model, weight_decay)
    optimizer = OPTIMIZERS.build(optimizer_cfg, default_args=dict(params=param_groups))
    learning_rate = optimizer_cfg.get("lr", optimizer.param_groups[0]["lr"])
    if trainer.get("constant_lr", False):
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    else:
        scheduler = _build_lr_scheduler(
            optimizer,
            warmup_steps=trainer.lr_warmup_steps,
            total_steps=trainer.max_train_steps,
            min_lr_ratio=trainer.lr_min_ratio,
            num_processes=accelerator.num_processes,
        )
    if accelerator.is_main_process:
        print(
            f"[optim] {type(optimizer).__name__} lr={learning_rate:g} "
            f"betas={optimizer_cfg['betas']}"
        )
        for label, wd, n_tensors, n_params in pg_stats:
            print(
                f"[optim] {label:<20} wd={wd:<7g} {n_tensors:>6} tensors  "
                f"{n_params / 1e6:>9.2f}M params"
            )
        if trainer.get("constant_lr", False):
            print(f"[lr] constant {learning_rate:g}")
        else:
            print(
                f"[lr] base={learning_rate:g} warmup={trainer.lr_warmup_steps} steps "
                f"-> cosine to {learning_rate * trainer.lr_min_ratio:g} at "
                f"{trainer.max_train_steps} steps ({accelerator.num_processes} "
                f"processes)"
            )
    return optimizer, scheduler


def init_wandb(accelerator, wandb_project, wandb_run_name, wandb_mode):
    """Initialize wandb on main process. Returns True if active."""
    if wandb_project is None or wandb_mode == "disabled":
        return False
    if not accelerator.is_main_process:
        return False
    try:
        import wandb

        os.environ["WANDB_MODE"] = wandb_mode
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )
        return True
    except ImportError:
        print("[WARN] wandb not installed, skipping.")
        return False


def finish_wandb():
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: Dict[str, Any]) -> None:
    cfg = Config(OmegaConf.to_container(cfg, resolve=True))
    find_unused_parameters = getattr(cfg.trainer, "find_unused_parameters", False)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=find_unused_parameters
            ),
            accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=2)),
        ],
    )
    verify_dist()
    if "datasets" not in cfg or "data_loader" not in cfg.datasets:
        raise KeyError(
            "Expected cfg.datasets.data_loader (see config/dataset/midtrain.yaml)."
        )
    # The registry object owns its DataLoader.
    datamodule = DATASETS.build(cfg.datasets.data_loader)
    dataloader = datamodule.train_dataloader()
    # Only build the val loader if validation is actually enabled, so a run with
    # val_steps=0 pays nothing for it.
    val_dataloader = (
        datamodule.val_dataloader()
        if (
            int(cfg.trainer.get("val_steps", 0) or 0) > 0
            or int(cfg.trainer.get("val_img_steps", 0) or 0) > 0
        )
        else None
    )
    model = MODELS.build(
        cfg.train_module,
        default_args=dict(loss=LOSS.build(cfg.loss), accelerator=accelerator),
    )
    model_logger = LOGGERS.build(cfg.model_logger)
    optimizer, scheduler = build_optimizer(
        model, cfg.optimizer, cfg.trainer, accelerator
    )
    model.to(device=accelerator.device)
    # A ZeroRedundancyOptimizer must not be moved by prepare: placing it reads its
    # state_dict, which needs every rank. Its parameters are on device already.
    sharded = type(optimizer).__name__ == "ZeroRedundancyOptimizer"
    rank_sharded_loader = bool(
        getattr(getattr(dataloader, "batch_sampler", None), "rank_sharded", False)
    )
    if rank_sharded_loader:
        # Stage 4 partitions complete chains inside its batch sampler. Wrapping the
        # loader would shard its already-local batches a second time and destroy
        # temporal adjacency.
        model, optimizer, scheduler = accelerator.prepare(
            model,
            optimizer,
            scheduler,
            device_placement=[None, False, None] if sharded else None,
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model,
            optimizer,
            dataloader,
            scheduler,
            device_placement=[None, False, None, None] if sharded else None,
        )
    accelerator.unwrap_model(model).on_after_prepare(accelerator)
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    wandb_active = init_wandb(
        accelerator,
        cfg.trainer.wandb_project,
        cfg.trainer.wandb_run_name,
        cfg.trainer.wandb_mode,
    )
    launch_training_task(
        accelerator,
        dataloader,
        model,
        optimizer,
        scheduler,
        model_logger,
        val_dataloader=val_dataloader,
        wandb_active=wandb_active,
        **cfg.trainer,
    )
    finish_wandb()


if __name__ == "__main__":
    import primesim

    # test
    print(f"======== primesim package path: {primesim.__path__} ========")
    print(f"======== primesim file path: {primesim.__file__} ========")
    main()
