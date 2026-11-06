"""Stage 2: distil the causal (stage 1) world model into a few-step one."""

import copy
from collections import OrderedDict
from contextlib import contextmanager

import torch
import torch.nn as nn
from accelerate import Accelerator
from mmengine.registry import MODELS

from primesim.pipeline import PrimeSimPipeline

from .training_module import BaseTrainModule


def _lora_layers(module: nn.Module):
    from peft.tuners.lora import LoraLayer

    return OrderedDict(
        (name, layer)
        for name, layer in module.named_modules()
        if isinstance(layer, LoraLayer)
    )


class DenseLoRAEMA:
    """FP32 EMA of each LoRA layer's merged weight delta, EMA(B @ A).

    Averaging the two LoRA factors separately would give EMA(B) @ EMA(A), which is
    not the same. One dense FP32 matrix per LoRA layer is kept beside the student
    layer. The EMA model stays LoRA-wrapped, so it has the student's layout, but its
    adapters are zero and its base weights are W_base + EMA(B @ A).
    """

    def __init__(self, student: nn.Module, decay: float):
        self.decay = float(decay)
        layers = _lora_layers(student)
        if not layers:
            raise RuntimeError("The student has no LoRA layers.")
        lora_ids = {
            id(p)
            for layer in layers.values()
            for adapters in (layer.lora_A, layer.lora_B)
            for p in adapters.parameters()
        }
        other = [
            n
            for n, p in student.named_parameters()
            if p.requires_grad and id(p) not in lora_ids
        ]
        if other:
            raise RuntimeError(f"Trainable parameters outside LoRA: {other[:10]}")
        self.shadow = OrderedDict(
            (n, self._delta(layer)) for n, layer in layers.items()
        )

    @staticmethod
    @torch.no_grad()
    def _delta(layer) -> torch.Tensor:
        weight = layer.get_base_layer().weight
        delta = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
        for adapter in layer.active_adapters:
            delta += layer.get_delta_weight(adapter).detach().float()
        return delta

    @torch.no_grad()
    def update(self, student: nn.Module):
        layers = _lora_layers(student)
        for name, shadow in self.shadow.items():
            shadow.mul_(self.decay).add_(
                self._delta(layers[name]), alpha=1 - self.decay
            )

    @torch.no_grad()
    def copy_to(self, student: nn.Module, target: nn.Module):
        """target = student's frozen base weights + EMA delta, adapters zeroed."""
        student_layers, target_layers = _lora_layers(student), _lora_layers(target)
        for name, shadow in self.shadow.items():
            weight = target_layers[name].get_base_layer().weight
            weight.copy_(student_layers[name].get_base_layer().weight)
            weight.add_(shadow.to(device=weight.device, dtype=weight.dtype))
            for adapters in (target_layers[name].lora_A, target_layers[name].lora_B):
                for p in adapters.parameters():
                    p.zero_()


@torch.no_grad()
def merged_state_dict(module: nn.Module) -> OrderedDict:
    """CPU state dict of a LoRA-wrapped DiT with each adapter folded into its base
    weight and the original (unwrapped) key names, so it loads as a plain
    PrimeSimModel checkpoint. The module itself is left untouched (merging in
    place and back would drift the frozen base weights by rounding)."""
    deltas = {}
    for name, layer in _lora_layers(module).items():
        weight = layer.get_base_layer().weight
        delta = sum(layer.get_delta_weight(a) for a in layer.active_adapters)
        deltas[f"{name}.base_layer.weight"] = weight + delta.to(weight.dtype)
    return OrderedDict(
        (
            name.replace(".base_layer.", "."),
            deltas.get(name, value).detach().cpu().clone(),
        )
        for name, value in module.state_dict().items()
        if ".lora_A." not in name and ".lora_B." not in name
    )


@MODELS.register_module()
class PrimeSimDistillModule(BaseTrainModule):
    """Stage 2 training module: causal consistency distillation.

    Starts from a stage 1 (causal) checkpoint. The student is that DiT with LoRA
    adapters, the only trainable weights; the teacher is a frozen copy (its zero
    LoRA leaves it exactly stage 1); the target is an EMA of the student, updated
    after every optimizer step. Sample videos use the EMA model with the few-step
    consistency sampler, and checkpoints are the EMA model (the student before
    ema_start_step) with LoRA merged, i.e. plain PrimeSimModel weights.
    """

    def __init__(
        self,
        action_dim,
        model,
        lora_rank=32,
        lora_target_modules="",
        ema_decay=0.99,
        ema_start_step=200,
        negative_prompt="",
        consistency_steps=(1000, 750, 500, 250),
        device="cpu",
        loss=None,
        accelerator=Accelerator,
        **kwargs,
    ):
        super().__init__()
        if device == "gpu":
            device = accelerator.device
        self.pipe = PrimeSimPipeline.from_pretrained(
            **model,
            action_dim=action_dim,
            causal=True,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        self.pipe.scheduler.set_timesteps(1000)
        self.pipe.freeze_except(["dit"])
        self.pipe.dit.requires_grad_(False)
        self.pipe.dit = self.add_lora_to_model(
            self.pipe.dit,
            target_modules=self.parse_lora_target_modules(
                self.pipe.dit, lora_target_modules
            ),
            lora_rank=lora_rank,
            upcast_dtype=self.pipe.torch_dtype,
        )
        trainable = sum(
            p.numel() for p in self.pipe.dit.parameters() if p.requires_grad
        )
        print(f"[stage 2] LoRA rank {lora_rank}: {trainable / 1e6:.1f}M trainable")
        # Copies after the LoRA injection, so all three share the student's layout.
        self.teacher_dit = copy.deepcopy(self.pipe.dit)
        self.ema_dit = copy.deepcopy(self.pipe.dit)
        self._freeze_copies()
        self.pipe.consistency_steps = tuple(consistency_steps)
        self.loss = loss
        self.ema_decay = ema_decay
        self.ema_start_step = ema_start_step
        self.negative_prompt = negative_prompt
        self.ema = None
        self.ema_steps = 0

    def _freeze_copies(self):
        for dit in (self.teacher_dit, self.ema_dit):
            dit.requires_grad_(False)
            dit.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "ema_dit"):
            self._freeze_copies()
        return self

    def on_after_prepare(self, accelerator):
        """Start the EMA once the model is on its device."""
        self.ema = DenseLoRAEMA(self.pipe.dit, decay=self.ema_decay)
        self.ema.copy_to(self.pipe.dit, self.ema_dit)

    def on_after_optimizer_step(self):
        self.ema_steps += 1
        self.ema.update(self.pipe.dit)
        self.ema.copy_to(self.pipe.dit, self.ema_dit)

    @contextmanager
    def sampling(self):
        """Sample videos with the EMA model."""
        student, self.pipe.dit = self.pipe.dit, self.ema_dit
        try:
            yield
        finally:
            self.pipe.dit = student

    def state_dict_for_save(self, accelerator):
        """The EMA model once past ema_start_step (else the student), LoRA merged."""
        if not accelerator.is_main_process:
            return OrderedDict()
        use_ema = self.ema_steps > self.ema_start_step
        return merged_state_dict(self.ema_dit if use_ema else self.pipe.dit)

    def forward(self, inputs):
        inputs = self.pipe.units(inputs)
        negative_context = self.pipe.encode_prompt(
            [self.negative_prompt] * inputs["batch_size"]
        )
        return self.loss(
            self.pipe,
            teacher_dit=self.teacher_dit,
            ema_dit=self.ema_dit,
            negative_context=negative_context,
            **inputs,
        )
