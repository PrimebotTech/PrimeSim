import torch
from accelerate import Accelerator
from mmengine.registry import MODELS

from primesim.pipeline import PrimeSimPipeline

from .training_module import BaseTrainModule


@MODELS.register_module()
class PrimeSimTrainModule(BaseTrainModule):
    def __init__(
        self,
        action_dim,
        model,
        causal=False,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        device="cpu",
        loss=None,
        accelerator=Accelerator,
        **kwargs,
    ):
        super().__init__()
        # Load models
        if device == "gpu":
            device = accelerator.device
        self.pipe = PrimeSimPipeline.from_pretrained(
            **model,
            action_dim=action_dim,
            causal=causal,
            torch_dtype=torch.bfloat16,
            device=device,
        )

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
        )
        self.loss = loss

        # The world-model parts always train, also when LoRA froze the rest of the DiT.
        self.pipe.dit.unfreeze_world_model()

    def forward(self, inputs):
        """Flow-matching loss for pipeline inputs (see runner.pipeline_inputs)."""
        inputs = self.pipe.units(inputs)
        return self.loss(self.pipe, **inputs)
