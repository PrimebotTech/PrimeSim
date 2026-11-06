import os

import torch
from accelerate import Accelerator

from primesim.runner import LOGGERS


@LOGGERS.register_module()
class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps=None,
        step_happened=True,
        **kwargs,
    ):
        if not step_happened:
            return
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(
        self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None
    ):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        if hasattr(unwrapped, "state_dict_for_save"):  # stage 2: the merged EMA DiT
            state_dict = unwrapped.state_dict_for_save(accelerator)
        else:
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                state_dict = unwrapped.export_trainable_state_dict(
                    state_dict, remove_prefix=self.remove_prefix_in_ckpt
                )
        if accelerator.is_main_process:
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
