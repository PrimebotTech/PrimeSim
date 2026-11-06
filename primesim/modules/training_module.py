from contextlib import contextmanager

import torch
from peft import LoraConfig, inject_adapter_in_model

from primesim.utils.loader import load_state_dict


class BaseTrainModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    # Hooks the runner calls; later training stages override them as needed.
    def on_after_prepare(self, accelerator):
        """After accelerator.prepare."""

    def on_after_optimizer_step(self):
        """After every optimizer step."""

    def training_phases(self):
        """None: one loss per step. Otherwise the phases of a step, each a forward
        model(inputs, phase=...) and backward (stage 3)."""
        return None

    @contextmanager
    def sampling(self):
        """Around every sample-video generation."""
        yield

    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules

    def trainable_param_names(self):
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.named_parameters(),
            )
        )
        trainable_param_names = set(
            [named_param[0] for named_param in trainable_param_names]
        )
        return trainable_param_names

    def add_lora_to_model(
        self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None
    ):
        if lora_alpha is None:
            lora_alpha = lora_rank
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules
        )
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model

    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace(
                    "lora_B.weight", "lora_B.default.weight"
                )
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if name in trainable_param_names
        }
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix) :]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict

    def auto_detect_lora_target_modules(
        self,
        model: torch.nn.Module,
        search_for_linear=False,
        linear_detector=lambda x: min(x.weight.shape) >= 512,
        block_list_detector=lambda x: isinstance(x, torch.nn.ModuleList) and len(x) > 1,
        name_prefix="",
    ):
        lora_target_modules = []
        if search_for_linear:
            for name, module in model.named_modules():
                module_name = name_prefix + ["", "."][name_prefix != ""] + name
                if isinstance(module, torch.nn.Linear) and linear_detector(module):
                    lora_target_modules.append(module_name)
        else:
            for name, module in model.named_children():
                module_name = name_prefix + ["", "."][name_prefix != ""] + name
                lora_target_modules += self.auto_detect_lora_target_modules(
                    module,
                    search_for_linear=block_list_detector(module),
                    linear_detector=linear_detector,
                    block_list_detector=block_list_detector,
                    name_prefix=module_name,
                )
        return lora_target_modules

    def parse_lora_target_modules(self, model, lora_target_modules):
        if lora_target_modules == "":
            print(
                "No LoRA target modules specified. The framework will automatically "
                "search for them."
            )
            lora_target_modules = self.auto_detect_lora_target_modules(model)
            print(f"LoRA will be patched at {lora_target_modules}.")
        else:
            lora_target_modules = lora_target_modules.split(",")
        return lora_target_modules

    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000)

        # Freeze untrainable models
        pipe.freeze_except(
            [] if trainable_models is None else trainable_models.split(",")
        )

        # Preset LoRA
        if preset_lora_path is not None:
            pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)

        # FP8
        # FP8 relies on a model-specific memory management scheme.
        # It is delegated to the subclass.

        # Add LoRA to the base models
        if lora_base_model is not None:
            if (not hasattr(pipe, lora_base_model)) or getattr(
                pipe, lora_base_model
            ) is None:
                print(
                    f"No {lora_base_model} models in the pipeline. "
                    f"We cannot patch LoRA on the model. "
                    f"If this occurs during the data processing stage, it is normal."
                )
                return
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=self.parse_lora_target_modules(
                    getattr(pipe, lora_base_model), lora_target_modules
                ),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(
                    f"LoRA checkpoint loaded: {lora_checkpoint}, total "
                    f"{len(state_dict)} keys"
                )
                if len(load_result[1]) > 0:
                    print(
                        f"Warning, LoRA key mismatch! Unexpected keys in LoRA "
                        f"checkpoint: {load_result[1]}"
                    )
            setattr(pipe, lora_base_model, model)
