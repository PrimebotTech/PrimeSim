"""Phase-aware AdamW for stage-4 generator/critic updates."""

import torch
from mmengine.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class PhaseAdamW(torch.optim.AdamW):
    """Update one named parameter group while preserving other groups' gradients."""

    supports_phase_steps = True

    def __init__(self, params, **kwargs):
        groups = list(params)
        phases = {group.get("phase") for group in groups}
        if None in phases or "" in phases:
            raise ValueError("Every PhaseAdamW parameter group needs a phase name.")
        super().__init__(groups, **kwargs)
        self.phases = frozenset(phases)
        self._active_phase = None

    def set_active_phase(self, phase):
        if phase not in self.phases:
            raise ValueError(
                f"Unknown optimizer phase {phase!r}; expected {sorted(self.phases)}."
            )
        self._active_phase = phase

    def step(self, closure=None):
        if self._active_phase is None:
            raise RuntimeError("Call set_active_phase() before PhaseAdamW.step().")
        all_groups = self.param_groups
        try:
            self.param_groups = [
                group
                for group in all_groups
                if group["phase"] == self._active_phase
            ]
            return super().step(closure=closure)
        finally:
            self.param_groups = all_groups
            self._active_phase = None
