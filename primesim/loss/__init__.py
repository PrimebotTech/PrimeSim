from mmengine import Registry

LOSS = Registry("loss")

# after LOSS exists: importing the loss registers it there
from .consistency_distillation_loss import ConsistencyDistillationLoss  # noqa: E402
from .dmd_loss import DMDLoss  # noqa: E402
from .fm_sft_loss import FlowMatchSFTLoss  # noqa: E402
from .teacher_forcing_loss import TeacherForcingLoss  # noqa: E402

__all__ = [
    "ConsistencyDistillationLoss",
    "DMDLoss",
    "FlowMatchSFTLoss",
    "TeacherForcingLoss",
]
