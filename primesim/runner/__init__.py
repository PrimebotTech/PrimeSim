from mmengine import Registry

LOGGERS = Registry("logger")

# after LOGGERS exists: importing the logger registers it there
from .logger import ModelLogger  # noqa: E402
from .phase_optimizer import PhaseAdamW  # noqa: E402, F401
from .runner import launch_training_task  # noqa: E402
