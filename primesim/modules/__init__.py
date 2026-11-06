import importlib

from .primesim_dit import (
    CAUSAL_CHUNK_SIZE,
    NUM_VIEWS,
    WORLD_MODEL_PREFIXES,
    PrimeSimModel,
)
from .t5_text_encoder import HuggingfaceTokenizer, T5TextEncoder
from .training_module import BaseTrainModule
from .wan_video_vae import WanVideoVAE38

# The training modules build a PrimeSimPipeline, and the pipeline imports the models
# above from this package, so they are imported on first use rather than here.
_TRAIN_MODULES = {
    "PrimeSimTrainModule": ".primesim_train_module",
    "PrimeSimDistillModule": ".primesim_distill_module",
    "PrimeSimDMDModule": ".primesim_dmd_module",
    "PrimeSimSelfForcingModule": ".primesim_self_forcing_module",
}


def __getattr__(name):
    if name in _TRAIN_MODULES:
        return getattr(importlib.import_module(_TRAIN_MODULES[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CAUSAL_CHUNK_SIZE",
    "NUM_VIEWS",
    "WORLD_MODEL_PREFIXES",
    "BaseTrainModule",
    "HuggingfaceTokenizer",
    "PrimeSimDMDModule",
    "PrimeSimDistillModule",
    "PrimeSimModel",
    "PrimeSimSelfForcingModule",
    "PrimeSimTrainModule",
    "T5TextEncoder",
    "WanVideoVAE38",
]
