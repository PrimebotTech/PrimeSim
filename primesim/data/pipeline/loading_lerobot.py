from typing import Any, Dict

from mmengine import TRANSFORMS


@TRANSFORMS.register_module()
class RefactorLerobot:
    """Rename the raw lerobot columns to the names the rest of the pipeline uses.

    This must be the FIRST step of every pipeline: the datamodule reads
    ``train_pipeline[0].key_mapper.keys()`` to decide which columns to request
    ``delta_timestamps`` for, so a source that is not named here never gets loaded.
    """

    def __init__(
        self,
        key_mapper=dict(
            head_image="head_image",
            left_image="left_image",
            right_image="right_image",
        ),
    ):
        self.key_mapper = key_mapper

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in self.key_mapper.items():
            if key not in sample:
                raise KeyError(
                    f"RefactorLerobot wants {key!r}, which the sample does not have. "
                    f"It has {sorted(sample)}. Either the dataset does not provide "
                    f"that column or it is missing from `action_sources`."
                )
            sample[value] = sample.pop(key)

        return sample


@TRANSFORMS.register_module()
class AlignFormatLerobot:
    """Give single-frame features an explicit leading time axis.

    ``[3, H, W] -> [1, 3, H, W]`` for images, ``[state_dim] -> [1, state_dim]``
    for a state vector, so everything downstream can assume a time axis.
    """

    def __init__(
        self,
        key_mapper=dict(
            head_image="head_image",
            left_image="left_image",
            right_image="right_image",
        ),
    ):
        self.key_mapper = key_mapper

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in self.key_mapper.items():
            frames = sample[key]
            if frames.ndim in (3, 1):
                frames = frames.unsqueeze(0)
            sample[value] = frames

        return sample


@TRANSFORMS.register_module()
class GetTextLerobot:
    def __init__(self, default_text="Help the robot do task by gripper."):
        self.default_text = default_text

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if sample.get("text", None) is None:
            sample["text"] = self.default_text

        return sample
