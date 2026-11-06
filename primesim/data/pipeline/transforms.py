import warnings
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from mmengine import TRANSFORMS
from scipy.spatial.transform import Rotation
from torchvision.transforms import v2

from primesim.utils.rotation_utils import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

from .loading_x_embodiment import (
    SLOT_OFFSETS,
    TOTAL_DIM,
    as_column,
    parse_wide_source,
    unpack_output,
)

ROT_TYPES = ("euler", "quat_xyzw", "quat_wxyz")

LIBERO_DELTA_XYZ_SCALE = np.asarray(
    (0.01306068, -0.01206440, -0.00979156), dtype=np.float32
)
LIBERO_DELTA_ROT_SCALE = np.asarray(
    (0.13776602, -0.09142323, -0.10814752), dtype=np.float32
)


def as_valid_quat(quat):
    """Replace all-zero quaternions with identity.

    A pose that was never recorded is logged as an all-zero quaternion; scipy
    rejects it. Same guard as ``GetActionRobocasa._as_valid_quat``.
    """
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    identity = np.zeros_like(quat)
    identity[..., 3] = 1.0

    return np.where(norm < 1e-8, identity, quat)


def as_rotation(block, rot_type):
    """A ``Rotation`` from a 3-wide euler or 4-wide quaternion block."""
    if rot_type == "euler":
        return Rotation.from_euler("xyz", block)
    if rot_type == "quat_wxyz":
        block = block[..., [1, 2, 3, 0]]  # -> xyzw, the order scipy expects

    return Rotation.from_quat(as_valid_quat(block))


def abs_to_relative_eef(action, state, rot_type="euler"):
    """The pose at t+1 relative to the pose at t, as xyz(3) + quat xyzw(4).

    ``action`` and ``state`` are ``[T, 3 + rot_width]`` blocks of absolute poses
    already sliced so columns 0:3 are xyz. The translation is expressed in the
    frame of ``state``, matching every other transform in this module.
    """
    a = action.detach().cpu().numpy()
    s = state.detach().cpu().numpy()

    R_curr = as_rotation(a[..., 3:], rot_type)
    R_prev_inv = as_rotation(s[..., 3:], rot_type).inv()

    rel_xyz = R_prev_inv.apply(a[..., :3] - s[..., :3])
    rel_quat = (R_prev_inv * R_curr).as_quat()
    rel = np.concatenate([rel_xyz, rel_quat], axis=-1)

    return torch.from_numpy(rel).to(action.dtype)


def libero_delta_to_relative_trajectory(action_7d: np.ndarray) -> np.ndarray:
    """Integrate LIBERO raw 7D action deltas into the unified action layout.

    Fills left_eef (0:7) and left_gripper (7) only -- every other slot stays zero --
    so the width comes from TOTAL_DIM and nothing here depends on where the later
    slots sit.

    This matches ``tool/worldsimprobe_libero_submit.py`` exactly so training and
    submission consume the same action semantics.
    """
    pose_xyz = np.zeros(3, dtype=np.float32)
    pose_rot = Rotation.identity()
    trajectory = np.zeros((action_7d.shape[0], TOTAL_DIM), dtype=np.float32)

    for i, step in enumerate(action_7d):
        pose_xyz += step[:3] * LIBERO_DELTA_XYZ_SCALE
        pose_rot = pose_rot * Rotation.from_rotvec(step[3:6] * LIBERO_DELTA_ROT_SCALE)
        trajectory[i, 0:3] = pose_xyz
        trajectory[i, 3:7] = pose_rot.as_quat().astype(np.float32)
        trajectory[i, 7] = step[6]

    return trajectory


def sample_relative_trajectory_for_model(
    trajectory: np.ndarray,
    action_len: int,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    times = (np.arange(action_len, dtype=np.float64) + 1.0) / target_fps
    indices = np.rint(times * source_fps - 1.0).astype(np.int64)
    indices = np.clip(indices, 0, trajectory.shape[0] - 1)
    return trajectory[indices]


@TRANSFORMS.register_module()
class TransformPrompt:
    def __init__(self, key_mapper, default_prompt):
        self.prompt_source = list(key_mapper[0].keys())[0]
        self.prompt_out = list(key_mapper[0].values())[0]
        self.default_prompt = default_prompt

    def __call__(self, data):
        if self.prompt_source not in data:
            data[self.prompt_out] = self.default_prompt
        else:
            data[self.prompt_out] = data.pop(self.prompt_source)
        return data


@TRANSFORMS.register_module()
class RemovePrompt:
    def __init__(
        self,
        key_mapper,
    ):
        self.prompt_source = list(key_mapper[0].keys())[0]
        self.prompt_out = list(key_mapper[0].values())[0]

    def __call__(self, data):
        if self.prompt_source not in data:
            data[self.prompt_out] = ""
        else:
            data[self.prompt_out] = ""
        return data


@TRANSFORMS.register_module()
class ResizeImages:
    """Resize every named view to ``height`` x ``width``.

    Input and output are ``[T, C, H, W]``; ``v2.Resize`` treats the leading axis as
    batch. Aspect ratio is NOT preserved -- a 480x640 frame is squashed, not padded.

    ``key_mapper`` is ``{source: dest}`` like every other transform here. The
    earlier version iterated the dict, which yields keys, and wrote the result back
    under the source name -- so a non-identity mapping silently did nothing.
    """

    def __init__(self, height, width, key_mapper):
        self.height = height
        self.width = width
        self.key_mapper = key_mapper
        self.transform = v2.Resize([height, width])

    def __call__(self, sample):
        for key, value in self.key_mapper.items():
            sample[value] = self.transform(sample[key])

        return sample


@TRANSFORMS.register_module()
class CropResizeImages:
    """Resize every named view to ``height`` x ``width`` *without* squashing it.

    Scales the frame so it covers the target, then centre-crops the overflow. Unlike
    :class:`ResizeImages` this preserves angles and shapes: a 45-degree gripper motion
    stays at 45 degrees, and a circle stays a circle, whatever the source aspect. That
    matters here because the model is action-conditioned -- under a squash the same eef
    delta maps to a different image-space *direction* in every dataset, since each
    source aspect implies its own horizontal compression factor.

    The cost is field of view: a 16:9 source loses 25% of its width to a 4:3 target.
    That is usually side margin on an exterior camera, so it is the right trade for
    sources *wider* than the target. For square or portrait sources the crop comes out
    of height instead -- where the arm and the table are -- so those keep
    :class:`ResizeImages` (see ``config/dataset/midtrain_datasets``: libero, mimicgen
    and rt1 are 1:1 and stay squashed).

    Input and output are ``[T, C, H, W]``; ``v2`` treats the leading axis as batch.
    ``key_mapper`` is ``{source: dest}``, as in :class:`ResizeImages`.

    ``max_discard`` bounds how much of an axis a crop may throw away; past it the view
    falls back to a squash. This matters because how bad a crop is depends on the
    *target* aspect, which lives in the launch config, not here: cropping a 16:9 source
    to 4:3 costs 25% of its width, but cropping it to a square 224x224 costs 44% -- far
    worse than squashing. ``config/launch/wsp_rollout.yaml`` does target a square
    (WorldSimProbe's ground truth is 256x256 LIBERO) while composing the same dataset
    mix, so without this guard flipping these configs to crop would quietly gut the
    field of view there.
    """

    def __init__(self, height, width, key_mapper, max_discard=0.3):
        self.height = height
        self.width = width
        self.key_mapper = key_mapper
        self.max_discard = max_discard
        self.crop = v2.CenterCrop([height, width])
        self._warned = set()

    def _cover_size(self, h, w):
        """Smallest aspect-preserving size that still covers ``height`` x ``width``."""
        scale = max(self.height / h, self.width / w)
        # round() can land a pixel short of the target and make CenterCrop pad instead
        # of crop, so clamp up to the target on both axes.
        return [max(self.height, round(h * scale)), max(self.width, round(w * scale))]

    def _discarded(self, h, w):
        """Largest fraction of either axis this crop would remove."""
        src, tgt = w / h, self.width / self.height
        return 1.0 - (tgt / src if src > tgt else src / tgt)

    def __call__(self, sample):
        for key, value in self.key_mapper.items():
            frames = sample[key]
            h, w = frames.shape[-2], frames.shape[-1]
            if self._discarded(h, w) > self.max_discard:
                if (h, w) not in self._warned:
                    self._warned.add((h, w))
                    warnings.warn(
                        f"CropResizeImages: cropping {h}x{w} (aspect {w / h:.3f}) to "
                        f"{self.height}x{self.width} "
                        f"(aspect {self.width / self.height:.3f}) "
                        f"would discard {self._discarded(h, w):.0%} of an axis, over "
                        f"max_discard={self.max_discard:.0%}; squashing this view "
                        f"instead. Set a target aspect closer to the source, or raise "
                        f"max_discard if the loss is acceptable."
                    )
                sample[value] = v2.functional.resize(
                    frames, [self.height, self.width], antialias=True
                )
                continue
            resized = v2.functional.resize(
                frames, self._cover_size(h, w), antialias=True
            )
            sample[value] = self.crop(resized)

        return sample


@TRANSFORMS.register_module()
class ImgNormalize:
    def __init__(
        self,
        key_mapper=None,
    ):
        self.max_value = 1
        self.min_value = -1
        self.key_mapper = key_mapper

    def __call__(self, sample):
        for key, value in self.key_mapper.items():
            sample[value] = (
                sample.pop(key) * (self.max_value - self.min_value) + self.min_value
            )

        return sample


@TRANSFORMS.register_module()
class ConcatenateMultiview:
    def __init__(self, key_mapper=None):
        self.cameras = eval(list(key_mapper[0].keys())[0])
        self.output_key = list(key_mapper[0].values())[0]

    def __call__(self, sample):
        sample[self.output_key] = torch.stack([sample.pop(key) for key in self.cameras])
        sample[self.output_key] = rearrange(
            sample[self.output_key], "v t c h w -> v c t h w"
        )

        return sample


@TRANSFORMS.register_module()
class FillMissingViews:
    """Synthesise the camera views a dataset does not record.

    The model takes a fixed number of views (3), but some datasets ship fewer --
    bridge and rt1 have one camera, libero and mimicgen two. A key cannot simply
    be listed twice in ``RefactorLerobot`` / ``AlignFormatLerobot`` because both
    read it under one name, so the duplicates are materialised here instead. Runs
    after ``RefactorLerobot`` has renamed the real cameras.

    Keys that already exist are left alone, so only the gaps are filled.
    """

    def __init__(self, source: str, keys: List[str], mode: str = "repeat"):
        if mode not in ("repeat", "zeros"):
            raise ValueError(f"mode must be 'repeat' or 'zeros', got {mode!r}")
        self.source = source
        self.keys = list(keys)
        self.mode = mode

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.source not in sample:
            raise KeyError(
                f"FillMissingViews needs {self.source!r} in the sample; it has "
                f"{sorted(k for k in sample if 'img' in k or 'image' in k)}. It must "
                f"run after RefactorLerobot has renamed the camera."
            )
        view = sample[self.source]
        for key in self.keys:
            if key in sample:
                continue
            sample[key] = (
                view.clone() if self.mode == "repeat" else torch.zeros_like(view)
            )

        return sample


@TRANSFORMS.register_module()
class Gather:
    """Drop everything the model does not consume. Keys that are absent are skipped."""

    def __init__(self, keys: List[str]):
        self.keys = list(keys)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {key: sample[key] for key in self.keys if key in sample}


@TRANSFORMS.register_module()
class RemapGripperAction:
    """Linearly remap one action column between gripper conventions.

    ``source_open`` / ``source_closed`` describe the input convention and
    ``target_open`` / ``target_closed`` describe the desired convention.  When
    the sample carries a matching entry in ``stats``, its mean, standard
    deviation, extrema, and quantiles are remapped as well.  The nested stats
    dictionary is copied so the dataset's shared metadata is never mutated.
    """

    def __init__(
        self,
        key_mapper,
        source_open,
        source_closed,
        target_open,
        target_closed,
        gripper_index=-1,
        clip=True,
    ):
        if len(key_mapper) != 1:
            raise ValueError(
                "key_mapper must contain exactly one source/destination pair, got "
                f"{key_mapper}"
            )
        self.source, self.output = next(iter(key_mapper.items()))
        self.source_open = float(source_open)
        self.source_closed = float(source_closed)
        self.target_open = float(target_open)
        self.target_closed = float(target_closed)
        self.gripper_index = int(gripper_index)
        self.clip = bool(clip)

        source_span = self.source_closed - self.source_open
        if source_span == 0:
            raise ValueError("source_open and source_closed must be different")
        self.scale = (self.target_closed - self.target_open) / source_span
        self.offset = self.target_open - self.scale * self.source_open

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.source not in sample:
            raise KeyError(
                f"RemapGripperAction wants {self.source!r}, but the sample has "
                f"{sorted(sample)}"
            )

        action = sample[self.source]
        width = action.shape[-1]
        if not -width <= self.gripper_index < width:
            raise ValueError(
                f"{self.source!r} has width {action.shape[-1]}, so gripper_index "
                f"{self.gripper_index} is out of range"
            )

        if isinstance(action, torch.Tensor):
            output = action.clone()
            gripper = output[..., self.gripper_index] * self.scale + self.offset
            if self.clip:
                lo, hi = sorted((self.target_open, self.target_closed))
                gripper = gripper.clamp(lo, hi)
        else:
            output = np.array(action, copy=True)
            gripper = output[..., self.gripper_index] * self.scale + self.offset
            if self.clip:
                lo, hi = sorted((self.target_open, self.target_closed))
                gripper = np.clip(gripper, lo, hi)
        output[..., self.gripper_index] = gripper
        sample[self.output] = output

        if "stats" in sample and self.source in sample["stats"]:
            self._remap_stats(sample)
        return sample

    def _remap_stats(self, sample: Dict[str, Any]) -> None:
        stats = dict(sample["stats"])
        source_stats = stats[self.source]
        output_stats = dict(source_stats)

        def row(name):
            return np.asarray(source_stats[name], dtype=np.float64).copy()

        if "mean" in source_stats:
            values = row("mean")
            values[self.gripper_index] = (
                values[self.gripper_index] * self.scale + self.offset
            )
            output_stats["mean"] = values.tolist()
        if "std" in source_stats:
            values = row("std")
            values[self.gripper_index] *= abs(self.scale)
            output_stats["std"] = values.tolist()

        for lower_name, upper_name in (("min", "max"), ("q01", "q99")):
            if lower_name not in source_stats or upper_name not in source_stats:
                continue
            lower, upper = row(lower_name), row(upper_name)
            mapped_lower = lower[self.gripper_index] * self.scale + self.offset
            mapped_upper = upper[self.gripper_index] * self.scale + self.offset
            lower[self.gripper_index] = min(mapped_lower, mapped_upper)
            upper[self.gripper_index] = max(mapped_lower, mapped_upper)
            output_stats[lower_name] = lower.tolist()
            output_stats[upper_name] = upper.tolist()

        stats[self.output] = output_stats
        if self.output != self.source:
            stats.pop(self.source)
        sample["stats"] = stats


@TRANSFORMS.register_module()
class LiberoDeltaToRelativeAction:
    """Convert raw LIBERO 7D action deltas to the unified relative trajectory layout."""

    def __init__(
        self,
        key_mapper,
        action_len,
        source_fps=None,
        target_fps=None,
        slot_cols=None,
        source_cols=None,
    ):
        self.key_mapper = key_mapper
        self.action_len = int(action_len)
        self.source_fps = None if source_fps is None else float(source_fps)
        self.target_fps = None if target_fps is None else float(target_fps)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        action_source = data[source]
        if isinstance(action_source, torch.Tensor):
            action_7d = as_column(action_source)
            action_np = action_7d.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            action_np = np.asarray(as_column(action_source), dtype=np.float32)
            action_7d = torch.as_tensor(action_np)

        pad_key = "action_is_pad"
        if pad_key in data:
            is_pad = (
                torch.as_tensor(data[pad_key])
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
                .reshape(-1)
            )
            if is_pad.shape[0] != action_np.shape[0]:
                raise ValueError(
                    f"'{pad_key}' has {is_pad.shape[0]} frames but '{source}' has "
                    f"{action_np.shape[0]} frames."
                )
            if is_pad.any():
                action_np = action_np.copy()
                action_np[is_pad, :6] = 0.0

        if np.any(action_np[:, 6] < 0.0):
            action_np = action_np.copy()
            action_np[:, 6] = (action_np[:, 6] < 0.0).astype(np.float32)

        if action_7d.shape[0] != self.action_len and (
            self.source_fps is None or self.target_fps is None
        ):
            raise ValueError(
                f"'{source}' has {action_7d.shape[0]} frames but the action needs "
                f"{self.action_len}. LIBERO raw actions are converted one-to-one "
                f"into the model's {TOTAL_DIM}D action layout unless source_fps and "
                f"target_fps are set."
            )

        traj_np = libero_delta_to_relative_trajectory(action_np)
        if traj_np.shape[0] != self.action_len:
            traj_np = sample_relative_trajectory_for_model(
                traj_np,
                self.action_len,
                self.source_fps,
                self.target_fps,
            )

        traj = torch.from_numpy(traj_np).to(
            dtype=action_7d.dtype, device=action_7d.device
        )

        data[out] = traj
        if mask_out is not None:
            action_mask = torch.zeros_like(traj, dtype=torch.bool)
            action_mask[..., 0:8] = True
            data[mask_out] = action_mask
        return data


@TRANSFORMS.register_module()
class ToRelativeAction:
    DELTA_SLOTS = ("torso", "head", "waist")

    def __init__(
        self,
        key_mapper,
        delta_slots=DELTA_SLOTS,
    ):
        self.key_mapper = key_mapper
        self.left_eef = slice(*SLOT_OFFSETS["left_eef"])
        self.right_eef = slice(*SLOT_OFFSETS["right_eef"])
        unknown = [s for s in delta_slots if s not in SLOT_OFFSETS]
        if unknown:
            raise ValueError(
                f"delta_slots names {unknown}, which are not slots. "
                f"Valid slots: {sorted(SLOT_OFFSETS)}."
            )
        # one flat index list: the slots are contiguous but need not be adjacent
        self.delta_cols = [
            col for name in delta_slots for col in range(*SLOT_OFFSETS[name])
        ]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sources = eval(list(self.key_mapper[0].keys())[0])
        out = list(self.key_mapper[0].values())[0]
        if len(sources) == 2:
            action_source, state_source = sources
            mask_source = None
        elif len(sources) == 3:
            action_source, state_source, mask_source = sources
        else:
            raise ValueError(
                f"key_mapper must name the action and state sources, and optionally "
                f"the mask -- got {len(sources)}: {sources}."
            )

        action, state = sample[action_source], sample[state_source]
        mask = sample.get(mask_source) if mask_source is not None else None
        sample[out] = self.relative_to_reference(action, state[0:1], mask)

        return sample

    def relative_to_reference(self, action, reference, mask=None):
        """Rebase absolute actions to an explicit (possibly batched) state.

        Stage 4 calls this after collation so every continuation in a sampled
        subchain keeps the opening frame as its action coordinate system.
        """
        rel_action = action.clone()
        rel_action[..., self.left_eef] = self._abs_to_relative_eef_action(
            action[..., self.left_eef], reference[..., self.left_eef]
        )
        rel_action[..., self.right_eef] = self._abs_to_relative_eef_action(
            action[..., self.right_eef], reference[..., self.right_eef]
        )
        if self.delta_cols:
            rel_action[..., self.delta_cols] = (
                action[..., self.delta_cols] - reference[..., self.delta_cols]
            )
        if mask is not None:
            rel_action = rel_action * mask.to(rel_action.dtype)
        return rel_action

    def _abs_to_relative_eef_action(self, action, state):
        curr_xyz = action[..., :3]
        prev_xyz = state[..., :3]
        # An unrecorded pose is logged as an all-zero quaternion. F.normalize
        # leaves it zero, and quaternion_to_rotation_matrix then divides by its
        # zero norm -- inf * 0 -> NaN, which would spread over the whole action.
        curr_quat = self._as_valid_quat(action[..., 3:7])
        prev_quat = self._as_valid_quat(state[..., 3:7])
        # Convert to [w, x, y, z] order for the rotation functions
        curr_quat = torch.cat([curr_quat[..., -1:], curr_quat[..., :-1]], dim=-1)
        prev_quat = torch.cat([prev_quat[..., -1:], prev_quat[..., :-1]], dim=-1)
        curr_rotm = quaternion_to_rotation_matrix(curr_quat)
        prev_rotm = quaternion_to_rotation_matrix(prev_quat)

        rel_xyz = torch.matmul(
            prev_rotm.transpose(-1, -2), (curr_xyz - prev_xyz).unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        rel_rotm = prev_rotm.transpose(-1, -2) @ curr_rotm
        rel_quat = rotation_matrix_to_quaternion(rel_rotm)
        # Convert to [x, y, z, w] order for the robot
        rel_quat = torch.cat([rel_quat[..., 1:], rel_quat[..., :1]], dim=-1)

        return torch.cat([rel_xyz, rel_quat], dim=-1)

    @staticmethod
    def _as_valid_quat(quat):
        """Unit-normalise, substituting identity (xyzw) for an all-zero quat."""
        norm = quat.norm(dim=-1, keepdim=True)
        identity = torch.zeros_like(quat)
        identity[..., 3] = 1.0

        return torch.where(norm < 1e-8, identity, F.normalize(quat, dim=-1))


@TRANSFORMS.register_module()
class ToDeltaAction:
    """Convert absolute actions into per-frame delta actions.

    Each action row t is expressed relative to the state at row t-1. The first
    row has no previous state within the window, so it uses state[0] as its
    reference and yields an identity delta; downstream losses typically mask it.
    """

    DELTA_SLOTS = ("torso", "head", "waist")

    def __init__(
        self,
        key_mapper,
        delta_slots=DELTA_SLOTS,
    ):
        self.key_mapper = key_mapper
        self.left_eef = slice(*SLOT_OFFSETS["left_eef"])
        self.right_eef = slice(*SLOT_OFFSETS["right_eef"])
        unknown = [s for s in delta_slots if s not in SLOT_OFFSETS]
        if unknown:
            raise ValueError(
                f"delta_slots names {unknown}, which are not slots. "
                f"Valid slots: {sorted(SLOT_OFFSETS)}."
            )
        self.delta_cols = [
            col for name in delta_slots for col in range(*SLOT_OFFSETS[name])
        ]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sources = eval(list(self.key_mapper[0].keys())[0])
        out = list(self.key_mapper[0].values())[0]
        if len(sources) == 2:
            action_source, state_source = sources
            mask_source = None
        elif len(sources) == 3:
            action_source, state_source, mask_source = sources
        else:
            raise ValueError(
                f"key_mapper must name the action and state sources, and optionally "
                f"the mask -- got {len(sources)}: {sources}."
            )

        action, state = sample[action_source], sample[state_source]

        # Delta: action row t is anchored to state at row t-1. Row 0 has no
        # predecessor in the window, so it falls back to state[0] and yields
        # an identity delta.
        #
        # The window holds one more state row than action row: frame i and action i
        # are the same instant, and the trailing frame has no action to drive it
        # (AblationLeRobotDataModuleV3 sets action_ts = frame_ts[:num_frames - 1]).
        # The reference is therefore cut to the action length -- building it from the
        # state length instead leaves the two disagreeing by exactly that row.
        steps = action.shape[0]
        if state.shape[0] < steps:
            raise ValueError(
                f"ToDeltaAction anchors each action row to the preceding state row, "
                f"so it needs at least as many state rows as action rows -- got "
                f"action {steps} and state {state.shape[0]}."
            )
        reference = torch.cat([state[0:1], state[: steps - 1]], dim=0)
        delta_action = action.clone()

        delta_action[..., self.left_eef] = self._abs_to_delta_eef_action(
            action[..., self.left_eef], reference[..., self.left_eef]
        )
        delta_action[..., self.right_eef] = self._abs_to_delta_eef_action(
            action[..., self.right_eef], reference[..., self.right_eef]
        )
        if self.delta_cols:
            delta_action[..., self.delta_cols] = (
                action[..., self.delta_cols] - reference[..., self.delta_cols]
            )
        if mask_source is not None and mask_source in sample:
            delta_action = delta_action * sample[mask_source].to(delta_action.dtype)

        sample[out] = delta_action

        return sample

    def _abs_to_delta_eef_action(self, action, state):
        curr_xyz = action[..., :3]
        prev_xyz = state[..., :3]
        # An unrecorded pose is logged as an all-zero quaternion. F.normalize
        # leaves it zero, and quaternion_to_rotation_matrix then divides by its
        # zero norm -- inf * 0 -> NaN, which would spread over the whole action.
        curr_quat = self._as_valid_quat(action[..., 3:7])
        prev_quat = self._as_valid_quat(state[..., 3:7])
        # Convert to [w, x, y, z] order for the rotation functions
        curr_quat = torch.cat([curr_quat[..., -1:], curr_quat[..., :-1]], dim=-1)
        prev_quat = torch.cat([prev_quat[..., -1:], prev_quat[..., :-1]], dim=-1)
        curr_rotm = quaternion_to_rotation_matrix(curr_quat)
        prev_rotm = quaternion_to_rotation_matrix(prev_quat)

        delta_xyz = torch.matmul(
            prev_rotm.transpose(-1, -2), (curr_xyz - prev_xyz).unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        delta_rotm = prev_rotm.transpose(-1, -2) @ curr_rotm
        delta_quat = rotation_matrix_to_quaternion(delta_rotm)
        # Convert to [x, y, z, w] order for the robot
        delta_quat = torch.cat([delta_quat[..., 1:], delta_quat[..., :1]], dim=-1)

        return torch.cat([delta_xyz, delta_quat], dim=-1)

    @staticmethod
    def _as_valid_quat(quat):
        """Unit-normalise, substituting identity (xyzw) for an all-zero quat."""
        norm = quat.norm(dim=-1, keepdim=True)
        identity = torch.zeros_like(quat)
        identity[..., 3] = 1.0

        return torch.where(norm < 1e-8, identity, F.normalize(quat, dim=-1))


ROBOTWIN_HOME_POSE = (
    # left_eef: xyz + quat xyzw, URDF root (footprint) frame, grasp centre
    0.4207490683,
    0.2996136248,
    0.9418964982,
    -7.346161965e-06,
    -1.495680735e-08,
    9.999833070e-03,
    0.9999499917,
    1.0,  # left_gripper (open)
    # right_eef
    0.4217657149,
    -0.3055889606,
    0.9408964515,
    -7.346145139e-06,
    2.177403680e-08,
    4.999979399e-03,
    0.9999874830,
    1.0,  # right_gripper (open)
    0.0,
    0.0,
    0.0,
    0.0,  # torso   -- unfilled by RoboTwin
    0.0,
    0.0,
    0.0,  # head    -- unfilled
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,  # waist   -- unfilled
    0.0,
    0.0,
    0.0,  # chassis -- unfilled
)  # length is checked against TOTAL_DIM below


@TRANSFORMS.register_module()
class BenchmarkToRelativeAction(ToRelativeAction):
    TOTAL_DIM = max(hi for _, hi in SLOT_OFFSETS.values())

    def __init__(
        self,
        key_mapper,
        delta_slots=ToRelativeAction.DELTA_SLOTS,
        home_pose=ROBOTWIN_HOME_POSE,
        prev_key="prev_action",
        require_prev=False,
    ):
        super().__init__(key_mapper, delta_slots=delta_slots)
        self.prev_key = prev_key
        self.require_prev = require_prev
        self.home_pose = None
        if home_pose is not None:
            pose = torch.as_tensor(np.asarray(home_pose, dtype=np.float32))
            if pose.numel() != self.TOTAL_DIM:
                raise ValueError(
                    f"home_pose has {pose.numel()} values, expected {self.TOTAL_DIM} "
                    f"(the slot layout is {dict(SLOT_OFFSETS)})."
                )
            self.home_pose = pose.reshape(1, self.TOTAL_DIM)
            self._check_quats(self.home_pose)

    def _check_quats(self, pose):
        for name in ("left_eef", "right_eef"):
            quat = pose[..., slice(*SLOT_OFFSETS[name])][..., 3:7]
            norm = float(quat.norm(dim=-1).max())
            if norm > 1e-8 and abs(norm - 1.0) > 1e-3:
                raise ValueError(
                    f"home_pose {name} quaternion has norm {norm:.6f}, not 1. These "
                    f"slots are xyzw (scalar-last); check the source convention."
                )

    def _reference(self, sample, action):
        prev = sample.get(self.prev_key) if self.prev_key else None
        if prev is not None:
            ref = torch.as_tensor(prev).to(dtype=action.dtype, device=action.device)
            ref = ref.reshape(-1, action.shape[-1])  # (N, A)
            if action.ndim == 2:
                if ref.shape[0] != 1:
                    raise ValueError(
                        f"'{self.prev_key}' holds {ref.shape[0]} rows but the action "
                        f"window is unbatched [T, A]; pass a single row."
                    )
                return ref
            batch = action.shape[0]
            if ref.shape[0] == 1:
                ref = ref.expand(batch, -1)
            elif ref.shape[0] != batch:
                raise ValueError(
                    f"'{self.prev_key}' holds {ref.shape[0]} rows but the action "
                    f"window has batch {batch}; pass one row per batch element."
                )
            return ref.unsqueeze(1)  # (B, 1, A)
        if self.require_prev:
            raise KeyError(
                f"{type(self).__name__}: no '{self.prev_key}' in the sample and "
                f"require_prev=True. A window that does not start at episode frame 0 "
                f"must be given the preceding action row as its reference."
            )
        if self.home_pose is None:
            raise ValueError(
                f"{type(self).__name__}: no '{self.prev_key}' in the sample and no "
                f"home_pose configured, so there is no reference to make the action "
                f"relative to. Pass one of the two."
            )
        return self.home_pose.to(dtype=action.dtype, device=action.device)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        action_source = eval(list(self.key_mapper[0].keys())[0])[0]
        action = sample[action_source]
        reference = self._reference(sample, action)  # broadcasts over the action rows
        rel_action = action.clone()

        rel_action[..., self.left_eef] = self._abs_to_relative_eef_action(
            action[..., self.left_eef], reference[..., self.left_eef]
        )
        rel_action[..., self.right_eef] = self._abs_to_relative_eef_action(
            action[..., self.right_eef], reference[..., self.right_eef]
        )
        if self.delta_cols:
            rel_action[..., self.delta_cols] = (
                action[..., self.delta_cols] - reference[..., self.delta_cols]
            )

        return rel_action
