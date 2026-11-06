
import numpy as np
import torch
from mmengine import TRANSFORMS
from scipy.spatial.transform import Rotation

from primesim.utils.rotation_utils import (
    rotation_6d_to_matrix,
    rotation_matrix_to_quaternion,
)

"""X-embodiment unified state/action layout -- 31 dims, identical for every robot.

    slice     width  contents
    [ 0: 7]     7    left  eef pose   xyz(3) + quat(4)
    [ 7: 8]     1    left  gripper
    [ 8:15]     7    right eef pose   xyz(3) + quat(4)
    [15:16]     1    right gripper
    [16:20]     4    torso
    [20:23]     3    head  (or neck)
    [23:28]     5    waist (or body)
    [28:31]     3    chassis  planar base velocity (vx, vy, yaw rate)
                --
                31

Arm joints are deliberately absent: they are arm-specific (6-DOF vs 7-DOF, different
link lengths and zero conventions), so the same column means something different per
embodiment and never aligned across the mix. The eef pose carries the transferable
part, and every dataset now supplies one.

head was 2 wide and waist 3 until AgiBotWorld2026's g2a arrived with 3 head and 5 waist
joints, which meant discarding two real dims per frame. Widening them shifted waist and
chassis, so the midtrain checkpoint's action projections had to be rewritten column by
column to follow -- see scripts/expand_action_dim.py, which is disposable and exists
only for that one migration. A dataset with fewer joints than a slot is wide leaves the
tail unwritten and the validity mask marks those columns invalid, so no other dataset
needed changing.
"""


SLOT_OFFSETS = {
    "left_eef": (0, 7),
    "left_gripper": (7, 8),
    "right_eef": (8, 15),
    "right_gripper": (15, 16),
    "torso": (16, 20),
    "head": (20, 23),
    "waist": (23, 28),
    "chassis": (28, 31),
}

TOTAL_DIM = max(hi for _, hi in SLOT_OFFSETS.values())


def quat_wxyz_to_xyzw(quat):
    """(..., 4) wxyz -> xyzw. The slots are xyzw everywhere; this dataset is wxyz."""
    return torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)


def _check_layout():
    """The slots must tile [0, TOTAL_DIM) exactly -- no gap, no overlap."""
    cursor = 0
    for name, (lo, hi) in SLOT_OFFSETS.items():
        if lo != cursor:
            raise ValueError(
                f"SLOT_OFFSETS is not contiguous at {name!r}: expected lo={cursor}, "
                f"got {lo}"
            )
        if hi <= lo:
            raise ValueError(f"SLOT_OFFSETS[{name!r}] is empty or inverted: {(lo, hi)}")
        cursor = hi
    if cursor != TOTAL_DIM:
        raise ValueError(f"slots cover {cursor} dims but TOTAL_DIM is {TOTAL_DIM}")


_check_layout()

SLOT_WIDTHS = {name: hi - lo for name, (lo, hi) in SLOT_OFFSETS.items()}


def get_pose_slots(traj_len):
    return {name: torch.zeros((traj_len, w)) for name, w in SLOT_WIDTHS.items()}


def build_mask_from_cols(traj_len, filled):
    mask = torch.zeros((traj_len, TOTAL_DIM), dtype=torch.bool)
    for name, cols in filled.items():
        lo, hi = SLOT_OFFSETS[name]
        for col in cols:
            if not 0 <= col < hi - lo:
                raise ValueError(
                    f"column {col} is outside the {name!r} slot, which is "
                    f"{hi - lo} wide ({lo}:{hi} of {TOTAL_DIM})"
                )
            mask[:, lo + col] = True
    return mask


def unpack_output(output):
    if isinstance(output, (tuple, list)):
        return output[0], (output[1] if len(output) > 1 else None)
    if isinstance(output, str) and output.strip().startswith("("):
        parsed = eval(output)
        return parsed[0], (parsed[1] if len(parsed) > 1 else None)
    return output, None


def unpack_sources(sources):
    if len(sources) == 3:
        return sources[0], sources[1], sources[2]
    if len(sources) == 2:
        return sources[0], sources[1], sources[1]
    raise ValueError(
        f"key_mapper must name 2 or 3 sources (eef, aux[, gripper]), got {sources}"
    )


def parse_wide_source(key, owner=""):
    if isinstance(key, (tuple, list)):
        sources = tuple(key)
    elif isinstance(key, str) and key.strip().startswith("("):
        parsed = eval(key)
        sources = (parsed,) if isinstance(parsed, str) else tuple(parsed)
    else:
        return key
    if len(sources) != 1:
        raise ValueError(
            f"{owner or 'this transform'} slices every slot out of ONE wide column, "
            f"so key_mapper names a single source; got {len(sources)}: {sources}."
        )
    return sources[0]


def resolve_source_cols(defaults, overrides, owner):
    cols = {**defaults, **{name: list(c) for name, c in (overrides or {}).items()}}
    unknown = [name for name in cols if name not in SLOT_OFFSETS]
    if unknown:
        raise ValueError(
            f"{owner}.source_cols names {unknown}, which are not slots of the "
            f"{TOTAL_DIM}-dim layout; the slots are {list(SLOT_OFFSETS)}."
        )
    return {name: list(cols[name]) for name in SLOT_OFFSETS if name in cols}


def as_column(block):
    return block[..., None] if block.ndim == 1 else block


def eef_rotvec_to_quat(eef_pos):
    """pos(3) + axis-angle(3) -> pos(3) + quat(4), xyzw."""
    xyz = eef_pos[..., :3].detach().cpu().numpy()
    rotvec = eef_pos[..., 3:].detach().cpu().numpy()
    quat = Rotation.from_rotvec(rotvec).as_quat()
    output = np.concatenate([xyz, quat], axis=-1)

    return torch.from_numpy(output).to(dtype=eef_pos.dtype, device=eef_pos.device)


def eef_euler_to_quat(eef_pos):
    """pos(3) + euler(3) -> pos(3) + quat(4), xyzw."""
    xyz = eef_pos[..., :3].detach().cpu().numpy()
    euler = eef_pos[..., 3:].detach().cpu().numpy()
    quat = Rotation.from_euler("xyz", euler).as_quat()
    output = np.concatenate([xyz, quat], axis=-1)

    return torch.from_numpy(output).to(dtype=eef_pos.dtype, device=eef_pos.device)


def eef_rot6d_to_quat(eef_pos):
    """pos(3) + rot6d(6) -> pos(3) + quat(4), xyzw.

    `rotation_6d_to_matrix` does the Gram-Schmidt and stacks the result with dim=-2,
    i.e. it reads the 6D block as the first two ROWS of R -- which is what HiFi-UMI's
    info.json declares (rotation_6d_layout: first_two_rows). That matters because the
    alternative reading, the first two columns, is the transpose and so the inverse
    rotation, and nothing downstream would distinguish them: both give det(R) = +1,
    unit quaternions and smooth trajectories. The convention lives in that one helper
    rather than being restated here.

    Unlike its axis-angle and euler siblings above this stays in torch throughout, so
    there is no numpy round-trip and no device transfer. rotation_matrix_to_quaternion
    returns the real part first, hence quat_wxyz_to_xyzw -- the slots are xyzw.
    """
    xyz, d6 = eef_pos[..., :3], eef_pos[..., 3:9]
    quat = quat_wxyz_to_xyzw(rotation_matrix_to_quaternion(rotation_6d_to_matrix(d6)))

    return torch.cat([xyz, quat.to(dtype=eef_pos.dtype)], dim=-1)


class SlotWriter:
    slot_cols = {}

    def _write(self, slots, slot_name, block, source=""):
        lo, hi = SLOT_OFFSETS[slot_name]
        width = block.shape[-1]
        cols = self.slot_cols.get(slot_name, list(range(width)))
        named = f"'{source}'" if source else f"the {slot_name!r} source"

        if len(cols) != width:
            raise ValueError(
                f"slot_cols[{slot_name!r}] names {len(cols)} columns ({cols}) but "
                f"{named} supplies {width} dims -- they must correspond one to one."
            )
        if width > hi - lo:
            raise ValueError(
                f"{named} supplies {width} dims for the {slot_name!r} slot, which is "
                f"only {hi - lo} wide ({lo}:{hi} of {TOTAL_DIM}). Reduce it in the "
                f"config or widen the slot in SLOT_OFFSETS."
            )
        bad = [col for col in cols if not 0 <= col < hi - lo]
        if bad:
            raise ValueError(
                f"slot_cols[{slot_name!r}] names column(s) {bad} outside the slot, "
                f"which is {hi - lo} wide ({lo}:{hi} of {TOTAL_DIM})."
            )
        if block.ndim != 2:
            raise ValueError(
                f"{named} has shape {tuple(block.shape)}; the {slot_name!r} slot "
                f"needs (frames, dims). An index list that came through as None "
                f"adds an axis here."
            )
        expected = slots[slot_name].shape[0]
        if block.shape[0] != expected:
            raise ValueError(
                f"{named} has {block.shape[0]} frames but this transform expects "
                f"{expected}. Either the feature is missing from the dataset's "
                "`action_sources` (so the datamodule only gave it state_len frames), "
                "or a "
                f"row offset does not match the window: an action transform passed "
                "state_len while `action_from_state` is off slices one row short, "
                "giving "
                f"{expected - 1} where {expected} is wanted."
            )
        slots[slot_name][..., cols] = block.to(slots[slot_name].dtype)

        return cols

    @staticmethod
    def _split_arms(feature, width, source):
        if feature.shape[-1] == width and feature.ndim >= 2 and feature.shape[-2] == 2:
            return feature[..., 0, :], feature[..., 1, :]
        if feature.shape[-1] == 2 * width:
            return feature[..., :width], feature[..., width:]
        raise ValueError(
            f"'{source}' has trailing shape {tuple(feature.shape[-2:])}, which is "
            f"neither a nested (2, {width}) pair of arms nor a flat {2 * width}."
        )


class WideVectorWriter(SlotWriter):
    """Base for a robot whose whole state (or action) is one wide column.

    The per-feature transforms below zip a tuple of sources against a tuple of
    slots; here there is a single source and `source_cols` says which of its columns
    each slot takes, so the subclass only has to declare SOURCE_COLS.
    """

    SOURCE_COLS = {}

    def _init(self, key_mapper, traj_len, slot_cols, source_cols):
        self.key_mapper = key_mapper
        self.traj_len = traj_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )
        # Row window into the source. Every dataset but one reads the whole thing; a
        # dataset whose action is its own shifted state hands both transforms the SAME
        # feature over one long window, and each narrows this to its half.
        self.rows = slice(None)

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = data[source][self.rows]

        needed = max(col for cols in self.source_cols.values() for col in cols) + 1
        if vector.shape[-1] < needed:
            raise ValueError(
                f"'{source}' is {vector.shape[-1]} wide but source_cols reads up to "
                f"column {needed - 1}. The whole layout is column indices into this "
                f"one feature, so a schema change invalidates all of them."
            )

        filled = {}
        slots = get_pose_slots(self.traj_len)
        for slot_name, cols in self.source_cols.items():
            filled[slot_name] = self._write(slots, slot_name, vector[..., cols], source)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.traj_len, filled)

        return data


class _X2WWideWriter(WideVectorWriter):
    """Shared X2W handling on top of WideVectorWriter's plain column copy.

    Two slots need more than a copy. The eef blocks may store their quaternion wxyz
    while every slot is xyzw, so ``quat_order`` says which. And the chassis slot wants
    a planar base velocity (vx, vy, yaw rate) that this robot does not record: it
    records three wheel angular velocities in rad/s. Those are a different physical
    quantity in different units, so copying them straight in would make columns 25:28
    mean something here that they mean nowhere else in the mix -- the exact
    misalignment the unified layout exists to prevent. The conversion is therefore
    opt-in and requires the two geometry constants it cannot be done without. Until
    those are supplied the chassis slot stays zero AND unfilled, so
    ``build_mask_from_cols`` marks it absent rather than asserting a stationary base.
    """

    EEF_SLOTS = ("left_eef", "right_eef")

    def _init_x2w(self, *, quat_order, wheel_cols, wheel_radius, track_width):
        self.quat_order = str(quat_order).lower()
        if self.quat_order not in ("xyzw", "wxyz"):
            raise ValueError(
                f"{type(self).__name__}.quat_order must be 'xyzw' or 'wxyz', got "
                f"{quat_order!r}."
            )
        self.wheel_cols = list(wheel_cols) if wheel_cols else []
        if self.wheel_cols and len(self.wheel_cols) < 2:
            raise ValueError(
                f"{type(self).__name__} needs the front-left and front-right wheel "
                f"columns to recover a base velocity, got {self.wheel_cols}."
            )
        self.wheel_radius = None if wheel_radius is None else float(wheel_radius)
        self.track_width = None if track_width is None else float(track_width)
        if (self.wheel_radius is None) != (self.track_width is None):
            raise ValueError(
                f"{type(self).__name__} needs wheel_radius AND track_width together: "
                "vx scales with the radius, and the yaw rate divides by the track."
            )
        if self.track_width is not None and self.track_width <= 0:
            raise ValueError(f"{type(self).__name__}.track_width must be positive.")

    @property
    def chassis_from_wheels(self):
        return bool(self.wheel_cols) and self.wheel_radius is not None

    def _wheels_to_planar(self, vector):
        """Differential-drive forward kinematics from the two front wheels.

            vx  = r (w_left + w_right) / 2
            vy  = 0                        -- non-holonomic, no sideways motion
            yaw = r (w_right - w_left) / track_width

        The rear wheel is deliberately unused: on a two-front-driven base it follows
        the motion rather than setting it, so averaging it into vx double-counts.
        """
        w_left = vector[..., self.wheel_cols[0]].float()
        w_right = vector[..., self.wheel_cols[1]].float()
        vx = self.wheel_radius * (w_left + w_right) / 2.0
        yaw = self.wheel_radius * (w_right - w_left) / self.track_width
        return torch.stack([vx, torch.zeros_like(vx), yaw], dim=-1)

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = data[source][self.rows]

        read_cols = [col for cols in self.source_cols.values() for col in cols]
        if self.chassis_from_wheels:
            read_cols = read_cols + self.wheel_cols[:2]
        needed = max(read_cols) + 1
        if vector.shape[-1] < needed:
            raise ValueError(
                f"'{source}' is {vector.shape[-1]} wide but this transform reads up to "
                f"column {needed - 1}. The whole layout is column indices into this "
                "one "
                f"feature, so a schema change invalidates all of them."
            )

        filled = {}
        slots = get_pose_slots(self.traj_len)
        for slot_name, cols in self.source_cols.items():
            block = vector[..., cols]
            if slot_name in self.EEF_SLOTS and self.quat_order == "wxyz":
                block = torch.cat(
                    [block[..., :3], quat_wxyz_to_xyzw(block[..., 3:])], dim=-1
                )
            filled[slot_name] = self._write(slots, slot_name, block, source)

        if self.chassis_from_wheels:
            filled["chassis"] = self._write(
                slots, "chassis", self._wheels_to_planar(vector), source
            )

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.traj_len, filled)

        return data


@TRANSFORMS.register_module()
class GetStateX2W(_X2WWideWriter):
    """X2W: the 89-wide state column sliced into the unified layout.

        slice     contents                            -> slot
        [ 0: 3]   waist                                  waist
        [ 3]      torso                                  torso (col 16 only)
        [ 4: 6]   head                                   head
        [20]      left  gripper                          left_gripper
        [21]      right gripper                          right_gripper
        [66:73]   left  eef pose, xyz + quat             left_eef
        [73:80]   right eef pose, xyz + quat             right_eef
        [83:86]   wheel angular velocity, rad/s          chassis, only once
                  (front left, front right, rear)        wheel_radius and track_width
                                                         are supplied -- see
                                                         _X2WWideWriter

    The 14 dims at 66:80 are two 7-wide eef blocks, so they carry a quaternion rather
    than euler angles and go in unconverted; set ``quat_order: wxyz`` if this repo_id
    stores the scalar part first. Torso is a single DOF against a 4-wide slot, so it
    fills column 16 alone and the mask leaves 17:20 false.
    """

    SOURCE_COLS = {
        "left_eef": list(range(66, 73)),
        "left_gripper": [20],
        "right_eef": list(range(73, 80)),
        "right_gripper": [21],
        "torso": [3],
        "head": [4, 5],
        "waist": [0, 1, 2],
    }

    def __init__(
        self,
        key_mapper,
        state_len,
        slot_cols=None,
        source_cols=None,
        quat_order="xyzw",
        wheel_cols=(83, 84, 85),
        wheel_radius=None,
        track_width=None,
    ) -> None:
        self._init(key_mapper, state_len, slot_cols, source_cols)
        self._init_x2w(
            quat_order=quat_order,
            wheel_cols=wheel_cols,
            wheel_radius=wheel_radius,
            track_width=track_width,
        )


@TRANSFORMS.register_module()
class GetActionX2W(_X2WWideWriter):
    """The same robot's 89-wide `action` column, same layout as the state.

    The commanded and achieved vectors share a schema on this robot, so the column map
    is inherited from GetStateX2W rather than restated -- a schema change has one
    place to be fixed.
    """

    SOURCE_COLS = GetStateX2W.SOURCE_COLS

    def __init__(
        self,
        key_mapper,
        action_len,
        slot_cols=None,
        source_cols=None,
        quat_order="xyzw",
        wheel_cols=(83, 84, 85),
        wheel_radius=None,
        track_width=None,
    ) -> None:
        self._init(key_mapper, action_len, slot_cols, source_cols)
        self._init_x2w(
            quat_order=quat_order,
            wheel_cols=wheel_cols,
            wheel_radius=wheel_radius,
            track_width=track_width,
        )


@TRANSFORMS.register_module()
class GetState2026Challenge(WideVectorWriter):
    """BEHAVIOR-1K 2026 Challenge (R1Pro in OmniGibson): the 61-dim
    `observation.state` column sliced into the unified layout.

        slice     contents                        -> slot
        [ 0: 3]   measured base planar velocity      dropped, see below
        [10:17]   left  arm joint velocity           dropped
        [17:24]   left  eef pose, xyz + quat wzyx    left_eef
        [24:26]   left  gripper finger positions     left_gripper (col 24)
        [26:28]   left  gripper finger velocities    dropped
        [35:42]   right arm joint velocity           dropped
        [42:49]   right eef pose, xyz + quat wzyx    right_eef
        [49:51]   right gripper finger positions     right_gripper (col 49)
        [51:53]   right gripper finger velocities    dropped
        [53:57]   trunk position                     torso
        [57:61]   trunk velocity                     dropped

    The feature carries no per-column names, so the layout was read off the data:
    cols 3:10 / 28:35 track the arm commands in `action` at corr 0.9-1.0.

    The eef quaternions are stored REVERSED, (w, z, y, x), so source_cols reads them
    back to front to land xyzw in the slot. A unit-norm check cannot tell component
    orders apart; this was settled against URDF FK (urdf_r1pro_2025, base_link ->
    <arm>_gripper_link): R_ref * R_fk^-1 drifts 0.0001 deg over 3000 frames read as
    wzyx, versus 114 / 164 deg (left / right) read as xyzw. Reading xyzw is not a
    constant frame offset -- it conjugates every relative rotation's axis.

    The layout has no velocity slots, so the five velocity blocks are dropped. The
    two fingers of a gripper move together (corr 1.00 left, 0.92 right), so the
    1-wide gripper slot takes the first of each pair. Base velocity is dropped here,
    as in GetStateAgiBotWorld: it correlates poorly with the commanded velocity in
    `action` (-0.03, 0.13, 0.50), so the two are most likely in different frames and
    do not belong in one slot. The 4-DOF trunk fills torso, which is 4 wide; `waist`
    is only 3.
    """

    SOURCE_COLS = {
        "left_eef": [17, 18, 19, 23, 22, 21, 20],
        "left_gripper": [24],
        "right_eef": [42, 43, 44, 48, 47, 46, 45],
        "right_gripper": [49],
        "torso": list(range(53, 57)),
    }

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None) -> None:
        self._init(key_mapper, state_len, slot_cols, source_cols)


@TRANSFORMS.register_module()
class GetAction2026Challenge(WideVectorWriter):
    """The same robot's 23-dim `action` column sliced into the unified layout.

    slice     contents                        -> slot
    [ 0: 3]   base planar velocity command       chassis
    [ 3: 7]   trunk command                      torso
    [14]      left  gripper command, +-1         left_gripper
    [22]      right gripper command, +-1         right_gripper

    """

    SOURCE_COLS = {
        "left_gripper": [14],
        "right_gripper": [22],
        "torso": list(range(3, 7)),
        "chassis": [0, 1, 2],
    }

    def __init__(
        self, key_mapper, action_len, slot_cols=None, source_cols=None
    ) -> None:
        self._init(key_mapper, action_len, slot_cols, source_cols)


@TRANSFORMS.register_module()
class GetAction2026ChallengeFromState(WideVectorWriter):
    """The action read out of `observation.state`, one frame ahead of the state.

    The 23-dim `action` column that GetAction2026Challenge maps carries joint
    commands only -- no eef pose -- so a dataset that does record eef in its state
    contributes nothing to the eef slots on the action side. Reading the action out
    of the 61-dim state instead gives the same eef layout for both, at the cost of
    using the achieved next pose rather than the commanded one.

    Same shape as GetActionBridge / GetActionKai0: the source column carries the
    whole frame window and `state_len` is a ROW OFFSET into it, so `state_len: 1`
    with `action_len: N` selects frames 1..N -- the pose reached by each transition.
    Requires `action_from_state: true` on the dataset so the column keeps the full
    window instead of being narrowed to the action rows.
    """

    SOURCE_COLS = GetState2026Challenge.SOURCE_COLS

    def __init__(
        self, key_mapper, action_len, state_len=0, slot_cols=None, source_cols=None
    ) -> None:
        self._init(key_mapper, action_len, slot_cols, source_cols)
        self.rows = slice(state_len, state_len + action_len)


@TRANSFORMS.register_module()
class GetStateDROID(SlotWriter):
    SOURCE_ORDER = (
        "left eef state",
        "left gripper state",
    )
    SLOT_ORDER = (
        "left_eef",
        "left_gripper",
    )

    def __init__(
        self,
        key_mapper,
        state_len,
        slot_cols=None,
        source_cols=None,
    ) -> None:
        self.key_mapper = key_mapper
        self.state_len = state_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            name: list(cols) for name, cols in (source_cols or {}).items()
        }

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.state_len)
        for slot_name, source in zip(self.SLOT_ORDER, sources):
            if source is None:
                continue
            block = as_column(data[source])
            source_cols = self.source_cols.get(slot_name)
            if source_cols is not None:
                block = block[..., source_cols]
            if slot_name == "left_eef":
                block = eef_euler_to_quat(block)
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)

        return data


@TRANSFORMS.register_module()
class GetActionDROID(SlotWriter):
    SOURCE_ORDER = (
        "left eef action",
        "left gripper action",
    )
    SLOT_ORDER = (
        "left_eef",
        "left_gripper",
    )

    def __init__(
        self,
        key_mapper,
        action_len,
        slot_cols=None,
        source_cols=None,
    ) -> None:
        self.key_mapper = key_mapper
        self.action_len = action_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            name: list(cols) for name, cols in (source_cols or {}).items()
        }

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.action_len)
        for slot_name, source in zip(self.SLOT_ORDER, sources):
            if source is None:
                continue
            block = as_column(data[source])
            source_cols = self.source_cols.get(slot_name)
            if source_cols is not None:
                block = block[..., source_cols]
            if slot_name == "left_eef":
                block = eef_euler_to_quat(block)
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)

        return data


@TRANSFORMS.register_module()
class GetStateRobomind(SlotWriter):
    SOURCE_ORDER = (
        "left eef state",
        "left gripper state",
        "right eef state",
        "right gripper state",
    )
    SLOT_ORDER = (
        "left_eef",
        "left_gripper",
        "right_eef",
        "right_gripper",
    )

    def __init__(
        self,
        key_mapper,
        state_len,
        slot_cols=None,
        source_cols=None,
    ) -> None:
        self.key_mapper = key_mapper
        self.state_len = state_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            name: list(cols) for name, cols in (source_cols or {}).items()
        }

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.state_len)
        for slot_name, source in zip(self.SLOT_ORDER, sources):
            if source is None:
                continue
            block = as_column(data[source])
            source_cols = self.source_cols.get(slot_name)
            if source_cols is not None:
                block = block[..., source_cols]
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)

        return data


@TRANSFORMS.register_module()
class GetActionRobomind(SlotWriter):
    SOURCE_ORDER = (
        "left eef action",
        "left gripper action",
        "right eef action",
        "right gripper action",
        "chassis",
    )
    SLOT_ORDER = (
        "left_eef",
        "left_gripper",
        "right_eef",
        "right_gripper",
        "chassis",
    )
    SOURCE_COLS = {"chassis": [0, 1, 5]}

    def __init__(
        self,
        key_mapper,
        action_len,
        slot_cols=None,
        source_cols=None,
    ) -> None:
        self.key_mapper = key_mapper
        self.action_len = action_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) not in (len(self.SOURCE_ORDER) - 1, len(self.SOURCE_ORDER)):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER) - 1} or "
                f"{len(self.SOURCE_ORDER)} sources ({', '.join(self.SOURCE_ORDER)}), "
                f"the chassis being optional; got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.action_len)
        for slot_name, source in zip(self.SLOT_ORDER, sources):
            if source is None:
                continue
            # as_column before the column selection: a gripper is one value per frame,
            # so it
            # arrives as (frames,), and indexing that with source_cols would slice the
            # frame
            # axis instead of a column axis. Matters here because SOURCE_COLS is non-
            # empty.
            block = as_column(data[source])
            source_cols = self.source_cols.get(slot_name)
            if source_cols is not None:
                block = block[..., source_cols]
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)

        return data


@TRANSFORMS.register_module()
class GetStateAgiBotWorld(SlotWriter):
    SOURCE_ORDER = ("eef position", "eef orientation", "head", "waist", "gripper")

    def __init__(
        self,
        key_mapper,
        state_len,
        slot_cols=None,
    ):
        self.key_mapper = key_mapper
        self.state_len = state_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.state_len)
        slots, filled = self._collect_arm_features(slots, sources, data, filled)
        slots, filled = self._collect_body_features(slots, sources, data, filled)
        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)
        return data

    def _collect_arm_features(self, slots, sources, data, filled):
        pos_source, quat_source, _, _, grip_source = sources

        eef_quat = data[quat_source]
        left_pos, right_pos = self._split_arms(data[pos_source], 3, pos_source)
        left_quat, right_quat = self._split_arms(eef_quat, 4, quat_source)
        for slot_name, pos, quat in (
            ("left_eef", left_pos, left_quat),
            ("right_eef", right_pos, right_quat),
        ):
            if self._unrecorded_rotation(quat):
                continue
            filled[slot_name] = self._write(
                slots, slot_name, torch.cat([pos, quat], dim=-1), pos_source
            )

        left_grip, right_grip = self._split_arms(data[grip_source], 1, grip_source)
        filled["left_gripper"] = self._write(
            slots, "left_gripper", left_grip, grip_source
        )
        filled["right_gripper"] = self._write(
            slots, "right_gripper", right_grip, grip_source
        )

        return slots, filled

    def _collect_body_features(self, slots, sources, data, filled):
        # One feature per part, written left-aligned unless slot_cols says otherwise.
        _, _, head_source, waist_source, _ = sources
        for slot_name, source in (("head", head_source), ("waist", waist_source)):
            filled[slot_name] = self._write(slots, slot_name, data[source], source)

        return slots, filled

    @staticmethod
    def _unrecorded_rotation(quat):
        return bool((torch.linalg.norm(quat, dim=-1) < 1e-8).any())


@TRANSFORMS.register_module()
class GetActionAgiBotWorld(SlotWriter):
    def __init__(
        self,
        key_mapper,
        action_len,
        slot_cols=None,
    ):
        self.key_mapper = key_mapper
        self.action_len = action_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    SOURCE_ORDER = (
        "eef position",
        "eef orientation",
        "head",
        "waist",
        "chassis",
        "gripper",
    )

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.action_len)
        slots, filled = self._collect_arm_features(slots, sources, data, filled)
        slots, filled = self._collect_body_features(slots, sources, data, filled)
        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)
        return data

    def _collect_arm_features(self, slots, sources, data, filled):
        pos_source, quat_source, _, _, _, grip_source = sources

        eef_quat = data[quat_source]
        left_pos, right_pos = self._split_arms(data[pos_source], 3, pos_source)
        left_quat, right_quat = self._split_arms(eef_quat, 4, quat_source)
        for slot_name, pos, quat in (
            ("left_eef", left_pos, left_quat),
            ("right_eef", right_pos, right_quat),
        ):
            if self._unrecorded_rotation(quat):
                continue
            filled[slot_name] = self._write(
                slots, slot_name, torch.cat([pos, quat], dim=-1), pos_source
            )

        left_grip, right_grip = self._split_arms(data[grip_source], 1, grip_source)
        filled["left_gripper"] = self._write(
            slots, "left_gripper", left_grip, grip_source
        )
        filled["right_gripper"] = self._write(
            slots, "right_gripper", right_grip, grip_source
        )

        return slots, filled

    def _collect_body_features(self, slots, sources, data, filled):
        # One feature per part, written left-aligned unless slot_cols says otherwise.
        _, _, head_source, waist_source, chassis_source, _ = sources
        for slot_name, source in (
            ("head", head_source),
            ("waist", waist_source),
            ("chassis", chassis_source),
        ):
            filled[slot_name] = self._write(slots, slot_name, data[source], source)

        return slots, filled

    @staticmethod
    def _unrecorded_rotation(quat):
        """True if any frame's quaternion is all-zero, i.e. never recorded.

        Whole-chunk granularity: one bad frame drops this arm's eef for the whole
        chunk. That is deliberate -- a zero quaternion means the recording lacked
        eef tracking, which holds for an entire episode rather than odd frames. If
        partially-recorded episodes ever turn up, the mask is per-frame capable and
        this could tighten to one row at a time.
        """
        return bool((torch.linalg.norm(quat, dim=-1) < 1e-8).any())


@TRANSFORMS.register_module()
class GetStateAgiBotWorld2026(WideVectorWriter):
    """AgiBotWorld2026 (g2a): the wide `observation.state` column into the layout.

    The beta drop above records one nested feature per body part, so its transform
    takes five sources and splits each in half. 2026 instead packs everything into a
    single `observation.state` vector, which is what WideVectorWriter is for, and the
    per-slot column lists carry the split.

    SOURCE_COLS is empty on purpose: the state layout is NOT the same in every shard.
    Widths run 162/169/193/216 and the field indices move with them, so
    `field_descriptions` in each shard's info.json is the only authority on where a
    field sits. Three distinct layouts exist across the 334 shards, hence three
    column maps, hence the map comes from the config. resolve_source_cols merges
    defaults with overrides, so a default here would leak into any slot a config
    leaves unnamed -- which for a shifted layout means reading the wrong columns
    rather than reading none, the failure that is hardest to notice downstream.

    Two details the config maps rely on. The position and orientation blocks are not
    adjacent and orientation comes first in the vector, but _write copies columns in
    the order named, so one list per eef slot holds xyz then quat with no shuffling
    step. And the quaternions are already xyzw, so they go in unconverted as the beta
    drop's do -- see the yamls for the evidence, since no field carries dim names.
    """

    SOURCE_COLS = {}

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None) -> None:
        self._init(key_mapper, state_len, slot_cols, source_cols)


@TRANSFORMS.register_module()
class GetActionAgiBotWorld2026(WideVectorWriter):
    """The same robot's wide `action` column into the unified layout.

    Unlike the state, `action` is laid out identically in all 334 shards, so one map
    serves every config. It is still supplied from the config rather than defaulted
    here, for the same leak reason as the state side, and because the chassis slot is
    conditional: `action/robot/velocity` is 6 wide in 245 shards and 2 wide in the
    other 89, where it is identically zero. Those 89 name no chassis, leaving columns
    25:28 masked invalid instead of writing a zero that would read as "the base is
    commanded to hold still".

    The action is an absolute target pose in the same frame and units as the state,
    so it pairs with ToRelativeAction downstream.
    """

    SOURCE_COLS = {}

    def __init__(
        self, key_mapper, action_len, slot_cols=None, source_cols=None
    ) -> None:
        self._init(key_mapper, action_len, slot_cols, source_cols)


@TRANSFORMS.register_module()
class GetStateRoboCoin(SlotWriter):
    def __init__(
        self,
        key_mapper,
        state_len,
        left_pos,
        left_euler,
        left_grip,
        right_pos,
        right_euler,
        right_grip,
        torso=None,
        head=None,
        waist=None,
        chassis=None,
        slot_cols=None,
    ):
        self.key_mapper = key_mapper
        self.left_pos = list(left_pos)
        self.left_euler = list(left_euler)
        self.left_grip = int(left_grip)
        self.right_pos = list(right_pos)
        self.right_euler = list(right_euler)
        self.right_grip = int(right_grip)

        self.torso = list(torso) if torso is not None else None
        self.head = list(head) if head is not None else None
        self.waist = list(waist) if waist is not None else None
        self.chassis = list(chassis) if chassis is not None else None
        self.state_len = state_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        state_sources = list(self.key_mapper[0].keys())[0]
        sources = eval(state_sources)
        state_source, aux_source, grip_source = unpack_sources(sources)
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])

        state = data[state_source]
        aux = data[aux_source]
        grip = data[grip_source]
        pose_slots = get_pose_slots(self.state_len)
        filled = {}

        left_eef = state[..., self.left_pos + self.left_euler]
        left_eef = eef_euler_to_quat(left_eef)
        right_eef = state[..., self.right_pos + self.right_euler]
        right_eef = eef_euler_to_quat(right_eef)
        filled["left_eef"] = self._write(pose_slots, "left_eef", left_eef, state_source)
        filled["right_eef"] = self._write(
            pose_slots, "right_eef", right_eef, state_source
        )

        # gripper_open_scale_* is float64 on some repo_ids and float32 on others --
        # mixed even within one config -- so cast or default collate breaks.
        for part_name, col in (
            ("left_gripper", self.left_grip),
            ("right_gripper", self.right_grip),
        ):
            block = grip[..., col][..., None].float()
            filled[part_name] = self._write(pose_slots, part_name, block, grip_source)

        # each arm independently: one arm missing its joint list must not drop the other
        for part_name, idx in (
            ("torso", self.torso),
            ("head", self.head),
            ("waist", self.waist),
            ("chassis", self.chassis),
        ):
            if not idx:
                continue
            filled[part_name] = self._write(
                pose_slots, part_name, aux[..., idx], aux_source
            )

        data[state_out] = torch.cat([pose_slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)
        return data


@TRANSFORMS.register_module()
class GetActionRoboCoin(SlotWriter):
    def __init__(
        self,
        key_mapper,
        action_len,
        left_pos,
        left_euler,
        left_grip,
        right_pos,
        right_euler,
        right_grip,
        torso=None,
        head=None,
        waist=None,
        chassis=None,
        slot_cols=None,
    ):
        self.key_mapper = key_mapper
        self.left_pos = list(left_pos)
        self.left_euler = list(left_euler)
        self.left_grip = int(left_grip)
        self.right_pos = list(right_pos)
        self.right_euler = list(right_euler)
        self.right_grip = int(right_grip)

        self.torso = list(torso) if torso is not None else None
        self.head = list(head) if head is not None else None
        self.waist = list(waist) if waist is not None else None
        self.chassis = list(chassis) if chassis is not None else None
        self.action_len = action_len
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        action_sources = list(self.key_mapper[0].keys())[0]
        sources = eval(action_sources)
        action_source, aux_source, grip_source = unpack_sources(sources)
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])

        action = data[action_source]
        aux = data[aux_source]
        grip = data[grip_source]
        pose_slots = get_pose_slots(self.action_len)
        filled = {}

        left_eef = action[..., self.left_pos + self.left_euler]
        left_eef = eef_euler_to_quat(left_eef)
        right_eef = action[..., self.right_pos + self.right_euler]
        right_eef = eef_euler_to_quat(right_eef)
        filled["left_eef"] = self._write(
            pose_slots, "left_eef", left_eef, action_source
        )
        filled["right_eef"] = self._write(
            pose_slots, "right_eef", right_eef, action_source
        )

        # gripper_open_scale_* is float64 on some repo_ids and float32 on others --
        # mixed even within one config -- so cast or default collate breaks.
        for part_name, col in (
            ("left_gripper", self.left_grip),
            ("right_gripper", self.right_grip),
        ):
            block = grip[..., col][..., None].float()
            filled[part_name] = self._write(pose_slots, part_name, block, grip_source)

        # each arm independently: one arm missing its joint list must not drop the other
        for part_name, idx in (
            ("torso", self.torso),
            ("head", self.head),
            ("waist", self.waist),
            ("chassis", self.chassis),
        ):
            if not idx:
                continue
            filled[part_name] = self._write(
                pose_slots, part_name, aux[..., idx], aux_source
            )

        data[action_out] = torch.cat(
            [pose_slots[name] for name in SLOT_OFFSETS], dim=-1
        )
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)
        return data


@TRANSFORMS.register_module()
class GetStateRobocasa(SlotWriter):
    EEF_COLS = list(range(7, 14))
    GRIPPER_COL = 14

    def __init__(self, key_mapper, state_len, slot_cols=None) -> None:
        # This used to insist on state_len == 1. That guard existed because the
        # datamodule gave observation.state a PAST-facing window while this dataset
        # lists it in action_sources, so anything past frame 0 was future data
        # masquerading as history. The unified clock made every window forward-facing
        # from the sampled index, so all `state_len` rows are legitimate state now --
        # and holding this dataset at 1 row while the other 34 give num_frames broke
        # batch collation.
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.rows = slice(0, self.state_len)

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = data[source]

        needed = max(list(self.EEF_COLS) + [self.GRIPPER_COL]) + 1
        if vector.shape[-1] < needed:
            raise ValueError(
                f"'{source}' is {vector.shape[-1]} wide but this layout reads up to "
                f"column {needed - 1}. RoboCasa's observation.state is 16 wide; "
                f"another width means the schema changed and every index here is "
                f"invalidated."
            )

        current = vector[self.rows]
        if current.shape[0] != self.state_len:
            raise ValueError(
                f"'{source}' has {vector.shape[0]} frames, so rows {self.rows} yield "
                f"{current.shape[0]}, but state_len is {self.state_len}. This dataset "
                f"needs `action_from_state: true` so the column keeps the whole frame "
                f"window rather than being narrowed to the action rows."
            )
        slots = get_pose_slots(self.state_len)
        filled = {
            "left_eef": self._write(
                slots, "left_eef", current[..., self.EEF_COLS], source
            ),
            "left_gripper": self._write(
                slots,
                "left_gripper",
                current[..., self.GRIPPER_COL : self.GRIPPER_COL + 1],
                source,
            ),
        }

        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)

        return data


@TRANSFORMS.register_module()
class GetActionRobocasa(SlotWriter):
    SOURCE_ORDER = ("state vector, for the achieved eef pose", "action vector")
    EEF_COLS = list(range(7, 14))  # into the STATE vector, not the action
    GRIPPER_COL = 11  # into the action vector

    def __init__(self, key_mapper, action_len, state_len=0, slot_cols=None) -> None:
        self.key_mapper = key_mapper
        self.action_len = int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        # `state_len` is a ROW OFFSET, not a length, as in GetActionBridge: both source
        # columns carry the whole frame window, and offset 1 selects frames 1..N -- the
        # pose reached by each transition.
        self.rows = slice(int(state_len), int(state_len) + self.action_len)

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        state_source, action_source = sources
        state_vector = data[state_source][self.rows]
        action_vector = data[action_source][self.rows]

        for name, vector, needed in (
            (state_source, state_vector, max(self.EEF_COLS) + 1),
            (action_source, action_vector, self.GRIPPER_COL + 1),
        ):
            if vector.shape[-1] < needed:
                raise ValueError(
                    f"'{name}' is {vector.shape[-1]} wide but this layout reads up to "
                    f"column {needed - 1}. RoboCasa is 16 wide on the state and 12 on "
                    f"the action; another width means the schema changed and every "
                    f"index here is invalidated."
                )
            if vector.shape[0] != self.action_len:
                raise ValueError(
                    f"'{name}' yields {vector.shape[0]} frames over rows {self.rows} "
                    f"but the action needs {self.action_len}. Both columns this "
                    "transform reads -- observation.state included, since the eef "
                    "pose "
                    f"comes from it -- must be in `action_sources`, and the dataset "
                    f"needs `action_from_state: true` so they keep the whole frame "
                    f"window for these rows to exist."
                )

        slots = get_pose_slots(self.action_len)
        filled = {
            "left_eef": self._write(
                slots, "left_eef", state_vector[..., self.EEF_COLS], state_source
            ),
            "left_gripper": self._write(
                slots,
                "left_gripper",
                action_vector[..., self.GRIPPER_COL][..., None],
                action_source,
            ),
        }

        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)

        return data


@TRANSFORMS.register_module()
class GetActionInternDataM1(SlotWriter):
    SOURCE_ORDER = (
        "eef action",
        "gripper",
    )
    SOURCE_COLS = {
        "left_eef": list(range(7)),
        "right_eef": list(range(7, 14)),
        "left_gripper": [
            0,
        ],
        "right_gripper": [
            1,
        ],
    }

    def __init__(self, key_mapper, action_len, slot_cols=None):
        self.key_mapper = key_mapper
        self.action_len = int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.action_len)
        slots, filled = self._collect_arm_features(slots, sources, data, filled)

        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)
        return data

    def _collect_arm_features(self, slots, sources, data, filled):
        eef_source, grip_source = sources

        for name, vector in (
            (eef_source, data[eef_source]),
            (grip_source, data[grip_source]),
        ):
            if vector.shape[0] != self.action_len:
                raise ValueError(
                    f"'{name}' has {vector.shape[0]} frames but the action needs "
                    f"{self.action_len}. Every feature the action slots read must be "
                    f"listed in the dataset's `action_sources`, or the datamodule "
                    f"leaves it at state_len frames."
                )

        left_eef, right_eef = self._split_arms(data[eef_source], 7, eef_source)
        left_eef = torch.cat(
            [left_eef[..., 0:3], quat_wxyz_to_xyzw(left_eef[..., 3:7])], dim=-1
        )
        filled["left_eef"] = self._write(slots, "left_eef", left_eef, eef_source)
        right_eef = torch.cat(
            [right_eef[..., 0:3], quat_wxyz_to_xyzw(right_eef[..., 3:7])], dim=-1
        )
        filled["right_eef"] = self._write(slots, "right_eef", right_eef, eef_source)

        left_grip, right_grip = self._split_arms(data[grip_source], 1, grip_source)
        filled["left_gripper"] = self._write(
            slots, "left_gripper", left_grip, grip_source
        )
        filled["right_gripper"] = self._write(
            slots, "right_gripper", right_grip, grip_source
        )

        return slots, filled


@TRANSFORMS.register_module()
class GetStateInternDataM1(SlotWriter):
    SOURCE_ORDER = (
        "ee_state",
        "gripper",
    )
    SOURCE_COLS = {
        "left_eef": list(range(7)),
        "right_eef": list(range(7, 14)),
        "left_gripper": [18],
        "right_gripper": [26],
    }

    def __init__(self, key_mapper, state_len, slot_cols=None):
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.state_len)
        slots, filled = self._collect_arm_features(slots, sources, data, filled)

        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)
        return data

    def _collect_arm_features(self, slots, sources, data, filled):
        eef_source, grip_source = sources
        gripper = data[grip_source]

        left_eef, right_eef = self._split_arms(data[eef_source], 7, eef_source)
        for slot_name, block in (("left_eef", left_eef), ("right_eef", right_eef)):
            pose = torch.cat(
                [block[..., 0:3], quat_wxyz_to_xyzw(block[..., 3:7])], dim=-1
            )
            filled[slot_name] = self._write(slots, slot_name, pose, eef_source)

        for slot_name, cols in (
            ("left_gripper", self.SOURCE_COLS["left_gripper"]),
            ("right_gripper", self.SOURCE_COLS["right_gripper"]),
        ):
            filled[slot_name] = self._write(
                slots, slot_name, gripper[..., cols], grip_source
            )

        return slots, filled


@TRANSFORMS.register_module()
class GetActionInternDataM1Franka(SlotWriter):
    SOURCE_ORDER = (
        "ee_pos action",
        "ee_ori action",
        "gripper action",
    )
    SOURCE_COLS = {
        "left_eef": list(range(3)),
        "left_gripper": [0],
    }

    def __init__(self, key_mapper, action_len, slot_cols=None):
        self.key_mapper = key_mapper
        self.action_len = int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.action_len)
        slots, filled = self._collect_arm_features(slots, sources, data, filled)

        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)
        return data

    def _collect_arm_features(self, slots, sources, data, filled):
        eef_pos_source, eef_ori_source, grip_source = sources

        eef = torch.cat([data[eef_pos_source], data[eef_ori_source]], dim=-1)
        filled["left_eef"] = self._write(
            slots, "left_eef", eef_euler_to_quat(eef), eef_pos_source
        )
        filled["left_gripper"] = self._write(
            slots, "left_gripper", as_column(data[grip_source]), grip_source
        )

        return slots, filled


@TRANSFORMS.register_module()
class GetStateInternDataM1Franka(SlotWriter):
    SOURCE_ORDER = (
        "ee_pos state",
        "ee_ori state",
        "gripper state",
    )
    SOURCE_COLS = {
        "left_eef": list(range(3)),
        "left_gripper": [0],
    }

    def __init__(self, key_mapper, state_len, slot_cols=None):
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.state_len)
        slots, filled = self._collect_arm_features(slots, sources, data, filled)

        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)

        return data

    def _collect_arm_features(self, slots, sources, data, filled):
        eef_pos_source, eef_ori_source, grip_source = sources

        eef = torch.cat([data[eef_pos_source], data[eef_ori_source]], dim=-1)
        filled["left_eef"] = self._write(
            slots, "left_eef", eef_euler_to_quat(eef), eef_pos_source
        )
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            as_column(data[grip_source])[..., self.SOURCE_COLS["left_gripper"]],
            grip_source,
        )

        return slots, filled


@TRANSFORMS.register_module()
class GetActionInternDataM1Robotiq(SlotWriter):
    SOURCE_ORDER = (
        "eef_position action",
        "eef_quaternion action",
        "gripper action",
    )
    SOURCE_COLS = {
        "left_eef": list(range(3)),
        "left_gripper": [0],
    }

    def __init__(self, key_mapper, action_len, slot_cols=None):
        self.key_mapper = key_mapper
        self.action_len = int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.action_len)
        slots, filled = self._collect_arm_features(slots, sources, data, filled)

        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)
        return data

    def _collect_arm_features(self, slots, sources, data, filled):
        eef_pos_source, eef_quat_source, grip_source = sources

        eef = torch.cat(
            [data[eef_pos_source], quat_wxyz_to_xyzw(data[eef_quat_source])], dim=-1
        )
        filled["left_eef"] = self._write(slots, "left_eef", eef, eef_pos_source)
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            as_column(data[grip_source])[..., self.SOURCE_COLS["left_gripper"]],
            grip_source,
        )

        return slots, filled


@TRANSFORMS.register_module()
class GetStateInternDataM1Robotiq(SlotWriter):
    SOURCE_ORDER = (
        "eef_position state",
        "eef_quaternion state",
        "gripper state",
    )
    SOURCE_COLS = {
        "left_eef": list(range(3)),
        "left_gripper": [0],
    }

    def __init__(self, key_mapper, state_len, slot_cols=None):
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.state_len)
        slots, filled = self._collect_arm_features(slots, sources, data, filled)

        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)
        return data

    def _collect_arm_features(self, slots, sources, data, filled):
        eef_pos_source, eef_quat_source, grip_source = sources

        eef = torch.cat(
            [data[eef_pos_source], quat_wxyz_to_xyzw(data[eef_quat_source])], dim=-1
        )
        filled["left_eef"] = self._write(slots, "left_eef", eef, eef_pos_source)
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            as_column(data[grip_source])[..., self.SOURCE_COLS["left_gripper"]],
            grip_source,
        )

        return slots, filled


@TRANSFORMS.register_module()
class GetActionGalaxeaOpenWorld(SlotWriter):
    SOURCE_ORDER = (
        "left eef pose",
        "right eef pose",
        "left gripper",
        "right gripper",
        "torso",
        "chassis",
    )
    SLOT_ORDER = (
        "left_eef",
        "right_eef",
        "left_gripper",
        "right_gripper",
        "torso",
        "chassis",
    )
    SOURCE_COLS = {"chassis": [0, 1, 5]}

    def __init__(
        self, key_mapper, action_len, slot_cols=None, source_cols=None, row_offsets=None
    ) -> None:
        self.key_mapper = key_mapper
        self.action_len = int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }
        # source -> first row. A state column spans the whole frame window, so offset 1
        # takes the pose reached by each transition while the other sources stay on the
        # action window.
        self.row_offsets = {name: int(off) for name, off in (row_offsets or {}).items()}

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        unknown = sorted(set(self.row_offsets) - {s for s in sources if s is not None})
        if unknown:
            raise ValueError(
                f"row_offsets names {unknown}, which are not among this transform's "
                f"sources {sources}."
            )
        filled = {}
        slots = get_pose_slots(self.action_len)
        for slot_name, source in zip(self.SLOT_ORDER, sources):
            if source is None:
                continue
            block = as_column(data[source])
            if source in self.row_offsets:
                start = self.row_offsets[source]
                block = block[start : start + self.action_len]
            source_cols = self.source_cols.get(slot_name)
            if source_cols is not None:
                block = block[..., source_cols]
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)

        return data


@TRANSFORMS.register_module()
class GetStateGalaxeaOpenWorld(SlotWriter):
    SOURCE_ORDER = (
        "left eef pose",
        "right eef pose",
        "left gripper",
        "right gripper",
        "torso",
        "chassis",
    )
    SLOT_ORDER = (
        "left_eef",
        "right_eef",
        "left_gripper",
        "right_gripper",
        "torso",
        "chassis",
    )
    SOURCE_COLS = {}

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None) -> None:
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.state_len)
        for slot_name, source in zip(self.SLOT_ORDER, sources):
            if source is None:  # a part this repo_id does not record
                continue
            block = as_column(data[source])
            source_cols = self.source_cols.get(slot_name)
            if source_cols is not None:
                block = block[..., source_cols]
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)

        return data


@TRANSFORMS.register_module()
class GetStateHyEmbodied(WideVectorWriter):
    """Hy-Embodied-0.5-VLA: the 16-dim `observation.state` column, two arms of eight.

    slice     contents                      -> slot
    [ 0: 3]   left  eef xyz                    left_eef  cols 0:3
    [ 3: 7]   left  eef quaternion, xyzw       left_eef  cols 3:7
    [ 7: 8]   left  gripper, 0..90             left_gripper
    [ 8:11]   right eef xyz                    right_eef cols 0:3
    [11:15]   right eef quaternion, xyzw       right_eef cols 3:7
    [15:16]   right gripper, 0..90             right_gripper

    """

    SOURCE_COLS = {
        "left_eef": list(range(0, 7)),
        "left_gripper": [7],
        "right_eef": list(range(8, 15)),
        "right_gripper": [15],
    }

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None) -> None:
        self._init(key_mapper, state_len, slot_cols, source_cols)
        self.rows = slice(0, state_len)


@TRANSFORMS.register_module()
class GetActionHyEmbodied(WideVectorWriter):
    """Hy-Embodied actions, which are the state one step ahead."""

    SOURCE_COLS = GetStateHyEmbodied.SOURCE_COLS

    def __init__(
        self,
        key_mapper,
        action_len,
        state_len=0,
        pad_key=None,
        slot_cols=None,
        source_cols=None,
    ) -> None:
        self._init(key_mapper, action_len, slot_cols, source_cols)
        self.rows = slice(state_len, state_len + action_len)
        self.pad_key = pad_key

    def __call__(self, data):
        data = super().__call__(data)
        _, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if self.pad_key is None or mask_out is None or self.pad_key not in data:
            return data

        is_pad = torch.as_tensor(data[self.pad_key]).reshape(-1)[self.rows].bool()
        if is_pad.shape[0] != data[mask_out].shape[0]:
            raise ValueError(
                f"{self.pad_key!r} has {is_pad.shape[0]} rows in the action window but "
                "the "
                f"mask has {data[mask_out].shape[0]}. The pad tensor covers the whole "
                "delta_timestamps window, so it has to be sliced the same way -- "
                "check "
                f"that state_len and action_len here match the datamodule's."
            )
        data[mask_out] = data[mask_out] & ~is_pad[:, None]
        return data


@TRANSFORMS.register_module()
class GetStateBridge(SlotWriter):
    """BridgeData V2 (WidowX 250): the 7-dim `observation.state` column.

        slice     contents                    -> slot
        [0:3]     eef xyz, metres                left_eef cols 0:3
        [3:6]     eef euler rpy             left_eef cols 3:7, via eef_euler_to_quat
        [6:7]     gripper                        left_gripper

    One arm, one camera, no joints recorded: every other slot stays zero and masked.
    """

    SOURCE_ORDER = ("state vector, xyz + rpy + gripper",)
    SOURCE_COLS = {"left_eef": list(range(0, 6)), "left_gripper": [6]}

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None):
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }
        self.rows = slice(0, self.state_len)

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]
        filled = {}
        slots = get_pose_slots(vector.shape[0])
        filled["left_eef"] = self._write(
            slots,
            "left_eef",
            eef_euler_to_quat(vector[..., self.source_cols["left_eef"]]),
            source,
        )
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            vector[..., self.source_cols["left_gripper"]],
            source,
        )
        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)
        return data


@TRANSFORMS.register_module()
class GetActionBridge(SlotWriter):
    SOURCE_ORDER = GetStateBridge.SOURCE_ORDER
    SOURCE_COLS = GetStateBridge.SOURCE_COLS

    def __init__(
        self,
        key_mapper,
        action_len,
        state_len=0,
        pad_key=None,
        slot_cols=None,
        source_cols=None,
    ):
        self.key_mapper = key_mapper
        self.state_len, self.action_len = int(state_len), int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }
        self.rows = slice(self.state_len, self.state_len + self.action_len)
        self.pad_key = pad_key

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]
        filled = {}
        slots = get_pose_slots(vector.shape[0])
        filled["left_eef"] = self._write(
            slots,
            "left_eef",
            eef_euler_to_quat(vector[..., self.source_cols["left_eef"]]),
            source,
        )
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            vector[..., self.source_cols["left_gripper"]],
            source,
        )
        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is None:
            return data

        data[mask_out] = build_mask_from_cols(vector.shape[0], filled)
        if self.pad_key is None or self.pad_key not in data:
            return data
        is_pad = torch.as_tensor(data[self.pad_key]).reshape(-1)[self.rows].bool()
        if is_pad.shape[0] != data[mask_out].shape[0]:
            raise ValueError(
                f"{self.pad_key!r} has {is_pad.shape[0]} rows in the action window but "
                "the "
                f"mask has {data[mask_out].shape[0]}. The pad tensor spans the whole "
                f"delta_timestamps window, so it has to be sliced the same way."
            )
        data[mask_out] = data[mask_out] & ~is_pad[:, None]
        return data


# The state's cols 6:8 are padding -- zero on every one of the 7490 frames -- so the
# state carries no gripper and that slot stays unfilled. Only the action records one.
UTOKYO_XARM_STATE_COLS = {"left_eef": list(range(0, 6))}
UTOKYO_XARM_ACTION_COLS = {"left_eef": list(range(0, 6)), "left_gripper": [6]}


def eef_euler_to_quat_x_hemisphere(eef_pos):
    """eef_euler_to_quat, with every quaternion flipped onto the qx >= 0 hemisphere.

    The xArm's tool points straight down, so roll sits at +-pi and wraps between the
    two from frame to frame (7.8% of frames are negative). from_euler maps roll pi and
    -pi to q and -q -- the same rotation, but a jump of 2 in the state vector the model
    reads -- 134 times across the dataset. The usual w >= 0 canonicalisation does not
    help here: w is ~0 at this attitude and still flips 132 times. qx is the dominant
    component instead (|qx| >= 0.873 on every frame), so pinning its sign removes all
    134 flips without introducing any.
    """
    pose = eef_euler_to_quat(eef_pos)
    sign = torch.where(pose[..., 3:4] < 0, -1.0, 1.0).to(pose.dtype)

    return torch.cat([pose[..., :3], pose[..., 3:] * sign], dim=-1)


class _UTokyoXArmWriter(SlotWriter):
    """UTokyo xArm7 pick-and-place (OXE, 10 Hz): one arm, xyz + rpy, no joints.

    The LeRobot conversion kept only the Cartesian pose; OXE's joint_state did not
    survive, so there is nothing to run FK on and the pose is taken as recorded:

        slice     contents                       -> slot
        [0:3]     eef xyz, metres, base frame       left_eef cols 0:3
        [3:6]     eef roll, pitch, yaw, radians     left_eef cols 3:7, via
                  (xArm fixed-axis xyz)             eef_euler_to_quat_x_hemisphere
        [6:7]     gripper, ACTION ONLY              left_gripper

    The action is an absolute target pose in the same frame as the state, which is
    what ToRelativeAction assumes. It leads the state by more than a frame, because
    the arm lags its target and the state refreshes at only half the rate (every other
    row is a repeat), so action[t] is not state[t+1] as it is on HiFi-UMI.

    The gripper is a continuous command in [0, 1] in steps of 1/255: 0 is open and 1
    is closed -- every episode runs 0 -> 1 on the grasp and 1 -> 0 on release. It is
    copied as is, like every other dataset's here.
    """

    SOURCE_COLS = {}

    def _init(self, key_mapper, rows, slot_cols, source_cols):
        self.key_mapper = key_mapper
        self.rows = rows
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )

    def _fill(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]

        filled = {}
        slots = get_pose_slots(vector.shape[0])
        for slot_name, cols in self.source_cols.items():
            block = vector[..., cols]
            if slot_name == "left_eef":
                block = eef_euler_to_quat_x_hemisphere(block)
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)

        return data, mask_out


@TRANSFORMS.register_module()
class GetStateUTokyoXArm(_UTokyoXArmWriter):
    SOURCE_COLS = UTOKYO_XARM_STATE_COLS

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None):
        self._init(key_mapper, slice(0, int(state_len)), slot_cols, source_cols)

    def __call__(self, data):
        data, _ = self._fill(data)
        return data


@TRANSFORMS.register_module()
class GetActionUTokyoXArm(_UTokyoXArmWriter):
    """The 7-dim `action` column, read from its own window.

    `action_from_state` is false for this dataset, so the datamodule hands `action` its
    own action_len-row window and state_len stays 0. `pad_key` (lerobot's
    action_is_pad) drops the rows past an episode's end from the mask: lerobot fills
    them by repeating the last frame, which would otherwise train a hold as if it had
    been recorded.
    """

    SOURCE_COLS = UTOKYO_XARM_ACTION_COLS

    def __init__(
        self,
        key_mapper,
        action_len,
        state_len=0,
        pad_key=None,
        slot_cols=None,
        source_cols=None,
    ):
        self.state_len, self.action_len = int(state_len), int(action_len)
        self._init(
            key_mapper,
            slice(self.state_len, self.state_len + self.action_len),
            slot_cols,
            source_cols,
        )
        self.pad_key = pad_key

    def __call__(self, data):
        data, mask_out = self._fill(data)
        if mask_out is None or self.pad_key is None or self.pad_key not in data:
            return data

        is_pad = torch.as_tensor(data[self.pad_key]).reshape(-1)[self.rows].bool()
        if is_pad.shape[0] != data[mask_out].shape[0]:
            raise ValueError(
                f"{self.pad_key!r} has {is_pad.shape[0]} rows in the action window but "
                "the "
                f"mask has {data[mask_out].shape[0]}. The pad tensor spans the whole "
                f"delta_timestamps window, so it has to be sliced the same way."
            )
        data[mask_out] = data[mask_out] & ~is_pad[:, None]
        return data


# Koch reads two columns per side: the FK'd pose for the eef, and the joint vector for
# the gripper, which FK does not touch. Column indices are into each source.
KOCH_SOURCE_COLS = {"left_eef": list(range(0, 7)), "left_gripper": [5]}


def quat_onto_z_hemisphere(pose):
    """Flip each xyz + quat (xyzw) pose's quaternion onto qz >= 0.

    The FK step stores scipy's canonical form, w >= 0, and on Koch w passes through 0:
    98 frames flip to -q between neighbours, a jump of 2 in a state vector that encodes
    a near-identical rotation. qz never gets near 0 on this robot (|qz| >= 0.227 on
    every state and action frame), so pinning its sign removes all 98 without adding
    any. Measured alternatives: qx >= 0 leaves 2 action flips, qy >= 0 adds 230.
    """
    sign = torch.where(pose[..., 5:6] < 0, -1.0, 1.0).to(pose.dtype)
    return torch.cat([pose[..., :3], pose[..., 3:] * sign], dim=-1)


class _KochWriter(SlotWriter):
    """Koch 6-motor follower (koch_pick_place_1_lego_raph, 30 Hz): one arm, two cameras.

        source                                  cols   -> slot
        observation.states.eef_fk.data /        0:7       left_eef, xyz + quat xyzw,
          actions.eef_fk.data                             via quat_onto_z_hemisphere
        observation.state / action              5         left_gripper, degrees

    The eef columns are scripts/add_eef_poses_koch.py's FK output -- see its docstring
    for the geometry and how the joint signs were measured. Positions are metres in the
    arm's base frame (z up, +y forward at zero pan).

    The gripper is the raw motor angle: about -40 deg open, +5 (state) / +9 (action)
    closed, so it rises on closing like UTokyo's 0..1 and opposite to RoboTwin's.
    Copied as is, like every other dataset here.

    The action is the leader arm, a target the follower reaches ~3 native frames (one
    10 Hz model frame) later, in the same frame and units as the state -- an absolute
    target, which is what ToRelativeAction assumes.
    """

    SOURCE_ORDER = ("eef pose, xyz + quat xyzw", "joint vector, gripper at col 5")
    SOURCE_COLS = KOCH_SOURCE_COLS

    def _init(self, key_mapper, rows, slot_cols, source_cols):
        self.key_mapper = key_mapper
        self.rows = rows
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )

    def _fill(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if not isinstance(sources, tuple) or len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {sources!r}. A missing comma "
                f"between two names silently concatenates them."
            )
        eef_source, joint_source = sources
        eef = as_column(data[eef_source])[self.rows][..., self.source_cols["left_eef"]]
        grip = as_column(data[joint_source])[self.rows][
            ..., self.source_cols["left_gripper"]
        ]
        if eef.shape[0] != grip.shape[0]:
            raise ValueError(
                f"'{eef_source}' has {eef.shape[0]} rows but '{joint_source}' has "
                f"{grip.shape[0]}; both must be in the dataset's delta_timestamps with "
                "the "
                f"same window (for the action side, both in `action_sources`)."
            )

        filled = {}
        slots = get_pose_slots(eef.shape[0])
        filled["left_eef"] = self._write(
            slots, "left_eef", quat_onto_z_hemisphere(eef), eef_source
        )
        filled["left_gripper"] = self._write(slots, "left_gripper", grip, joint_source)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(eef.shape[0], filled)

        return data, mask_out


@TRANSFORMS.register_module()
class GetStateKoch(_KochWriter):
    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None):
        self._init(key_mapper, slice(0, int(state_len)), slot_cols, source_cols)

    def __call__(self, data):
        data, _ = self._fill(data)
        return data


@TRANSFORMS.register_module()
class GetActionKoch(_KochWriter):
    """The action side, read from its own window like GetActionUTokyoXArm.

    `action_from_state` is false for this dataset, so the datamodule hands both
    action columns their own action_len-row window and state_len stays 0. `pad_key`
    (lerobot's action_is_pad) drops rows past an episode's end from the mask.
    """

    def __init__(
        self,
        key_mapper,
        action_len,
        state_len=0,
        pad_key=None,
        slot_cols=None,
        source_cols=None,
    ):
        self.state_len, self.action_len = int(state_len), int(action_len)
        self._init(
            key_mapper,
            slice(self.state_len, self.state_len + self.action_len),
            slot_cols,
            source_cols,
        )
        self.pad_key = pad_key

    def __call__(self, data):
        data, mask_out = self._fill(data)
        if mask_out is None or self.pad_key is None or self.pad_key not in data:
            return data

        is_pad = torch.as_tensor(data[self.pad_key]).reshape(-1)[self.rows].bool()
        if is_pad.shape[0] != data[mask_out].shape[0]:
            raise ValueError(
                f"{self.pad_key!r} has {is_pad.shape[0]} rows in the action window but "
                "the "
                f"mask has {data[mask_out].shape[0]}. The pad tensor spans the whole "
                f"delta_timestamps window, so it has to be sliced the same way."
            )
        data[mask_out] = data[mask_out] & ~is_pad[:, None]
        return data


# Into the 7-wide observation.state: xyz, rotation vector, gripper width.
UMI_CUP_SOURCE_COLS = {"left_eef": list(range(0, 6)), "left_gripper": [6]}


class _UMICupWriter(SlotWriter):
    """UMI cup arrangement in the wild (umi_cup_in_the_wild, 10 Hz): the handheld UMI
    gripper, one wrist fisheye camera, no robot.

        cols     contents                                     -> slot
        [0:3]    gripper pose xyz, metres, SLAM map frame         left_eef cols 0:3
        [3:6]    rotation VECTOR                             left_eef cols 3:7, quat
        [6]      gripper width, metres (0.003 .. 0.091)           left_gripper

    Measured: read as a rotation vector the pose turns a median 0.22 deg per frame and
    at most 9.4; read as euler xyz it jumps up to 170, so it is a rotvec. Column 6 is
    bit-identical to the dataset's gripper_width. start_pos / end_pose are per-episode
    constants -- end_pose looks to the episode's end -- and are not read.

    There is no action column: the action is the state one row ahead, as on Bridge
    (`action_from_state: true`, `state_len: 1` on the action transform).

    from_rotvec gives the w >= 0 quaternion, which flips sign 149 times across the
    dataset; quat_continuous_in_window with anchor qx (14 flips left by qx >= 0 alone,
    the fewest of the four) makes every window flip-free.

    Poses come from UMI's SLAM, and tracking fails sometimes; the dataset config keeps
    the 13 episodes where it visibly does out of training -- see the yaml.
    """

    SOURCE_COLS = UMI_CUP_SOURCE_COLS
    QUAT_ANCHOR = 0

    def _init(self, key_mapper, rows, slot_cols, source_cols):
        self.key_mapper = key_mapper
        self.rows = rows
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )

    def _fill(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]
        filled = {}
        slots = get_pose_slots(vector.shape[0])
        eef = quat_continuous_in_window(
            eef_rotvec_to_quat(vector[..., self.source_cols["left_eef"]]),
            anchor=self.QUAT_ANCHOR,
        )
        filled["left_eef"] = self._write(slots, "left_eef", eef, source)
        filled["left_gripper"] = self._write(
            slots, "left_gripper", vector[..., self.source_cols["left_gripper"]], source
        )

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)
        return data, mask_out


@TRANSFORMS.register_module()
class GetStateUMICup(_UMICupWriter):
    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None):
        self._init(key_mapper, slice(0, int(state_len)), slot_cols, source_cols)

    def __call__(self, data):
        data, _ = self._fill(data)
        return data


@TRANSFORMS.register_module()
class GetActionUMICup(_UMICupWriter):
    """The state one row ahead; see GetActionXVLASoftFold for the row-offset
    contract."""

    def __init__(
        self,
        key_mapper,
        action_len,
        state_len=0,
        pad_key=None,
        slot_cols=None,
        source_cols=None,
    ):
        self.state_len, self.action_len = int(state_len), int(action_len)
        self._init(
            key_mapper,
            slice(self.state_len, self.state_len + self.action_len),
            slot_cols,
            source_cols,
        )
        self.pad_key = pad_key

    def __call__(self, data):
        data, mask_out = self._fill(data)
        if mask_out is None or self.pad_key is None or self.pad_key not in data:
            return data
        is_pad = torch.as_tensor(data[self.pad_key]).reshape(-1)[self.rows].bool()
        if is_pad.shape[0] != data[mask_out].shape[0]:
            raise ValueError(
                f"{self.pad_key!r} has {is_pad.shape[0]} rows in the action window but "
                "the "
                f"mask has {data[mask_out].shape[0]}. The pad tensor spans the whole "
                f"delta_timestamps window, so it has to be sliced the same way."
            )
        data[mask_out] = data[mask_out] & ~is_pad[:, None]
        return data


# ALOHA reads two 14-wide columns per side: the FK'd poses (left 0:7, right 7:14) and
# the
# joint vector (left joints 0:6 + gripper 6, right joints 7:13 + gripper 13).
ALOHA_SOURCE_COLS = {
    "left_eef": list(range(0, 7)),
    "left_gripper": [6],
    "right_eef": list(range(7, 14)),
    "right_gripper": [13],
}
ALOHA_ARM_JOINTS = {"left": list(range(0, 6)), "right": list(range(7, 13))}
ALOHA_DROPOUT_TOL = 1.0e-3


class _AlohaWriter(SlotWriter):
    """Static ALOHA, two Interbotix vx300s (aloha_static_fork_pick_up, 50 Hz).

        source                                  cols     -> slot
        observation.states.eef_fk.data /        0:7         left_eef,  xyz + quat xyzw
          actions.eef_fk.data                   7:14        right_eef, xyz + quat xyzw
        observation.state / action              6           left_gripper
                                                13          right_gripper

    The eef columns are scripts/add_eef_poses_aloha.py's FK output, each arm in its own
    base frame -- see its docstring for the URDF and how the convention was checked.
    They are copied as stored: they are scipy's w >= 0 form and |w| >= 0.44 on every
    frame of both columns, so there are no sign flips to undo.

    Grippers are ALOHA's normalised reading, 1 = open, about -0.05 .. 0.98, copied as
    is. The action is the leader arms, an absolute target in the same frames.
    """

    SOURCE_ORDER = (
        "eef poses, 2 x (xyz + quat xyzw)",
        "joint vector, grippers at 6 and 13",
    )
    SOURCE_COLS = ALOHA_SOURCE_COLS

    def _init(self, key_mapper, rows, slot_cols, source_cols):
        self.key_mapper = key_mapper
        self.rows = rows
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )

    def _fill(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if not isinstance(sources, tuple) or len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {sources!r}. A missing comma "
                f"between two names silently concatenates them."
            )
        eef_source, joint_source = sources
        eef = as_column(data[eef_source])[self.rows]
        joints = as_column(data[joint_source])[self.rows]
        if eef.shape[0] != joints.shape[0]:
            raise ValueError(
                f"'{eef_source}' has {eef.shape[0]} rows but '{joint_source}' has "
                f"{joints.shape[0]}; both must be in the dataset's delta_timestamps "
                "with the "
                f"same window (for the action side, both in `action_sources`)."
            )

        filled = {}
        slots = get_pose_slots(eef.shape[0])
        for slot_name, cols in self.source_cols.items():
            src, name = (
                (eef, eef_source)
                if slot_name.endswith("_eef")
                else (joints, joint_source)
            )
            filled[slot_name] = self._write(slots, slot_name, src[..., cols], name)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(eef.shape[0], filled)

        return data, mask_out, joints


@TRANSFORMS.register_module()
class GetStateAloha(_AlohaWriter):
    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None):
        self._init(key_mapper, slice(0, int(state_len)), slot_cols, source_cols)

    def __call__(self, data):
        data, _, _ = self._fill(data)
        return data


@TRANSFORMS.register_module()
class GetActionAloha(_AlohaWriter):
    """The leader arms, read from their own window like GetActionKoch.

    Two things clear rows from the mask. `pad_key` (lerobot's action_is_pad) drops
    rows past an episode's end. And a recording dropout -- an arm whose six action
    joints all sit at +-pi, as the left arm does at episode 41 frame 501 -- drops that
    arm's eef and gripper for the row. Its eef was held from the previous frame by the
    FK script, but its gripper reads -2.476 against a real range of -0.15 .. 0.99, so
    the row is masked rather than trained on with either value.
    """

    def __init__(
        self,
        key_mapper,
        action_len,
        state_len=0,
        pad_key=None,
        slot_cols=None,
        source_cols=None,
    ):
        self.state_len, self.action_len = int(state_len), int(action_len)
        self._init(
            key_mapper,
            slice(self.state_len, self.state_len + self.action_len),
            slot_cols,
            source_cols,
        )
        self.pad_key = pad_key

    def __call__(self, data):
        data, mask_out, joints = self._fill(data)
        if mask_out is None:
            return data

        for arm, cols in ALOHA_ARM_JOINTS.items():
            dropout = ((joints[..., cols].abs() - np.pi).abs() < ALOHA_DROPOUT_TOL).all(
                dim=-1
            )
            if dropout.any():
                for slot_name in (f"{arm}_eef", f"{arm}_gripper"):
                    lo, hi = SLOT_OFFSETS[slot_name]
                    data[mask_out][dropout, lo:hi] = False

        if self.pad_key is None or self.pad_key not in data:
            return data
        is_pad = torch.as_tensor(data[self.pad_key]).reshape(-1)[self.rows].bool()
        if is_pad.shape[0] != data[mask_out].shape[0]:
            raise ValueError(
                f"{self.pad_key!r} has {is_pad.shape[0]} rows in the action window but "
                "the "
                f"mask has {data[mask_out].shape[0]}. The pad tensor spans the whole "
                f"delta_timestamps window, so it has to be sliced the same way."
            )
        data[mask_out] = data[mask_out] & ~is_pad[:, None]
        return data


# Into the 96-wide observation.state; see _XVLASoftFoldWriter for the full layout.
XVLA_SOFT_FOLD_SOURCE_COLS = {
    "left_eef": list(range(14, 21)),
    "left_gripper": [21],
    "right_eef": list(range(22, 29)),
    "right_gripper": [29],
}


def quat_continuous_in_window(pose, anchor=1):
    """(T, 7) xyz + quat (xyzw) -> the same rotations, sign-continuous along T.

    Row 0 goes onto q[anchor] >= 0 -- anchor 0..3 = x, y, z, w; the default 1 (qy) is
    xvla-soft-fold's -- and every later row takes whichever of q / -q agrees with the
    row before it. On xvla-soft-fold the stored quaternions flip sign ~3600 times
    per arm in the first file alone. qy >= 0 by itself removes all but 27 across the
    2.85M frames (|qy| dips to 0.037 on a few extreme poses), and so does a hemisphere
    around each arm's dominant orientation -- those 27 are poses ~174 deg from it. The
    sequential pass makes a window flip-free regardless; row 0's rule keeps the same
    pose on the same sign across samples everywhere but those 27 frames.
    """
    quat = pose[..., 3:7].clone()
    quat = torch.where(quat[..., :1, anchor : anchor + 1] < 0, -quat, quat)
    for t in range(1, quat.shape[-2]):
        dot = (quat[..., t, :] * quat[..., t - 1, :]).sum(-1, keepdim=True)
        quat[..., t, :] = torch.where(dot < 0, -quat[..., t, :], quat[..., t, :])
    return torch.cat([pose[..., :3], quat], dim=-1)


class _XVLASoftFoldWriter(SlotWriter):
    """X-VLA soft-fold (cloth folding, bimanual, 20 Hz): the 96-wide observation.state.

        cols     contents                                            -> slot
        [ 0:14]  eef euler, per arm xyz + rpy + gripper               (unused)
        [14:21]  left  eef xyz + quat XYZW                              left_eef
        [21]     left  gripper                                          left_gripper
        [22:29]  right eef xyz + quat XYZW                              right_eef
        [29]     right gripper                                          right_gripper
        [30:50]  eef "6d", per arm xyz + 6 + gripper                  (unused)
        [50:52]  eef left/right timestamps                            (unused)
        [52:66]  qpos, per arm 6 joints + gripper                     (unused)
        [66:94]  qvel, effort                                         (unused)
        [94:96]  qpos left/right timestamps                           (unused)

    Measured on the data: the quaternion block is xyzw -- it matches the euler block
    read as scipy "xyz" to 1.5e-5 deg, both arms -- and unit to 1e-4. xyz and gripper
    are bit-identical across the euler / quat / 6d blocks. The "6d" block matches
    neither the first two rows nor the first two columns of R(quat), so it is unused.
    The gripper equals qpos' gripper columns, about -0.002 .. 0.069 (opening width,
    larger = more open), copied as is.

    The `action` column is joint space, and action[t] == qpos[t+1] exactly on all
    2,852,512 frames -- it is the next state. So the action is read off the same state
    column one row ahead (`action_from_state: true`, `state_len: 1` on the action
    transform, as in GetActionBridge), and no FK is needed.

    info.json says robot_type "franka"; the 2 x (6 joints + gripper) layout and the
    cam_high / cam_left_wrist / cam_right_wrist rig are ALOHA-style. Nothing here
    depends on which: the poses are recorded, not computed.
    """

    SOURCE_COLS = XVLA_SOFT_FOLD_SOURCE_COLS
    EEF_SLOTS = ("left_eef", "right_eef")

    def _init(self, key_mapper, rows, slot_cols, source_cols):
        self.key_mapper = key_mapper
        self.rows = rows
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )

    def _fill(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]

        needed = max(col for cols in self.source_cols.values() for col in cols) + 1
        if vector.shape[-1] < needed:
            raise ValueError(
                f"'{source}' is {vector.shape[-1]} wide but source_cols reads up to "
                f"column {needed - 1}; this transform expects the 96-wide state."
            )

        filled = {}
        slots = get_pose_slots(vector.shape[0])
        for slot_name, cols in self.source_cols.items():
            block = vector[..., cols]
            if slot_name in self.EEF_SLOTS:
                block = quat_continuous_in_window(block)
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)

        return data, mask_out


@TRANSFORMS.register_module()
class GetStateXVLASoftFold(_XVLASoftFoldWriter):
    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None):
        self._init(key_mapper, slice(0, int(state_len)), slot_cols, source_cols)

    def __call__(self, data):
        data, _ = self._fill(data)
        return data


@TRANSFORMS.register_module()
class GetActionXVLASoftFold(_XVLASoftFoldWriter):
    """The action as the state one row ahead.

    Same shape as GetActionBridge: the state column carries the whole frame window and
    `state_len` is a ROW OFFSET into it, so `state_len: 1` with `action_len: N` selects
    rows 1..N -- the pose each transition reaches. Requires `action_from_state: true`.
    `pad_key` (observation.state_is_pad, since the rows ARE state rows) drops the rows
    past an episode's end from the mask.
    """

    def __init__(
        self,
        key_mapper,
        action_len,
        state_len=0,
        pad_key=None,
        slot_cols=None,
        source_cols=None,
    ):
        self.state_len, self.action_len = int(state_len), int(action_len)
        self._init(
            key_mapper,
            slice(self.state_len, self.state_len + self.action_len),
            slot_cols,
            source_cols,
        )
        self.pad_key = pad_key

    def __call__(self, data):
        data, mask_out = self._fill(data)
        if mask_out is None or self.pad_key is None or self.pad_key not in data:
            return data

        is_pad = torch.as_tensor(data[self.pad_key]).reshape(-1)[self.rows].bool()
        if is_pad.shape[0] != data[mask_out].shape[0]:
            raise ValueError(
                f"{self.pad_key!r} has {is_pad.shape[0]} rows in the action window but "
                "the "
                f"mask has {data[mask_out].shape[0]}. The pad tensor spans the whole "
                f"delta_timestamps window, so it has to be sliced the same way."
            )
        data[mask_out] = data[mask_out] & ~is_pad[:, None]
        return data


RT1_FRACTAL_SOURCE_COLS = {
    "left_eef": list(range(0, 7)),
    "left_gripper": [7],
}


@TRANSFORMS.register_module()
class GetStateRT1Fractal(SlotWriter):
    SOURCE_ORDER = ("state vector, xyz + quat xyzw + gripper",)
    SOURCE_COLS = RT1_FRACTAL_SOURCE_COLS

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None) -> None:
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )
        self.rows = slice(0, self.state_len)

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]

        filled = {}
        slots = get_pose_slots(vector.shape[0])
        for slot_name, cols in self.source_cols.items():
            filled[slot_name] = self._write(slots, slot_name, vector[..., cols], source)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)

        return data


@TRANSFORMS.register_module()
class GetActionRT1Fractal(SlotWriter):
    SOURCE_ORDER = GetStateRT1Fractal.SOURCE_ORDER
    SOURCE_COLS = RT1_FRACTAL_SOURCE_COLS

    def __init__(
        self, key_mapper, action_len, state_len=0, slot_cols=None, source_cols=None
    ) -> None:
        self.key_mapper = key_mapper
        self.state_len, self.action_len = int(state_len), int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )
        self.rows = slice(self.state_len, self.state_len + self.action_len)

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]

        filled = {}
        slots = get_pose_slots(vector.shape[0])
        for slot_name, cols in self.source_cols.items():
            filled[slot_name] = self._write(slots, slot_name, vector[..., cols], source)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)

        return data


KAI0_SOURCE_COLS = {
    "left_eef": list(range(0, 7)),
    "left_gripper": [7],
    "right_eef": list(range(8, 15)),
    "right_gripper": [15],
}


@TRANSFORMS.register_module()
class GetStateKai0(WideVectorWriter):
    SOURCE_COLS = KAI0_SOURCE_COLS

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None) -> None:
        self._init(key_mapper, state_len, slot_cols, source_cols)
        self.rows = slice(0, state_len)


@TRANSFORMS.register_module()
class GetActionKai0(WideVectorWriter):
    SOURCE_COLS = KAI0_SOURCE_COLS

    def __init__(
        self, key_mapper, action_len, state_len=0, slot_cols=None, source_cols=None
    ) -> None:
        self._init(key_mapper, action_len, slot_cols, source_cols)
        self.rows = slice(state_len, state_len + action_len)


HIFI_UMI_SOURCE_COLS = {
    "left_eef": list(range(10, 19)),
    "left_gripper": [19],
    "right_eef": list(range(0, 9)),
    "right_gripper": [9],
}

# Position and gripper have real per-dim stats; the rotation does not -- see
# the six rot6d columns are dropped here.
HIFI_UMI_STATS_COLS = {
    "left_eef": [10, 11, 12],
    "left_gripper": [19],
    "right_eef": [0, 1, 2],
    "right_gripper": [9],
}


@TRANSFORMS.register_module()
class GetStateHiFiUMI(SlotWriter):
    """HiFi-UMI-2K (simpleai_umi_4.0): the 20-wide `observation.state` column.

        slice     contents                        -> slot
        [ 0: 3]   right eef xyz                      right_eef (cols 0:3)
        [ 3: 9]   right eef rotation, 6D             right_eef (cols 3:7, as quat)
        [ 9:10]   right gripper angle, rad           right_gripper
        [10:13]   left  eef xyz                      left_eef  (cols 0:3)
        [13:19]   left  eef rotation, 6D             left_eef  (cols 3:7, as quat)
        [19:20]   left  gripper angle, rad           left_gripper

    Two things separate this from a plain WideVectorWriter column copy. The source
    stores the RIGHT arm first, which costs no code -- SOURCE_COLS names the columns
    per slot and `resolve_source_cols` emits them in SLOT_OFFSETS order, so the read
    is right-then-left while the write is left-then-right. And each 9-wide eef block
    becomes 7 dims, since the layout's eef slots are xyz + quat: that width change is
    what SOURCE_COLS alone cannot express, so `__call__` is spelled out below rather
    than inherited.

    `valid_source` is off by default. info.json's invalid_policy is
    carry_forward_state_with_observation.state_valid_false -- an invalid frame repeats
    the previous pose instead of storing a sentinel, so a bad row is a stale duplicate
    and nothing downstream can tell. Naming `observation.state_valid` here folds those
    flags into the mask; it also means adding that feature to RefactorLerobot's
    key_mapper so the datamodule requests the column.
    """

    SOURCE_ORDER = ("state vector, right then left, xyz + rot6d + gripper",)
    SOURCE_COLS = HIFI_UMI_SOURCE_COLS
    EEF_SLOTS = ("left_eef", "right_eef")

    def __init__(
        self,
        key_mapper,
        state_len,
        slot_cols=None,
        source_cols=None,
        valid_source=None,
    ) -> None:
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )
        self.valid_source = valid_source
        self.rows = slice(0, self.state_len)

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]

        needed = max(col for cols in self.source_cols.values() for col in cols) + 1
        if vector.shape[-1] < needed:
            raise ValueError(
                f"'{source}' is {vector.shape[-1]} wide but source_cols reads up to "
                f"column {needed - 1}. The whole layout is column indices into this "
                f"one feature, so a schema change invalidates all of them."
            )

        filled = {}
        slots = get_pose_slots(self.state_len)
        for slot_name, cols in self.source_cols.items():
            block = vector[..., cols]
            if slot_name in self.EEF_SLOTS:
                block = eef_rot6d_to_quat(block)
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = self._mask(data, filled)

        return data

    def _mask(self, data, filled):
        """The structural mask, narrowed by the per-frame validity flags if asked.

        build_mask_from_cols marks which COLUMNS this robot fills, the same for every
        frame; the flags are per frame and per source dim, so they are ANDed on top.
        Position and gripper map one to one. The six rot6d flags collapse with .all()
        and cover all four quaternion columns: one bad component invalidates the whole
        rotation, and there is no partial quaternion to salvage.
        """
        mask = build_mask_from_cols(self.state_len, filled)
        if self.valid_source is None:
            return mask

        valid = as_column(data[self.valid_source])[self.rows].bool()
        for slot_name, cols in self.source_cols.items():
            lo, _ = SLOT_OFFSETS[slot_name]
            if slot_name in self.EEF_SLOTS:
                mask[:, lo : lo + 3] &= valid[:, cols[:3]]
                mask[:, lo + 3 : lo + 7] &= valid[:, cols[3:]].all(dim=-1, keepdim=True)
            else:
                mask[:, lo : lo + len(cols)] &= valid[:, cols]

        return mask


@TRANSFORMS.register_module()
class GetActionHiFiUMI(SlotWriter):
    """The same robot's 20-wide `action` column, same layout as the state.

    info.json's action_layout.type is absolute_next_state_target: rows are
    [state_1, state_2, ..., state_last, state_last], an absolute pose in the same frame
    and units as the state, one step ahead. That needs nothing here -- it is what
    ToRelativeAction downstream already assumes -- but it does give a sharp alignment
    check: action[:-1] must equal state[1:] on the position and gripper columns.

    `action_from_state` is false for this dataset, so the datamodule hands `action` its
    own action_len-row window and state_len stays 0. The body is GetStateHiFiUMI's,
    written out rather than shared: the only differences are the row window and which
    validity feature `valid_source` names.
    """

    SOURCE_ORDER = GetStateHiFiUMI.SOURCE_ORDER
    SOURCE_COLS = HIFI_UMI_SOURCE_COLS
    EEF_SLOTS = ("left_eef", "right_eef")

    def __init__(
        self,
        key_mapper,
        action_len,
        state_len=0,
        slot_cols=None,
        source_cols=None,
        valid_source=None,
    ) -> None:
        self.key_mapper = key_mapper
        self.state_len, self.action_len = int(state_len), int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = resolve_source_cols(
            self.SOURCE_COLS, source_cols, type(self).__name__
        )
        self.valid_source = valid_source
        self.rows = slice(self.state_len, self.state_len + self.action_len)

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[self.rows]

        needed = max(col for cols in self.source_cols.values() for col in cols) + 1
        if vector.shape[-1] < needed:
            raise ValueError(
                f"'{source}' is {vector.shape[-1]} wide but source_cols reads up to "
                f"column {needed - 1}. The whole layout is column indices into this "
                f"one feature, so a schema change invalidates all of them."
            )

        filled = {}
        slots = get_pose_slots(self.action_len)
        for slot_name, cols in self.source_cols.items():
            block = vector[..., cols]
            if slot_name in self.EEF_SLOTS:
                block = eef_rot6d_to_quat(block)
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = self._mask(data, filled)

        return data

    def _mask(self, data, filled):
        """As GetStateHiFiUMI._mask, over the action window and `action_valid`."""
        mask = build_mask_from_cols(self.action_len, filled)
        if self.valid_source is None:
            return mask

        valid = as_column(data[self.valid_source])[self.rows].bool()
        for slot_name, cols in self.source_cols.items():
            lo, _ = SLOT_OFFSETS[slot_name]
            if slot_name in self.EEF_SLOTS:
                mask[:, lo : lo + 3] &= valid[:, cols[:3]]
                mask[:, lo + 3 : lo + 7] &= valid[:, cols[3:]].all(dim=-1, keepdim=True)
            else:
                mask[:, lo : lo + len(cols)] &= valid[:, cols]

        return mask


ROBOTWIN_SOURCE_COLS = {
    "left_gripper": [6],
    "right_gripper": [13],
}


@TRANSFORMS.register_module()
class GetStateRoboTwin(WideVectorWriter):
    """RoboTwin (aloha-agilex, simulated): the 14-dim `observation.state` column.

        slice     contents                        -> slot
        [ 6: 7]   left  gripper, 0..1                left_gripper
        [13:14]   right gripper, 0..1                right_gripper

    No eef pose is recorded, so left_eef/right_eef stay zero -- as in GetStateRobomind,
    joints are the only pose this robot exposes.
    """

    SOURCE_COLS = ROBOTWIN_SOURCE_COLS

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None) -> None:
        self._init(key_mapper, state_len, slot_cols, source_cols)


@TRANSFORMS.register_module()
class GetActionRoboTwin(WideVectorWriter):
    SOURCE_COLS = ROBOTWIN_SOURCE_COLS

    def __init__(
        self, key_mapper, action_len, slot_cols=None, source_cols=None
    ) -> None:
        self._init(key_mapper, action_len, slot_cols, source_cols)


LIBERO_SOURCE_COLS = {
    "left_eef": list(range(0, 6)),
    "left_gripper": [6],
}


@TRANSFORMS.register_module()
class GetStateLibero(SlotWriter):
    SOURCE_COLS = LIBERO_SOURCE_COLS

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None):
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[slice(0, self.state_len)]
        filled = {}
        slots = get_pose_slots(vector.shape[0])
        filled["left_eef"] = self._write(
            slots,
            "left_eef",
            eef_rotvec_to_quat(vector[..., self.source_cols["left_eef"]]),
            source,
        )
        grip = self.source_cols["left_gripper"]
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            vector[..., grip],
            source,
        )
        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)
        return data


@TRANSFORMS.register_module()
class GetActionLibero(SlotWriter):
    SOURCE_COLS = LIBERO_SOURCE_COLS

    def __init__(
        self, key_mapper, action_len, state_len=0, slot_cols=None, source_cols=None
    ):
        self.key_mapper = key_mapper
        self.state_len, self.action_len = int(state_len), int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])[
            slice(self.state_len, self.state_len + self.action_len)
        ]
        filled = {}
        slots = get_pose_slots(vector.shape[0])
        filled["left_eef"] = self._write(
            slots,
            "left_eef",
            eef_rotvec_to_quat(vector[..., self.source_cols["left_eef"]]),
            source,
        )
        grip = self.source_cols["left_gripper"]
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            vector[..., grip],
            source,
        )
        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)
        return data


MIMICGEN_SOURCE_COLS = {
    "left_eef": list(range(0, 7)),  # xyz + quat, wxyz -> permuted to the layout's xyzw
    "left_gripper": [7],
}


@TRANSFORMS.register_module()
class GetStateMimicGen(SlotWriter):
    """MimicGen (robosuite, 12 tasks): the 8-dim `observation.state` column.

        slice     contents                    -> slot
        [0:3]     eef xyz                        left_eef cols 0:3
        [3:7]     eef quaternion, WXYZ           left_eef cols 3:7, permuted to xyzw
        [7:8]     gripper                        left_gripper

    info.json names the columns `x, y, z, qw, qx, qy, qz, gripper`, so w comes FIRST,
    and the data agrees: |q| is 1.000000 with std 1.7e-08 and column 3 is the dominant
    component (|mean| 0.869). The layout is xyzw throughout, so the block is permuted,
    not converted.

    One arm, no joints, so everything else stays zero and out of the mask.
    """

    SOURCE_COLS = MIMICGEN_SOURCE_COLS

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None):
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])
        eef = self.source_cols["left_eef"]
        filled = {}
        slots = get_pose_slots(vector.shape[0])
        filled["left_eef"] = self._write(
            slots,
            "left_eef",
            torch.cat(
                [vector[..., eef[:3]], quat_wxyz_to_xyzw(vector[..., eef[3:]])], dim=-1
            ),
            source,
        )
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            vector[..., self.source_cols["left_gripper"]],
            source,
        )
        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)
        return data


@TRANSFORMS.register_module()
class GetActionMimicGen(SlotWriter):
    SOURCE_COLS = MIMICGEN_SOURCE_COLS

    def __init__(self, key_mapper, action_len, slot_cols=None, source_cols=None):
        self.key_mapper = key_mapper
        self.action_len = int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            **self.SOURCE_COLS,
            **{name: list(cols) for name, cols in (source_cols or {}).items()},
        }

    def __call__(self, data):
        source = parse_wide_source(
            list(self.key_mapper[0].keys())[0], type(self).__name__
        )
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        vector = as_column(data[source])
        eef = self.source_cols["left_eef"]
        filled = {}
        slots = get_pose_slots(vector.shape[0])
        filled["left_eef"] = self._write(
            slots,
            "left_eef",
            torch.cat(
                [vector[..., eef[:3]], quat_wxyz_to_xyzw(vector[..., eef[3:]])], dim=-1
            ),
            source,
        )
        filled["left_gripper"] = self._write(
            slots,
            "left_gripper",
            vector[..., self.source_cols["left_gripper"]],
            source,
        )
        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(vector.shape[0], filled)
        return data


########################### Norm Stats Transforms #############################


@TRANSFORMS.register_module()
class GetStateInternDataA1Genie1(SlotWriter):
    SOURCE_ORDER = (
        "left eef state",
        "left gripper state",
        "right eef state",
        "right gripper state",
    )
    SLOT_ORDER = (
        "left_eef",
        "left_gripper",
        "right_eef",
        "right_gripper",
    )
    QUAT_SLOTS = ("left_eef", "right_eef")

    def __init__(
        self,
        key_mapper,
        state_len,
        slot_cols=None,
        source_cols=None,
    ) -> None:
        self.key_mapper = key_mapper
        self.state_len = int(state_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            name: list(cols) for name, cols in (source_cols or {}).items()
        }

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        state_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.state_len)
        for slot_name, source in zip(self.SLOT_ORDER, sources):
            if source is None:
                continue
            # as_column before the column selection: a gripper is one value per frame,
            # so
            # it arrives as (frames,) and indexing that with source_cols would slice the
            # frame axis instead of a column axis.
            block = as_column(data[source])
            source_cols = self.source_cols.get(slot_name)
            if source_cols is not None:
                block = block[..., source_cols]
            if slot_name in self.QUAT_SLOTS:
                if block.shape[-1] != 7:
                    raise ValueError(
                        f"'{source}' supplies {block.shape[-1]} dims for the "
                        f"{slot_name!r} slot; an eef pose has to be xyz(3) + quat(4) "
                        "so "
                        "the scalar-first quaternion can be reordered. A joint vector "
                        "or "
                        f"an already-flattened pair of arms would be scrambled here."
                    )
                block = torch.cat(
                    [block[..., 0:3], quat_wxyz_to_xyzw(block[..., 3:7])], dim=-1
                )
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[state_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.state_len, filled)

        return data


@TRANSFORMS.register_module()
class GetActionInternDataA1Genie1(SlotWriter):
    SOURCE_ORDER = (
        "left eef action",
        "left gripper action",
        "right eef action",
        "right gripper action",
    )
    SLOT_ORDER = (
        "left_eef",
        "left_gripper",
        "right_eef",
        "right_gripper",
    )
    QUAT_SLOTS = ("left_eef", "right_eef")

    def __init__(
        self,
        key_mapper,
        action_len,
        slot_cols=None,
        source_cols=None,
    ) -> None:
        self.key_mapper = key_mapper
        self.action_len = int(action_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        self.source_cols = {
            name: list(cols) for name, cols in (source_cols or {}).items()
        }

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        action_out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        filled = {}
        slots = get_pose_slots(self.action_len)
        for slot_name, source in zip(self.SLOT_ORDER, sources):
            if source is None:
                continue
            # as_column before the column selection: a gripper is one value per frame,
            # so
            # it arrives as (frames,) and indexing that with source_cols would slice the
            # frame axis instead of a column axis.
            block = as_column(data[source])
            source_cols = self.source_cols.get(slot_name)
            if source_cols is not None:
                block = block[..., source_cols]
            if slot_name in self.QUAT_SLOTS:
                if block.shape[-1] != 7:
                    raise ValueError(
                        f"'{source}' supplies {block.shape[-1]} dims for the "
                        f"{slot_name!r} slot; an eef pose has to be xyz(3) + quat(4) "
                        "so "
                        "the scalar-first quaternion can be reordered. A joint vector "
                        "or "
                        f"an already-flattened pair of arms would be scrambled here."
                    )
                block = torch.cat(
                    [block[..., 0:3], quat_wxyz_to_xyzw(block[..., 3:7])], dim=-1
                )
            filled[slot_name] = self._write(slots, slot_name, block, source)

        data[action_out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.action_len, filled)

        return data


class _GetG2A(SlotWriter):
    """G2A: extra 14-d eef_fk (left|right xyz+xyzw) plus native joint/chassis vector.

    key_mapper names two sources: (eef_fk, native_vec). The five body joints are
    written into the waist slot, the same placement as AgiBotWorld2026. The torso
    slot stays zero, so its action_mask stays False.
    """

    SOURCE_ORDER = ("eef fk", "native vector")
    EEF_COLS = {
        "left_eef": list(range(0, 7)),
        "right_eef": list(range(7, 14)),
    }
    VEC_COLS = {}
    QUAT_SLOTS = ("left_eef", "right_eef")

    def _init_g2a(self, key_mapper, traj_len, slot_cols, source_cols):
        self.key_mapper = key_mapper
        self.traj_len = int(traj_len)
        self.slot_cols = {name: list(cols) for name, cols in (slot_cols or {}).items()}
        overrides = {name: list(cols) for name, cols in (source_cols or {}).items()}
        self.eef_cols = {
            **self.EEF_COLS,
            **{k: v for k, v in overrides.items() if k in self.EEF_COLS},
        }
        self.vec_cols = {
            **self.VEC_COLS,
            **{k: v for k, v in overrides.items() if k in self.VEC_COLS},
        }

    def __call__(self, data):
        sources = eval(list(self.key_mapper[0].keys())[0])
        out, mask_out = unpack_output(list(self.key_mapper[0].values())[0])
        if len(sources) != len(self.SOURCE_ORDER):
            raise ValueError(
                f"key_mapper must name {len(self.SOURCE_ORDER)} sources "
                f"({', '.join(self.SOURCE_ORDER)}), got {len(sources)}: {sources}. "
                f"A missing comma between two names silently concatenates them."
            )
        eef_source, vec_source = sources
        eef = as_column(data[eef_source])
        vec = as_column(data[vec_source])
        filled = {}
        slots = get_pose_slots(self.traj_len)
        for slot_name, cols in self.eef_cols.items():
            filled[slot_name] = self._write(
                slots, slot_name, eef[..., cols], eef_source
            )
        for slot_name, cols in self.vec_cols.items():
            filled[slot_name] = self._write(
                slots, slot_name, vec[..., cols], vec_source
            )
        data[out] = torch.cat([slots[name] for name in SLOT_OFFSETS], dim=-1)
        if mask_out is not None:
            data[mask_out] = build_mask_from_cols(self.traj_len, filled)
        return data


@TRANSFORMS.register_module()
class GetActionWorldOracle(WideVectorWriter):
    """AgiBot G2's 52-dim `action` column, sliced into the unified layout.

    Same robot and same action semantics as GetActionAgiBotWorld. The difference is
    only the export: WorldOracle ships one wide vector where the native AgiBot
    lerobot export carries six named columns, so arm and body features come from
    column indices rather than from separate sources. WideVectorWriter exists for
    exactly this case.

    slice        contents                        -> slot
    [ 0]         left  gripper command             left_gripper
    [ 1]         right gripper command             right_gripper
    [ 2: 5]      left  eef position, m             left_eef   pos
    [ 5: 8]      right eef position, m             right_eef  pos
    [ 8:12]      left  eef orientation, unit quat  left_eef   quat
    [12:16]      right eef orientation, unit quat  right_eef  quat
    [16:30]      both arms' joint angles           unused
    [30:33]      head joints                      head
    [33:38]      waist joints                     waist
    [38:44]      chassis velocity                 chassis, first three (planar)
    [44:52]      both arms' reference pose        unused

    No torso source exists on this robot, so that slot stays empty and masked, the
    same as the native AgiBot path.

    Verified against a dumped episode (agibot/pickfruit/ep000000, 3553 frames):
    every 4-tuple in 8:16 has unit norm, so it is a quaternion and not the
    "XYZ + arm-shape angle" that agibot/*/schema.json claims. Trust the data.
    """

    SOURCE_COLS = {
        "left_eef": [2, 3, 4, 8, 9, 10, 11],
        "right_eef": [5, 6, 7, 12, 13, 14, 15],
        "left_gripper": [0],
        "right_gripper": [1],
        "head": [30, 31, 32],
        "waist": [33, 34, 35, 36, 37],
        "chassis": [38, 39, 40],
    }

    def __init__(
        self, key_mapper, action_len, slot_cols=None, source_cols=None
    ) -> None:
        self._init(key_mapper, action_len, slot_cols, source_cols)


@TRANSFORMS.register_module()
class GetStateWorldOracle(WideVectorWriter):
    """AgiBot G2's 208-dim `observation.state` column, sliced into the unified layout.

    Different indices from the action: the state carries a wider proprioceptive record
    (wrench, velocities, joint efforts, camera extrinsics), so the same quantities sit
    further into the vector.

    slice        contents                        -> slot
    [ 0]         left  gripper stroke             left_gripper
    [ 1]         right gripper stroke             right_gripper
    [26:32]      both eef positions, m            left_eef / right_eef   pos
    [56:64]      both eef orientations, quat XYZW left_eef / right_eef   quat
    [120:123]    head joints                     head
    [123:128]    waist joints                    waist

    No body velocity is recorded, so the chassis slot stays empty and masked. AgiBot G2
    actuates only the two arms; the action's chassis block is identically zero too, which
    is why routing it would condition the model on a constant.
    """

    SOURCE_COLS = {
        "left_eef": [26, 27, 28, 56, 57, 58, 59],
        "right_eef": [29, 30, 31, 60, 61, 62, 63],
        "left_gripper": [0],
        "right_gripper": [1],
        "head": [120, 121, 122],
        "waist": [123, 124, 125, 126, 127],
    }

    def __init__(self, key_mapper, state_len, slot_cols=None, source_cols=None) -> None:
        self._init(key_mapper, state_len, slot_cols, source_cols)
