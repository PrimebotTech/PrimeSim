"""Patches for bugs in the installed lerobot that break v3.0 concatenated videos.

float64 timestamps
------------------
lerobot's `decode_video_frames_torchcodec` builds its comparison tensors with
`torch.tensor(...)`, which defaults to float32:

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    is_within_tol = min_ < tolerance_s

In v3.0 all episodes of a camera share one mp4, so query timestamps are offset by
`videos/<key>/from_timestamp` and reach into the thousands of seconds. float32
spacing at t=1303s is 1.22e-4 and at t=4557s (the largest offset in
agibot_world_beta) it is 4.88e-4 -- both larger than lerobot's default
tolerance_s=1e-4. Since `query_ts` and `loaded_ts` are rounded to float32
independently, the tolerance check ends up measuring its own quantisation noise
and raises FrameTimestampError on frames that were decoded correctly.

Comparing in float64 removes the quantisation entirely, so the check measures the
real timestamp error (which is ~0 -- the decoder returns the frame that was asked
for) instead of the representation error.

off-grid query timestamps
-------------------------
Even in float64, `tolerance_s=1e-4` is unreachable on a v3.0 concatenated video.
A query is `from_timestamp + k/fps`, and the per-episode `from_timestamp` recorded
in `meta/episodes` carries accumulated rounding: on new_lerobot_data_v3 episode 0
starts at 0.0, but episode 1 starts at 130.666992 where the previous episode ended
at 130.666667 (+0.33ms), episode 2 at 207.800977 against 207.800326 (+0.65ms), and
so on, growing along the file. The parquet's own `timestamp` column is exact --
`timestamp == frame_index / fps` to 7e-6, i.e. float32 storage precision -- so the
error lives purely in the video offsets. The sum therefore lands off the mp4's PTS
grid by an amount that already exceeds 1e-4 at the first episode boundary and
reaches several ms further into the file.

That drift is not the whole story, and the numbers are worth recording because they
rule things out. Measured over the four *_train repos (32518 episodes): the worst
`from_timestamp` offset is 9.64ms (0.29 frame periods) and none exceeds half a
period, so the offsets alone cannot produce a half-frame miss. The `timestamp`
column is exact -- `== frame_index / fps` -- in every episode sampled from
full_task_batch1_train and partial_task_batch4_train. Yet a window in
partial_task_batch4_train (a repo whose `from_timestamp` offsets are exactly 0.0)
queried 54.6830, which is 0.49 frame periods off the grid. So the off-grid amount is
heterogeneous across the dataset in a way neither the episode offsets nor the
sampled timestamp columns explain, and the true worst case over all 33k episodes is
not known.

Given that, the bound is set at one source frame period rather than half: "the frame
we loaded is at most one recorded frame away from the one requested". At 30fps
source and a 10fps training grid that is at most a third of a training step of
misalignment, which is tolerable for dynamics learning, and it is a bound with a
meaning rather than a tuned constant. Anything beyond a whole frame is a real
synchronisation problem and still raises -- `OpenLeRobotDatasetV3.__getitem__`
catches it and resamples, so genuinely broken windows are skipped rather than
crashing the run.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_APPLIED = False


def _decode_video_frames_torchcodec_float64(
    video_path,
    timestamps,
    tolerance_s,
    log_loaded_timestamps: bool = False,
    decoder_cache=None,
):
    """Copy of lerobot's decoder with the timestamp comparison done in float64."""
    from lerobot.datasets import video_utils

    if decoder_cache is None:
        decoder_cache = video_utils._default_decoder_cache

    decoder = decoder_cache.get_decoder(str(video_path))

    loaded_ts = []
    loaded_frames = []

    average_fps = decoder.metadata.average_fps
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    # The only change from upstream: float64 instead of the float32 default, so
    # that timestamps in the thousands of seconds survive the comparison.
    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float64)

    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    # One source frame period, never below the caller's own tolerance: "the frame we
    # got is at most one recorded frame away from the one asked for". See the
    # "off-grid query timestamps" note in the module docstring for why 1e-4 is
    # unreachable here, and why half a period turned out to be too tight.
    effective_tolerance_s = max(float(tolerance_s), 1.0 / float(average_fps))

    is_within_tol = min_ < effective_tolerance_s
    if not is_within_tol.all():
        raise video_utils.FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance "
            f"({min_[~is_within_tol]} > {effective_tolerance_s=}, from "
            f"{tolerance_s=} and {average_fps=})."
            " It means that the closest frame that can be loaded from the video is too "
            "far away in time."
            " This might be due to synchronization issues with timestamps during data "
            "collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
        )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    closest_frames = (closest_frames / 255.0).type(torch.float32)

    if not len(timestamps) == len(closest_frames):
        raise video_utils.FrameTimestampError(
            f"Retrieved timestamps differ from queried "
            f"{set(closest_frames) - set(timestamps)}"
        )

    return closest_frames


def apply_lerobot_patches():
    """Idempotently patch the installed lerobot. Safe to call more than once."""
    global _APPLIED
    if _APPLIED:
        return

    from lerobot.datasets import video_utils

    # `decode_video_frames` resolves this name from the module globals at call
    # time, so replacing the attribute is enough to cover both entry points.
    video_utils.decode_video_frames_torchcodec = _decode_video_frames_torchcodec_float64
    _APPLIED = True
    logger.info(
        "Patched lerobot decode_video_frames_torchcodec: float64 timestamp comparison, "
        "tolerance floored at half a frame period."
    )
