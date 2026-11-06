"""Episode-aligned sampling for cross-window self-forcing training.

Stage 4 keeps the generator's KV cache alive across adjacent windows.  Ordinary
shuffle/batch sampling cannot provide that ordering, so this module turns every
LeRobot episode into a temporal run and schedules equal-length subchains across
all distributed ranks.  The sampler also carries exact chain-boundary metadata
in its indices; :class:`AmortizedConcatDataset` materializes it in each batch.
"""

import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass(frozen=True)
class EpisodeRun:
    config_id: int
    repo_id: str
    episode_index: int
    global_indices: np.ndarray
    timestamps: np.ndarray

    def __len__(self):
        return int(self.global_indices.shape[0])

    @property
    def key(self):
        return self.config_id, self.repo_id, self.episode_index


@dataclass(frozen=True)
class AmortizedSampleIndex:
    index: int
    episode_start: bool
    episode_end: bool
    episode_id: int
    segment_index: int
    episode_num_segments: int


def contiguous_spans(values):
    """Yield ``(value, start, stop)`` for maximal runs of equal values."""
    if len(values) == 0:
        return
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(values)]))
    for start, stop in zip(starts, stops):
        yield values[start], int(start), int(stop)


def episode_runs(concat, *, split="train"):
    """Return every addressable episode in an outer ``ConcatDataset``.

    Expected nesting is ``ConcatDataset(PipelineSubset(MixedDataset))``.  Stage 4
    requires stride one because a larger stride randomly chooses a raw frame and
    therefore cannot address a deterministic temporal grid.
    """
    bases = [0] + list(concat.cumulative_sizes[:-1])
    runs = []
    total_frames = kept_frames = 0
    for config_id, (subset, base) in enumerate(zip(concat.datasets, bases)):
        mixed = subset.dataset
        stride = int(getattr(mixed, "stride", 1))
        if stride != 1:
            raise ValueError(
                "Stage-4 episode sampling requires stride=1; "
                f"dataset config {config_id} has stride={stride}."
            )
        inverse = {int(value): pos for pos, value in enumerate(subset.indices)}
        inner = mixed._dataset
        repo_bases = [0] + list(inner.cumulative_sizes[:-1])
        for repo_base, dataset in zip(repo_bases, inner.datasets):
            table = getattr(dataset.hf_dataset, "_data", None)
            if table is None:
                raise ValueError(
                    f"{dataset.repo_id}: the Arrow table is unavailable; "
                    "episode sampling needs timestamp metadata."
                )
            timestamps = table["timestamp"].to_numpy()
            episodes = table["episode_index"].to_numpy()
            for episode_index, lo, hi in contiguous_spans(episodes):
                total_frames += hi - lo
                positions = np.fromiter(
                    (inverse.get(repo_base + frame, -1) for frame in range(lo, hi)),
                    dtype=np.int64,
                    count=hi - lo,
                )
                keep = positions >= 0
                if (
                    subset.preserves_episodes
                    and bool(keep.any())
                    and not bool(keep.all())
                ):
                    raise ValueError(
                        f"{dataset.repo_id} episode {int(episode_index)} was split "
                        "across train and validation."
                    )
                if int(keep.sum()) < 2:
                    continue
                kept_frames += int(keep.sum())
                runs.append(
                    EpisodeRun(
                        config_id=config_id,
                        repo_id=str(dataset.repo_id),
                        episode_index=int(episode_index),
                        global_indices=base + positions[keep],
                        timestamps=timestamps[lo:hi][keep].astype(np.float64),
                    )
                )
    dropped = total_frames - kept_frames
    share = 100.0 * dropped / total_frames if total_frames else 0.0
    print(
        f"[stage4-data] {split}: {len(runs)} episodes, "
        f"{kept_frames}/{total_frames} frames addressable ({share:.1f}% excluded)"
    )
    return runs


class AmortizedBatchSampler(Sampler):
    """Schedule consecutive, equal-length episode subchains on every rank."""

    # The sampler shards complete chains itself.  Accelerate must not shard the
    # resulting DataLoader a second time.
    rank_sharded = True

    def __init__(
        self,
        runs,
        *,
        batch_size,
        opening_horizon,
        continuation_horizon,
        fps,
        shuffle=True,
        seed=0,
        segment_curriculum=None,
        rank=None,
        world_size=None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if opening_horizon < 2 or continuation_horizon < 2:
            raise ValueError("Stage-4 windows must contain at least two frames.")
        if rank is None or world_size is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank() if rank is None else rank
                world_size = dist.get_world_size() if world_size is None else world_size
            else:
                rank = int(os.environ.get("RANK", 0)) if rank is None else rank
                world_size = (
                    int(os.environ.get("WORLD_SIZE", 1))
                    if world_size is None
                    else world_size
                )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size = int(batch_size)
        self.opening_delta = (int(opening_horizon) - 1) / float(fps)
        self.continuation_delta = (int(continuation_horizon) - 1) / float(fps)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.curriculum = self._validate_curriculum(segment_curriculum)
        self.epoch = 0
        self._len_cache = {}
        self.runs = sorted(runs, key=lambda run: run.key)
        if not self.runs:
            raise ValueError("No complete training episodes are available for stage 4.")
        self.episode_ids = {
            run.key: episode_id for episode_id, run in enumerate(self.runs)
        }
        print(
            f"[stage4-data] {len(self.runs)} episodes, rank "
            f"{self.rank}/{self.world_size}, batch_size={self.batch_size}"
        )

    @staticmethod
    def _validate_curriculum(curriculum):
        if curriculum is None:
            return ()
        stages = []
        for raw in curriculum:
            stage = {
                "start_epoch": int(raw["start_epoch"]),
                "min_segments": int(raw["min_segments"]),
                "max_segments": int(raw["max_segments"]),
                "length_sampling_power": float(
                    raw.get("length_sampling_power", 0.0)
                ),
            }
            if stage["start_epoch"] < 0 or stage["min_segments"] < 1:
                raise ValueError("Curriculum epochs and segment counts must be valid.")
            if stage["max_segments"] < stage["min_segments"]:
                raise ValueError("max_segments must be >= min_segments.")
            if not np.isfinite(stage["length_sampling_power"]) or stage[
                "length_sampling_power"
            ] < 0:
                raise ValueError(
                    "length_sampling_power must be finite and non-negative."
                )
            stages.append(stage)
        stages.sort(key=lambda stage: stage["start_epoch"])
        if not stages or stages[0]["start_epoch"] != 0:
            raise ValueError(
                "The first stage-4 curriculum entry must start at epoch 0."
            )
        if len({stage["start_epoch"] for stage in stages}) != len(stages):
            raise ValueError("Curriculum start_epoch values must be unique.")
        return tuple(stages)

    def _active_curriculum(self):
        if not self.curriculum:
            return None
        active = self.curriculum[0]
        for stage in self.curriculum[1:]:
            if self.epoch < stage["start_epoch"]:
                break
            active = stage
        return active

    def _sample_subchain(self, run, stage):
        duration = float(run.timestamps[-1] - run.timestamps[0])
        if duration + 1e-9 < self.opening_delta:
            return np.empty(0, dtype=np.int64)
        max_fit = 1 + int(
            np.floor(
                (duration - self.opening_delta) / self.continuation_delta + 1e-9
            )
        )
        minimum = 1 if stage is None else stage["min_segments"]
        maximum = max_fit if stage is None else min(stage["max_segments"], max_fit)
        if maximum < minimum:
            return np.empty(0, dtype=np.int64)
        episode_id = self.episode_ids[run.key]
        rng = np.random.default_rng((self.seed, self.epoch, episode_id, 0xC011AB1E))
        choices = np.arange(minimum, maximum + 1, dtype=np.int64)
        power = 0.0 if stage is None else stage["length_sampling_power"]
        if power:
            weights = choices.astype(np.float64) ** power
            num_segments = int(rng.choice(choices, p=weights / weights.sum()))
        else:
            num_segments = int(rng.integers(minimum, maximum + 1))
        required = self.opening_delta + (num_segments - 1) * self.continuation_delta
        latest_start = float(run.timestamps[-1]) - required
        possible = int(
            np.searchsorted(run.timestamps, latest_start + 1e-9, side="right")
        )
        if possible <= 0:
            return np.empty(0, dtype=np.int64)
        start = float(run.timestamps[int(rng.integers(0, possible))])
        points = np.concatenate(
            (
                [start],
                start
                + self.opening_delta
                + np.arange(num_segments - 1) * self.continuation_delta,
            )
        )
        positions = np.searchsorted(run.timestamps, points, side="left")
        if bool((positions >= len(run)).any()):
            raise RuntimeError("A sampled stage-4 subchain exceeded its episode.")
        return run.global_indices[positions].astype(np.int64, copy=False)

    def _schedule(self):
        stage = self._active_curriculum()
        sampled = {run.key: self._sample_subchain(run, stage) for run in self.runs}
        buckets = defaultdict(list)
        for run in self.runs:
            roots = sampled[run.key]
            if len(roots):
                buckets[(run.config_id, len(roots))].append((run, roots))

        group_size = self.batch_size * self.world_size
        groups = []
        for (config_id, num_segments), records in sorted(buckets.items()):
            rng = np.random.default_rng(
                (self.seed, self.epoch, int(config_id), int(num_segments))
            )
            if self.shuffle and len(records) > 1:
                records = [records[index] for index in rng.permutation(len(records))]
            remainder = len(records) % group_size
            if remainder:
                records += [
                    records[index % len(records)]
                    for index in range(group_size - remainder)
                ]
            for start in range(0, len(records), group_size):
                groups.append((num_segments, records[start : start + group_size]))
        if self.shuffle and len(groups) > 1:
            rng = np.random.default_rng((self.seed, self.epoch, 0x5F3759DF))
            groups = [groups[index] for index in rng.permutation(len(groups))]

        rank_start = self.rank * self.batch_size
        batches = []
        for num_segments, group in groups:
            local = group[rank_start : rank_start + self.batch_size]
            for segment_index in range(num_segments):
                batches.append(
                    [
                        AmortizedSampleIndex(
                            index=int(roots[segment_index]),
                            episode_start=segment_index == 0,
                            episode_end=segment_index + 1 == num_segments,
                            episode_id=self.episode_ids[run.key],
                            segment_index=segment_index,
                            episode_num_segments=num_segments,
                        )
                        for run, roots in local
                    ]
                )
        return batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        schedule = self._schedule()
        self._len_cache[self.epoch] = len(schedule)
        yield from schedule
        self.epoch += 1

    def __len__(self):
        if self.epoch not in self._len_cache:
            self._len_cache[self.epoch] = len(self._schedule())
        return self._len_cache[self.epoch]
