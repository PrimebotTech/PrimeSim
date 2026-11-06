import copy
import random
from typing import Any, Dict, List

from mmengine import DATASETS
from mmengine.dataset import Compose
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .amortized_sampler import (
    AmortizedBatchSampler,
    AmortizedSampleIndex,
    contiguous_spans,
    episode_runs,
)
from .open_lerobot_dataset import OpenLeRobotMixedDatasetV3  # noqa: F401


class PipelineSubset(Subset):
    """A subset that applies its own pipeline over a shared base dataset."""

    def __init__(
        self,
        dataset,
        indices,
        pipeline=None,
        *,
        preserves_episodes=False,
        actions_are_relative=None,
    ):
        super().__init__(dataset, indices)
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.preserves_episodes = bool(preserves_episodes)
        self.actions_are_relative = actions_are_relative

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self[index] for index in idx]
        item = self.dataset.__getitem__(self.indices[idx], pipeline=self.pipeline)
        if self.actions_are_relative is not None and isinstance(item, dict):
            item["actions_are_relative"] = bool(self.actions_are_relative)
        return item

    def __getitems__(self, indices):
        return [self.__getitem__(index) for index in indices]


class AmortizedConcatDataset(ConcatDataset):
    """Materialize chain metadata carried by ``AmortizedSampleIndex``."""

    def __getitem__(self, idx):
        metadata = idx if isinstance(idx, AmortizedSampleIndex) else None
        raw_index = metadata.index if metadata is not None else idx
        item = super().__getitem__(int(raw_index))
        if metadata is None or not isinstance(item, dict):
            return item
        item["episode_start"] = bool(metadata.episode_start)
        item["episode_end"] = bool(metadata.episode_end)
        item["amortized_episode_id"] = int(metadata.episode_id)
        item["amortized_segment_index"] = int(metadata.segment_index)
        item["amortized_episode_num_segments"] = int(metadata.episode_num_segments)
        return item

    def __getitems__(self, indices):
        return [self.__getitem__(index) for index in indices]


def _episode_index_groups(dataset):
    """The mixed-dataset indices belonging wholly to each source episode."""
    inner = getattr(dataset, "_dataset", None)
    if not isinstance(inner, ConcatDataset):
        raise TypeError("Episode splitting requires a mixed LeRobot dataset.")
    stride = int(getattr(dataset, "stride", 1))
    repo_bases = [0] + list(inner.cumulative_sizes[:-1])
    groups = []
    for repo_base, child in zip(repo_bases, inner.datasets):
        table = getattr(child.hf_dataset, "_data", None)
        if table is None:
            raise ValueError(
                f"{child.repo_id}: the Arrow table is unavailable for episode split."
            )
        episodes = table["episode_index"].to_numpy()
        for episode_index, lo, hi in contiguous_spans(episodes):
            raw_lo, raw_hi = repo_base + lo, repo_base + hi
            first, stop = (raw_lo + stride - 1) // stride, raw_hi // stride
            if first < stop:
                groups.append(
                    (str(child.repo_id), int(episode_index), range(first, stop))
                )
    return groups


def _split_indices_by_episode(dataset, train_ratio, valid_ratio, shuffle, rng):
    groups = _episode_index_groups(dataset)
    order = list(range(len(groups)))
    if shuffle:
        rng.shuffle(order)
    num_train = int(len(order) * float(train_ratio))
    num_valid = (
        len(order) - num_train
        if valid_ratio is None
        else min(int(len(order) * float(valid_ratio)), len(order) - num_train)
    )
    train_groups = order[:num_train]
    valid_groups = order[num_train : num_train + num_valid]
    train = [index for group_id in train_groups for index in groups[group_id][2]]
    valid = [index for group_id in valid_groups for index in groups[group_id][2]]
    addressable = sum(len(group[2]) for group in groups)
    return train, valid, num_train, num_valid, len(groups), len(dataset) - addressable


def _resize_window_pipeline(pipeline, *, num_frames, action_len, defer_relative):
    """Clone a dataset pipeline for stage 4's 37-frame continuation query."""
    pipeline = [copy.deepcopy(dict(transform)) for transform in pipeline]
    found_state = found_action = False
    resized = []
    for transform in pipeline:
        transform_type = str(transform.get("type", ""))
        if transform_type.startswith("GetState") and "state_len" in transform:
            transform["state_len"] = int(num_frames)
            found_state = True
        elif (
            transform_type.startswith("GetAction")
            or transform_type == "LiberoDeltaToRelativeAction"
        ) and "action_len" in transform:
            transform["action_len"] = int(action_len)
            found_action = True
        if defer_relative and transform_type == "ToRelativeAction":
            continue
        resized.append(transform)
    if not (found_state and found_action):
        raise ValueError(
            "Stage 4 needs GetState*/GetAction* transforms with state_len/action_len."
        )
    return resized


def raw_action_len_for_model(action_len, model_fps, raw_action_fps):
    return int(round(action_len / model_fps * raw_action_fps))


@DATASETS.register_module()
class XEmbDataModule:
    """Window LeRobot data onto one clock, with optional stage-4 episode chains."""

    def __init__(
        self,
        dataset_list: List[Dict[str, Any]],
        batch_size: int,
        num_workers: int,
        num_frames: int,
        fps: float,
        action_dim: int,
        seed: int,
        stride: int,
        shuffle: bool = True,
        prefetch_factor: int = 8,
        train_ratio: float = 0.98,
        valid_ratio: float = None,
        decoder_cache_limit: int = 64,
        amortized_sampling: bool = False,
        split_by_episode: bool = False,
        continuation_num_frames: int = 0,
        defer_train_relative_action: bool = False,
        segment_curriculum=None,
    ):
        self.decoder_cache_limit = int(decoder_cache_limit)
        self.dataset_list = dataset_list
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.num_frames = int(num_frames)
        self.continuation_num_frames = int(continuation_num_frames or num_frames)
        self.sample_num_frames = max(self.num_frames, self.continuation_num_frames)
        self.sample_action_len = self.sample_num_frames - 1
        self.fps = float(fps)
        self.action_dim = int(action_dim)
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.persistent_workers = num_workers > 0
        self.train_ratio = float(train_ratio)
        self.valid_ratio = valid_ratio
        self.seed = int(seed)
        self.stride = int(stride)
        self.shuffle = bool(shuffle)
        self.amortized_sampling = bool(amortized_sampling)
        self.split_by_episode = bool(split_by_episode)
        self.defer_train_relative_action = bool(defer_train_relative_action)
        self.segment_curriculum = segment_curriculum
        if self.amortized_sampling and self.stride != 1:
            raise ValueError("Stage-4 amortized sampling requires stride=1.")
        if self.defer_train_relative_action and not self.amortized_sampling:
            raise ValueError("Deferred relative actions are only used by stage 4.")
        if self.segment_curriculum and not self.amortized_sampling:
            raise ValueError("A segment curriculum requires amortized sampling.")
        if self.train_ratio + float(self.valid_ratio or 0.0) > 1.0 + 1e-9:
            raise ValueError("train_ratio + valid_ratio must not exceed 1.")
        self._train_dataset = self._valid_dataset = None
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        del stage
        train_datasets, valid_datasets = [], []
        split_rng = random.Random(self.seed)
        for dataset_config in self.dataset_list:
            train_pipeline = dataset_config.train_pipeline
            valid_pipeline = dataset_config.valid_pipeline
            if self.amortized_sampling:
                train_pipeline = _resize_window_pipeline(
                    train_pipeline,
                    num_frames=self.sample_num_frames,
                    action_len=self.sample_action_len,
                    defer_relative=self.defer_train_relative_action,
                )
                valid_pipeline = _resize_window_pipeline(
                    valid_pipeline,
                    num_frames=self.sample_num_frames,
                    action_len=self.sample_action_len,
                    defer_relative=False,
                )
            fps = float(dataset_config.get("fps", self.fps))
            frame_ts = [index / fps for index in range(self.sample_num_frames)]
            raw_action_fps = dataset_config.get("raw_action_fps")
            if raw_action_fps is None:
                action_ts = frame_ts[: self.sample_action_len]
            else:
                raw_len = raw_action_len_for_model(
                    self.sample_action_len, fps, float(raw_action_fps)
                )
                action_ts = [index / float(raw_action_fps) for index in range(raw_len)]

            observation_keys = list(train_pipeline[0]["key_mapper"].keys())
            delta_timestamps = dict.fromkeys(observation_keys, frame_ts)
            if not bool(dataset_config.get("action_from_state", False)):
                for action_source in dataset_config.action_sources:
                    delta_timestamps[action_source] = action_ts
            tolerances = dict.fromkeys(
                dataset_config.repo_ids, dataset_config.get("tolerances_s", 0.0001)
            )
            dataset = DATASETS.build(
                dict(
                    # tolerances_s is expanded per repo below; leaving it in the
                    # config as well would pass the same keyword twice
                    **{k: v for k, v in dataset_config.items() if k != "tolerances_s"},
                    delta_timestamps=delta_timestamps,
                    tolerances_s=tolerances,
                    seed=self.seed,
                    stride=self.stride,
                    decoder_cache_limit=self.decoder_cache_limit,
                )
            )
            if self.split_by_episode:
                (
                    train_indices,
                    valid_indices,
                    train_episodes,
                    valid_episodes,
                    total_episodes,
                    dropped,
                ) = _split_indices_by_episode(
                    dataset,
                    self.train_ratio,
                    self.valid_ratio,
                    self.shuffle,
                    split_rng,
                )
                print(
                    f"[data] episode split: {train_episodes} train / "
                    f"{valid_episodes} valid of {total_episodes}; "
                    f"{dropped} boundary indices dropped"
                )
            else:
                indices = list(range(len(dataset)))
                if self.shuffle:
                    split_rng.shuffle(indices)
                train_len = int(len(indices) * self.train_ratio)
                valid_len = (
                    len(indices) - train_len
                    if self.valid_ratio is None
                    else min(
                        int(len(indices) * self.valid_ratio),
                        len(indices) - train_len,
                    )
                )
                train_indices = indices[:train_len]
                valid_indices = indices[train_len : train_len + valid_len]
            train_datasets.append(
                PipelineSubset(
                    dataset,
                    train_indices,
                    pipeline=train_pipeline,
                    preserves_episodes=self.split_by_episode,
                    actions_are_relative=(
                        bool(dataset_config.get("actions_are_relative", False))
                        if self.amortized_sampling
                        else None
                    ),
                )
            )
            valid_datasets.append(
                PipelineSubset(
                    dataset,
                    valid_indices,
                    pipeline=valid_pipeline,
                    preserves_episodes=self.split_by_episode,
                    actions_are_relative=(
                        bool(dataset_config.get("actions_are_relative", False))
                        if self.amortized_sampling
                        else None
                    ),
                )
            )
        self._train_dataset = AmortizedConcatDataset(train_datasets)
        self._valid_dataset = ConcatDataset(valid_datasets)
        print(
            f"[data] split: {len(self._train_dataset)} train / "
            f"{len(self._valid_dataset)} valid samples"
        )

    def train_dataloader(self):
        common = dict(
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
        if self.amortized_sampling:
            sampler = AmortizedBatchSampler(
                episode_runs(self._train_dataset),
                batch_size=self.batch_size,
                opening_horizon=self.num_frames,
                continuation_horizon=self.continuation_num_frames,
                fps=self.fps,
                shuffle=self.shuffle,
                seed=self.seed,
                segment_curriculum=self.segment_curriculum,
            )
            return DataLoader(self._train_dataset, batch_sampler=sampler, **common)
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
            **common,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=False,
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def on_exception(self, exception):
        pass

    def teardown(self, stage=None):
        pass
