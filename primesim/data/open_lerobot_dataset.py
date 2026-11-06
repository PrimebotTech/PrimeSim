import json
import logging
import random
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import List

import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from mmengine import DATASETS
from torch.utils.data import ConcatDataset, Dataset

from .lerobot_patches import apply_lerobot_patches

logger = logging.getLogger(__name__)
apply_lerobot_patches()


@DATASETS.register_module()
class OpenLeRobotMixedDatasetV3(Dataset):
    def __init__(
        self,
        repo_ids: List[str],
        root: str | Path | None = None,
        stride: int = 1,
        seed: int = 42,
        decoder_cache_limit: int = 64,
        episodes: dict | None = None,
        pipeline: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        token: str | bool | None = None,
        **kwargs,
    ):
        super().__init__()
        root = Path(root) if root else HF_LEROBOT_HOME
        tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        self._check_features(root, repo_ids, delta_timestamps)
        datasets = []
        for repo_id in repo_ids:
            dataset = OpenLeRobotDatasetV3(
                repo_id,
                root=root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                pipeline=pipeline,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
                token=token,
            )
            print("Loaded dataset repo_id:", repo_id)
            datasets.append(dataset)

        self._dataset = ConcatDataset(datasets)
        self.stride = stride
        self.rng = random.Random(seed)

    @staticmethod
    def _check_features(root, repo_ids, delta_timestamps):
        if not delta_timestamps:
            return
        requested = set(delta_timestamps)
        bookkeeping = set(OpenLeRobotDatasetV3.BOOKKEEPING_COLUMNS)

        missing, declared = {}, {}
        for repo_id in repo_ids:
            info = root / repo_id / "meta" / "info.json"
            if not info.exists():
                missing[repo_id] = ["<no meta/info.json>"]
                continue
            with open(info) as fh:
                features = set(json.load(fh).get("features", {}))
            declared[repo_id] = features
            absent = requested - features - bookkeeping
            if absent:
                missing[repo_id] = sorted(absent)

        if missing:
            detail = "\n".join(
                f"    {r}: missing {k}" for r, k in sorted(missing.items())
            )
            raise ValueError(
                f"delta_timestamps requests keys that {len(missing)} of "
                f"{len(repo_ids)} repo_ids do not provide:\n{detail}"
            )
        if len(declared) > 1:
            common = set.intersection(*declared.values())
            uncommon = set().union(*declared.values()) - common
            if uncommon:
                logger.info(
                    "%d feature(s) not present in all repo_ids -- ignored, as "
                    "nothing requests them: %s",
                    len(uncommon),
                    sorted(uncommon),
                )

    def __len__(self):
        return len(self._dataset) // self.stride

    def __getitem__(self, idx: int, pipeline=None) -> dict[str, torch.Tensor]:
        # Strided sampling to traverse across the dataset quicker
        item = self._dataset[(idx * self.stride) + self.rng.randint(0, self.stride - 1)]
        return pipeline(item) if pipeline is not None else item


@DATASETS.register_module()
class OpenLeRobotDatasetV3(LeRobotDataset):
    BOOKKEEPING_COLUMNS = (
        "index",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    )

    def __init__(
        self,
        repo_id: str,
        root: str,
        max_retries=10,
        delta_timestamps=None,
        episodes=None,
        decoder_cache_limit=64,
        tolerance_s=0.0001,
        *args,
        **kwargs,
    ):
        super().__init__(
            root=root,
            repo_id=repo_id,
            image_transforms=None,
            delta_timestamps=delta_timestamps,
            episodes=episodes,
            tolerance_s=tolerance_s,
        )
        if self.meta.stats is None:
            raise ValueError(f"{repo_id} does not contain a stats.json file")
        self.task_map = self._add_tasks()
        self.max_retries = max_retries
        self.repo_id = repo_id
        self.decoder_cache_limit = int(decoder_cache_limit)

    def load_hf_dataset(self):
        import pyarrow.dataset as pa_ds
        from datasets import Dataset
        from lerobot.datasets.utils import (
            get_hf_features_from_features,
            hf_transform_to_torch,
        )

        if self.episodes is None:
            paths = sorted(
                str(path) for path in (self.root / "data").glob("*/*.parquet")
            )
            filters = None
        else:
            paths = sorted(
                {
                    str(self.root / self.meta.get_data_file_path(ep))
                    for ep in self.episodes
                }
            )
            filters = pa_ds.field("episode_index").isin(self.episodes)
        if len(paths) == 0:
            raise FileNotFoundError(
                f"Provided directory does not contain any parquet file: "
                f"{self.root / 'data'}"
            )

        columns = self._columns_to_load()
        features = get_hf_features_from_features(
            {key: self.meta.features[key] for key in columns}
        )
        hf_dataset = Dataset.from_parquet(
            paths, filters=filters, features=features, columns=columns
        )
        hf_dataset.set_transform(hf_transform_to_torch)

        return hf_dataset

    def _columns_to_load(self):
        keep = set(self.BOOKKEEPING_COLUMNS)
        keep |= (
            set(self.delta_timestamps)
            if self.delta_timestamps
            else set(self.meta.features)
        )

        unknown = keep - set(self.meta.features) - set(self.BOOKKEEPING_COLUMNS)
        if unknown:
            raise ValueError(
                f"{self.repo_id}: delta_timestamps asks for keys that are not features "
                f"of this dataset: {sorted(unknown)}"
            )

        return [
            key
            for key, ft in self.meta.features.items()
            if key in keep and ft["dtype"] != "video"
        ]

    def _add_tasks(self):
        tasks_path = self.root / "meta" / "tasks.parquet"

        if tasks_path.exists():
            tasks_df = pd.read_parquet(tasks_path)
            task_map = dict(zip(tasks_df["task_index"], tasks_df.index))
        else:
            task_map = {}
            print(
                f"Warning: tasks.parquet not found at {tasks_path}. 'task' column will "
                f"be empty."
            )

        return task_map

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return self._build_item(idx)
            except Exception as e:
                last_exc = e
                if attempt == 0:
                    print(
                        f"[CustomLerobotDatasetV3] failed on index {idx} "
                        f"(episode data error), resampling. First error: {repr(e)}"
                    )
                    traceback.print_exc()
                idx = random.randint(0, self.num_frames - 1)

        raise RuntimeError(
            f"Failed to load a valid sample after {self.max_retries} resampling "
            f"attempts. "
            f"Last error: {repr(last_exc)}"
        ) from last_exc

    def _build_item(self, idx) -> dict:
        item = self.hf_dataset[idx]
        episode_idx = item["episode_index"].item()
        curr_item = self._get_prompt(item)
        curr_item = self._get_frame(curr_item, episode_idx)
        curr_item["stats"] = dict(self.meta.stats)
        curr_item["repo_id"] = self.repo_id

        self._trim_decoder_cache()

        return curr_item

    def _trim_decoder_cache(self):
        if self.decoder_cache_limit <= 0:
            return
        try:
            from lerobot.datasets.video_utils import _default_decoder_cache as cache
        except ImportError:
            return
        try:
            if cache.size() > self.decoder_cache_limit:
                cache.clear()
        except AttributeError:
            print("cache.clear() didn't work")

    def _get_prompt(self, item):
        task_idx = int(item["task_index"].item())
        item["text"] = self.task_map.get(task_idx)
        return item

    def _get_frame(self, item, episode_idx) -> dict:
        query_indices, padding = self._get_query_indices(
            item["index"].item(), episode_idx
        )
        query_timestamps = self._get_query_timestamps(
            item["timestamp"].item(), query_indices
        )
        if self.delta_timestamps:
            query_timestamps = {
                k: v for k, v in query_timestamps.items() if k in self.delta_timestamps
            }
        query_result = self._query_hf_dataset(query_indices)
        item = {**item, **padding, **query_result}

        if len(self.meta.video_keys) > 0:
            video_frames = self._query_videos(query_timestamps, episode_idx)
            item = {**video_frames, **item}

        return item
