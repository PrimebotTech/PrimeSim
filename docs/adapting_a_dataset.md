# Adapting your own dataset

PrimeSim trains on **LeRobot v3** datasets. You do not write a new dataloader: you
convert your data to v3, describe it in a config, and say how its state/action
columns fill the shared 31-dim slot layout.

## 1. Put your data in LeRobot v3

Each dataset is one directory holding a `meta/` folder, per-episode `parquet` rows,
and the videos those rows point at. The schema is `preview-2` era v3: video keys
live in `metadata.json`, and frames are addressed by timestamp rather than by an
explicit index.

The practical requirement, in order:

- **`meta/info.json`** carries `fps`, `robot_type` and the feature list.
- **Every observation key you intend to use must be declared**, otherwise it is
  dropped on load with a `not present in all repo_ids` warning.
- **`fps` must be the true capture rate.** The loader converts frame windows to
  seconds as `i / fps`, so a wrong value produces offsets that are not whole
  frames and LeRobot rejects them (`check_delta_timestamps`).

If you have raw robot logs, convert them with LeRobot's own writer rather than
building the directory by hand — the interleaving of video chunks and parquet
rows is easy to get subtly wrong.

## 2. Describe it in a config

Add `config/dataset/midtrain_datasets/<your_dataset>.yaml` and enable it in
`config/dataset/midtrain.yaml` under `defaults:`, using the same
`midtrain_datasets/<name>@<alias>` form as the existing entries.

The fields that matter:

```yaml
root: /path/to/root/dir        # parent directory holding the repo_ids
repo_ids:
  - your_dataset_name          # subdirectory names under root
train_pipeline: ${..train_pipeline}
valid_pipeline: ${..valid_pipeline}
```

**`fps` is set once for the whole run**, in the launch file alongside `num_frames`
(`config/launch/*.yaml`, default `10`), and every dataset is read at that rate.
Your dataset does not declare its own — the loader resamples. If your data was
captured at 30 fps and the run says `fps: 10`, each training sample spans three
times as much wall-clock time as the numbers suggest, which changes both motion
speed and the meaning of `stride`. Match them, or accept the resampling knowingly.

Set `num_frames = 4k + 1` and `action_len = num_frames - 1`. The VAE keeps the first
frame and compresses every four subsequent frames into one latent frame. The
current launches use 33 frames and 32 action steps; stage 4 uses 37-frame
continuation windows, with 36 action steps per continuation.

## 3. Fill the state/action slots

State and action share one **31-dim layout** across every robot. A model trained
on it can be fed any platform, provided your columns land in the right slices.
`SLOT_OFFSETS` in `primesim/data/pipeline/loading_x_embodiment.py` is the source
of truth; the slices tile `[0, 31)` exactly, with no gap and no overlap.

### The common case: one wide column

If your dataset already stores a wide state vector and a wide action vector, use
`WideVectorWriter`. It copies a source column into the slots you name:

```yaml
- type: WideVectorWriter
  key_mapper:
    state.your_wide_state: state
  source_cols: [left_eef, left_gripper, right_eef, right_gripper]
```

Anything a slot does not cover is left at zero, so a robot with fewer joints than
a slot is wide still trains.

### The custom case: write the slots yourself

Subclass `SlotWriter` and implement `_write`:

```python
class GetStateMyRobot(SlotWriter):
    def _write(self, slots, slot_name, block, source=""):
        # slots[slot_name] is (traj_len, 31); block is your raw column
        slots[slot_name][:, 0:7] = quat_wxyz_to_xyzw(block[:, 3:10])
        slots[slot_name][:, 7] = block[:, 10]
        ...
```

Two invariants the base class checks for you:

- **Convention is xyzw everywhere.** LeRobot stores quaternions `wxyz`; call
  `quat_wxyz_to_xyzw` rather than reordering by hand.
- **One owner per slot.** Two writers touching the same slice raises, because
  the later one silently wins otherwise.

### Validation

`SlotWriter` asserts the slots tile the full 31 dims. If your writer leaves a gap
the error names the cursor position, which is the fastest way to find the missing
block.

## 4. Dataset statistics and normalisation

Keep the dataset statistics generated during LeRobot v3 conversion: the loader
requires them and attaches them to each sample as `stats`. The supplied LIBERO
and RoboTwin pipelines use explicit state/action transforms and do not apply
per-slot mean/std normalisation. Preserve the units and action conventions
expected by the checkpoint you are adapting.

`ImgNormalize` maps image tensors from `[0, 1]` to `[-1, 1]`. If you add state/action
normalisation for a custom training setup, implement it explicitly and apply the
same transformation during training and inference.

## 5. Check it before training

```bash
python scripts/train.py launch=finetune dataset=your_dataset
```

Watch the first few batches for:

- `N feature(s) not present in all repo_ids` — a key in your pipeline that the
  dataset does not declare
- a `delta_timestamps` error — the pipeline window is not a whole number of
  frames at the configured `fps`
- a slot tiling assertion — a gap or overlap in your writer

## Gotchas

**Multi-view is ordered.** Views are concatenated as `(v, t, h, w)`, so the order
of `repo_ids` and of camera keys decides which camera the model calls view 0.
Keep it stable across datasets or the view embeddings mean different things per
platform.

**One fps for the whole run.** `fps` is a single launch-level value, so a 20 fps
dataset and a 30 fps dataset in the same job are both read at that one rate. The
loader resamples rather than erroring, so a mismatch shows up as motion that is
subtly too fast or too slow rather than as a crash — check it deliberately.
