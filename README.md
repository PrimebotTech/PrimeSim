<div align="center">

<img src="assets/primesim_promo.gif" width="520" alt="PrimeSim">

</div>

<div align="center">

  <a href="https://arxiv.org/abs/2611.04870"><img src="https://img.shields.io/badge/arXiv-2611.04870-b31b1b.svg" alt="arXiv"></a>
  <a href="https://primebottech.github.io/PrimeSim-website/"><img src="https://img.shields.io/badge/Project%20page-primebottech.github.io-2ea44f" alt="Project page"></a>
  <a href="https://huggingface.co/qiyuan-robot/PrimeSim"><img src="https://img.shields.io/badge/Weights-Hugging%20Face-ffd21e" alt="Weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>

</div>

# PrimeSim



PrimeSim is a generalist, action-conditioned, multiview robot world model built for
robot simulation, trained on 30,110 hours of cross-embodiment robot demonstration
data. Given the first frame of each camera view, a prompt and a robot action
trajectory, it generates the video of what happens next.

## 🔥 Highlights

- **General multi-embodiment representation.** Every robot's state and actions map
  into one 31-dim layout (end-effector pose and gripper for each arm, torso, head,
  waist and base velocity), with the slots a robot doesn't have left at zero, so
  single-arm, bimanual and humanoid data train one model. Configs ship for Libero,
  RoboTwin, AgiBot World and G2; `config/dataset/midtrain.yaml` picks the mix.
- **Multiview adaptation.** Three camera views are generated together, stacked in one
  latent. Each view restarts the positional encoding and gets a learned view embedding;
  self-attention stays within each view and spans all views every third block.
- **Controllability with robot actions.** Every four action steps become one context
  token that each latent frame attends to alongside the text, bound by a framewise mask
  to the action chunk that produced that frame, with a residual per-frame action
  adapter.
- **A comprehensive post-training pipeline.** The midtrained model is made
  autoregressive, distilled into a few-step generator, refined with asymmetric DMD,
  and extended to long horizons with cross-window Self-Forcing++.

## 🕹️ Quick Start

### Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .            # or: pip install -e .
```

Optional, faster attention (picked up automatically): `uv pip install -e ".[flash-attn]"`
(also `[sage]`, `[xformers]`).

### Model files

PrimeSim uses the text encoder (umt5-xxl), tokenizer and VAE of Wan2.2-TI2V-5B.
Download just those (about 14 GB) from Hugging Face:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.2_VAE.pth" "google/umt5-xxl/*" \
  --local-dir checkpoints
```

and point `config/model/primesim.yaml` at them:

```yaml
model:
  text_encoder_path: checkpoints/models_t5_umt5-xxl-enc-bf16.pth
  vae_path: checkpoints/Wan2.2_VAE.pth
  tokenizer_path: checkpoints/google/umt5-xxl
```

Each launch sets the PrimeSim checkpoint it starts from (`model.checkpoint_path`), and
the datasets live in `config/dataset/`.

### Inference

```bash
python scripts/inference.py \
  --checkpoint /path/to/stage4.safetensors \
  --text-encoder /path/to/models_t5_umt5-xxl-enc-bf16.pth \
  --vae /path/to/Wan2.2_VAE.pth \
  --tokenizer /path/to/umt5-xxl \
  --demo-dir example_data/g2_example_1 \
  --output outputs/g2_example_1.mp4
```

### Training

Every stage is one command and starts from the checkpoint set in its launch config.

The midtraining pipeline itself is included, not just the fine-tuning stages. The
full recipe -- data loading, the 31-dim cross-embodiment slot layout, midtraining
on the mixed corpus and the posttraining chain -- is `config/dataset/midtrain.yaml`
together with the dataset configs it pulls in, and `launch=midtrain` runs it end to
end. Point it at your own LeRobot v3 datasets and it will midtrain a model from
scratch rather than fine-tuning an existing one; `docs/midtrain_data.md` lists the
corpus we used and `docs/adapting_a_dataset.md` covers adapting your own.

| Stage | Launch | Starts from | Trains |
| --- | --- | --- | --- |
| Fine-tuning | `launch=finetune` | a midtrained checkpoint | the bidirectional world model (flow matching) |
| Stage 1 | `launch=stage1_ar` | the bidirectional model | an autoregressive, chunk-causal model (teacher forcing) |
| Stage 2 | `launch=stage2_cd` | stage 1 | a few-step model (causal consistency distillation) |
| Stage 3 | `launch=stage3_dmd` | stage 2, plus the bidirectional model for the scores | the few-step model, refined with asymmetric DMD |
| Stage 4 | `launch=stage4_sf` | stage 3, plus the bidirectional model for the scores | long-horizon self-rollout training on episode subchains from the dataset you select |

```bash
accelerate launch scripts/train.py launch=finetune dataset=your_dataset
```

`dataset=your_dataset` is required: the fine-tuning launch has no default dataset, and
the choice picks a config from `config/dataset/`. Two worked examples ship with it:

- **`config/dataset/robotwin.yaml`** (RoboTwin 2.0) is the one to copy. It carries its
  whole pipeline inline -- the refactors, the state/action slots, the image resize,
  the multiview stack and the `Gather` that assembles the batch -- so it shows every
  transform a real dataset needs, in order.
- **`config/dataset/libero.yaml`** also defines its training and validation pipelines
  inline. It shows single-arm data handling, repeats the available camera to fill
  three views, and converts LIBERO's delta actions into relative actions at the
  model's frame rate.

For your own data, **[docs/adapting_a_dataset.md](docs/adapting_a_dataset.md)** walks
through building the pipeline: converting to LeRobot v3, mapping your state and action
columns into the shared 31-dim slot layout, and getting the frame rate right.

On several GPUs, add the usual `accelerate` flags, e.g.
`accelerate launch --num_processes 8 scripts/train.py launch=stage1_ar dataset=your_dataset`.

Each launch must be given the checkpoint it starts from: set `checkpoint_path` in
its yaml under `config/launch/`, or pass it on the command line. The DMD and
Self-Forcing stages also need `score_checkpoint`, the bidirectional model used for
the scores, which is a different file from the one being trained.

Checkpoints are written to `models/<run_name>/step-<N>.safetensors`. Each is a plain
model checkpoint that the next stage can start from. Sample videos and metrics are
logged to Weights & Biases.

## ⭐ Acknowledgements

PrimeSim builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

## ⚖️ License

Apache-2.0, see [LICENSE](LICENSE).
