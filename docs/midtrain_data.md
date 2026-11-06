# Pretraining data

The midtrain mix (`config/dataset/midtrain.yaml`) is built from the open-source
robot datasets below. Each row names the configs in
`config/dataset/midtrain_datasets/` that read it and where the raw data can be
downloaded.

Every dataset is read as **LeRobot v3.0**. Where the source ships an older
LeRobot version (v2.x) or a non-LeRobot format (HDF5, RLDS), it was converted to
v3.0 before training. The `root:` in each config is the path on our cluster; point
it at your own download.

| Dataset | Config(s) | Download | Source format |
|---|---|---|---|
| AgiBot World Beta | `agibot_world` | [agibot-world/AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | raw; converted to LeRobot v3.0 |
| AgiBot World 2026 | `agibot_world_2026_l1_static`, `agibot_world_2026_l1_mobile`, `agibot_world_2026_l2`, `agibot_world_2026_l3` | [agibot-world/AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | uses the `ImitationLearning/`, `RichInteraction/` and `ReinforcementLearning/` subsets; converted to LeRobot v3.0 |
| RoboCOIN | `robocoin_*` (16 configs) | [RoboCOIN on Hugging Face](https://huggingface.co/RoboCOIN), one repo per task (e.g. [RoboCOIN/Agilex_Cobot_Magic_connect_block](https://huggingface.co/datasets/RoboCOIN/Agilex_Cobot_Magic_connect_block)) | LeRobot v2.1; converted to v3.0. The config's `repo_ids` are the repo names |
| RoboMIND 2.0 | `robomind_agilex`, `robomind_agilex_mobile`, `robomind_ark`, `robomind_ark_mobile` | [RoboMIND 2.0 collection on ModelScope](https://modelscope.cn/collections/X-Humanoid/RoboMIND20) (project page: [x-humanoid-robomind.github.io](https://x-humanoid-robomind.github.io/)) | HDF5; converted to LeRobot v3.0 |
| DROID 1.0.1 | `droid` | [lerobot/droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1) | LeRobot v3.0 |
| BEHAVIOR Challenge 2026 demos | `challenge2026` | [behavior-1k/2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) | LeRobot v3.0 |
| RoboCasa365 | `robocasa_v30` (midtrain atomic + composite) | `python -m robocasa.scripts.download_datasets --split midtrain --source human` ([docs](https://robocasa.ai/docs/build/html/datasets/using_datasets.html)); also on Hugging Face as [robocasa365-midtrain-atomic](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-midtrain-atomic) and [robocasa365-midtrain-composite](https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-midtrain-composite) | LeRobot |
| InternData-M1 | `interndata_m1_agilex`, `interndata_m1_franka`, `interndata_m1_robotiq` | [InternRobotics/InternData-M1](https://huggingface.co/datasets/InternRobotics/InternData-M1) (`simulated/`) | LeRobot; converted to v3.0 |
| InternData-A1 | `interndata_a1_*` (16 configs: genie1 sim/real, franka, lift2, split_aloha) | [InternRobotics/InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1) (`sim/`, `sim_updated/`, `real/`; v3.0 copies in `sim_updated_lerobotv30/`, `real_lerobotv30/`) | LeRobot; genie1 real eef poses added by FK |
| Galaxea Open-World | `galaxea_openworld` | [OpenGalaxea/Galaxea-Open-World-Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) (`lerobot/`) | LeRobot (one `.tar.gz` per task); converted to v3.0 |
| Hy-Embodied-0.5 VLA | `hy_embodied_vla` | [tencent/Hy-Embodied-0.5-VLA-Data](https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data) | LeRobot v3.0 |
| BridgeData V2 | `bridge` | [IPEC-COMMUNITY/bridge_orig_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) | LeRobot v2.0; converted to v3.0 |
| RT-1 (Fractal) | `rt1` | [IPEC-COMMUNITY/fractal20220817_data_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) | LeRobot v2.0; converted to v3.0 |
| Kai0 | `kai0` | [OpenDriveLab-org/Kai0](https://huggingface.co/datasets/OpenDriveLab-org/Kai0) (`Task_{A,B,C}/base`) | LeRobot v2.1; converted to v3.0 |
| RoboTwin 2.0 | `robotwin` | [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) (`lerobot_dataset/RoboTwin_lerobot_v30.zip`) | LeRobot v3.0 |
| LIBERO | `libero` | IPEC-COMMUNITY: [libero_10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot), [libero_90](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_90_no_noops_lerobot), [libero_goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot), [libero_object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot), [libero_spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot) | LeRobot v2.1 (20 fps); converted to v3.0 |
| MimicGen | `mimicgen` | [amandlek/mimicgen_datasets](https://huggingface.co/datasets/amandlek/mimicgen_datasets) | robomimic HDF5; converted to LeRobot v3.0 |
| HiFi-UMI-2K | `hifi_umi_2k` | [simple-world-lab/HiFi-UMI-2K](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K) | LeRobot v3.0 |

Licences differ per dataset (several are non-commercial, e.g. Kai0 is
CC BY-NC-SA 4.0). Check each source's licence before use.

## Not open source

These configs are in the mix but read in-house data that is not publicly
available. Drop them from `dataset_list` in `config/dataset/midtrain.yaml` when
training on public data only:

- `g2_*`, `g2a_*`: in-house AgiBot G2 / G2A teleoperation data
- `xr2`: in-house XR2 / X2W data
