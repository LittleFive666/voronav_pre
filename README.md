# VoroNav: Vision-Language Object Navigation

A vision-language object navigation system that enables autonomous agents to navigate indoor environments and locate target objects using semantic mapping, visual reasoning, and intelligent path planning.

## Overview

VoroNav is an end-to-end framework for object-goal navigation in photorealistic indoor environments. The system combines semantic mapping, vision-language models, and exploration strategies to enable agents to understand natural language object queries, build spatial representations of environments, and efficiently navigate to target objects.

The system leverages state-of-the-art vision models (Grounding DINO, Segment Anything) for visual understanding and employs Fast Marching Method (FMM) based planning for collision-free navigation in complex indoor scenes.

## Key Features

- **Semantic Mapping**: Builds multi-channel 2D top-down semantic maps from RGB-D observations with obstacle detection and explored area tracking
- **Vision-Language Integration**: Uses BLIP for image captioning and scene understanding, Grounding DINO for object detection, and SAM for segmentation
- **Intelligent Path Planning**: FMM-based planner generates collision-free paths to exploration frontiers and detected target objects
- **Habitat Simulator Integration**: Full support for HM3D dataset with realistic physics and sensor simulation
- **Frontier-Based Exploration**: Automatically identifies and prioritizes unexplored regions for efficient environment coverage
- **Multi-Modal Reasoning**: Combines visual observations, semantic predictions, and spatial memory for navigation decisions
- **Comprehensive Evaluation**: Tracks Success Rate, SPL, SCA (collision avoidance), and SEA (exploration efficiency) metrics

## Installation

The code has been tested only with Python 3.7 on Ubuntu 20.04.

### 1. Installing Dependencies

We use challenge-2022 versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab).

#### Installing habitat-sim:
```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout tags/challenge-2022
pip install -r requirements.txt
python setup.py install --headless
# For Mac OS, use: python setup.py install
```

#### Installing habitat-lab:
```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/challenge-2022
pip install -e .
```

#### Install PyTorch
Install [PyTorch](https://pytorch.org/) according to your system configuration. The code is tested on PyTorch v1.7.0 and CUDA toolkit v11.4:

```bash
# Linux with GPU
conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=11.4

# Mac OS
conda install pytorch==1.7.0 torchvision==0.8.1 -c pytorch
```

#### Install Detectron2
Install [Detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration.

### 2. Clone Repository and Install Requirements

```bash
git clone https://github.com/[your-username]/VoroNav.git
cd VoroNav
pip install -r requirements.txt
```

### 3. Download HM3D Dataset

Download [HM3D](https://aihabitat.org/datasets/hm3d/) dataset using the download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/datasets/HM3D.md):

```bash
python -m habitat_sim.utils.datasets_download \
    --username <api-token-id> \
    --password <api-token-secret> \
    --uids hm3d_minival
```

### 4. Download Segmentation Model

Download the [segmentation model](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view?usp=share_link) and place it in the `RedNet/model` path.

### 5. Setting Up Dataset Directory Structure

The code requires the datasets in a `data` folder in the following format (same as habitat-lab):

```
VoroNav/
  data/
    scene_datasets/
    matterport_category_mappings.tsv
    object_norm_inv_perplexity.npy
    versioned_data/
    objectgoal_hm3d/
        train/
        val/
        val_mini/
```

### 6. Install Grounding DINO and Segment Anything

#### Set Environment Variables

If you want to build a local GPU environment for Grounded-SAM, set the environment variables:

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

#### Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

#### Install Grounding DINO:

```bash
pip install --no-build-isolation -e GroundingDINO
```

## System Architecture

The system consists of several interconnected components:

### Core Modules

- **Semantic Mapping Module** (`model.py`): 
  - Processes RGB-D observations into voxel-based representations
  - Projects 3D point clouds into 2D semantic maps
  - Maintains obstacle maps, exploration maps, and semantic category channels
  - Handles stair detection and traversability analysis

- **Navigation Agent** (`agents/sem_exp.py`):
  - Implements exploration and goal-seeking behaviors
  - Manages frontier detection and goal selection
  - Integrates vision-language models for object detection
  - Coordinates with the planner for action execution

- **Environment Interface** (`envs/habitat/`):
  - Wraps Habitat simulator for HM3D environments
  - Provides standardized observation and action spaces
  - Handles episode management and metrics tracking

- **Vision Models**:
  - **GroundingDINO**: Open-vocabulary object detection
  - **Segment Anything (SAM)**: Instance segmentation
  - **BLIP**: Image captioning and visual reasoning

- **Planning System** (`envs/utils/fmm_planner.py`):
  - Fast Marching Method for optimal path computation
  - Handles dynamic obstacle avoidance
  - Computes traversability costs

## Project Structure

```
voronav/
├── main.py                      # Main training/evaluation loop
├── model.py                     # Semantic mapping module
├── arguments.py                 # Configuration parameters
├── constants.py                 # Object categories and scene definitions
├── chat.py                      # Vision-language integration
│
├── agents/
│   ├── sem_exp.py              # Semantic exploration agent
│   └── utils/
│       ├── semantic_prediction.py   # MaskRCNN predictions
│       └── visualization.py         # Visualization utilities
│
├── envs/
│   ├── habitat/
│   │   ├── objectgoal_env.py   # Habitat environment wrapper
│   │   ├── configs/            # Task configurations
│   │   └── utils/
│   │       └── vector_env.py   # Vectorized environments
│   └── utils/
│       ├── depth_utils.py      # Depth processing utilities
│       ├── fmm_planner.py      # Fast Marching Method planner
│       ├── map_builder.py      # Map construction utilities
│       └── pose.py             # Pose transformations
│
├── GroundingDINO/              # Object detection model
├── segment_anything/           # Segmentation model
│
├── utils/
│   ├── storage.py              # Rollout storage for training
│   └── model.py                # Neural network utilities
│
└── docs/                       # Documentation and examples
    ├── INSTRUCTIONS.md
    ├── DOCKER_INSTRUCTIONS.md
    ├── example.gif
    └── legend.png
```

## Core Components

### Semantic Mapping

The semantic mapping module builds comprehensive spatial representations of the environment:

- **Multi-Channel Maps**: Each map contains obstacle, explored area, agent pose, and semantic category channels
- **Height Projection**: 3D voxel grids are projected to 2D using configurable height ranges
- **Dynamic Updates**: Maps are incrementally updated as the agent explores
- **Coordinate Transformations**: Handles conversion between agent-centric and global coordinate frames

Key parameters:
- Map resolution: 5cm per cell (configurable)
- Map size: 48m × 48m (4800cm)
- Vision range: 100 cells in agent's field of view

### Object Detection and Recognition

The system employs a multi-stage vision pipeline:

1. **Scene Understanding**: BLIP generates natural language descriptions of observations
2. **Object Localization**: Grounding DINO detects target objects with bounding boxes
3. **Instance Segmentation**: SAM refines object boundaries for precise localization
4. **Semantic Mapping**: Detected objects are projected onto the semantic map

### Path Planning and Exploration

Navigation combines goal-oriented and exploratory behaviors:

- **FMM Planner**: Computes optimal paths considering obstacles and traversability
- **Frontier Detection**: Identifies boundaries between explored and unexplored regions
- **Goal Selection**: Prioritizes frontiers based on area, distance, and exploration value
- **Collision Handling**: Detects and recovers from collisions dynamically

### Stair Detection and Handling

Special handling for multi-floor navigation:
- Height-based stair detection from depth observations
- Dynamic obstacle map clearing during stair traversal
- Prevents false obstacles from stair geometry

## Configuration

Key configuration parameters in `arguments.py`:

### Environment Settings
- `--env_frame_width/height`: Environment frame dimensions (640×480)
- `--frame_width/height`: Processing frame dimensions (160×120)
- `--max_episode_length`: Maximum steps per episode (600)
- `--hfov`: Horizontal field of view (79°)
- `--turn_angle`: Agent rotation angle (30°)
- `--camera_height`: Camera height in meters (0.88m)

### Mapping Settings
- `--map_size_cm`: Global map size (4800cm = 48m)
- `--map_resolution`: Map cell resolution (5cm)
- `--vision_range`: Local map vision range (100 cells)
- `--global_downscaling`: Downscaling factor for global map (2)

### Task Settings
- `--task_config`: Path to Habitat task config
- `--split`: Dataset split (train/val/val_mini)
- `--success_dist`: Success distance threshold (1.0m)
- `--num_eval_episodes`: Episodes per evaluation (201)

### Visualization
- `--visualize`: Render observations and maps (0/1/2)
- `--print_images`: Save visualization images (0/1)

Configuration files are located in `envs/habitat/configs/tasks/`.

## Usage Examples

### Evaluation

Run evaluation on HM3D validation set:

```bash
python main.py \
    --split val \
    --eval 1 \
    --num_eval_episodes 200 \
    --task_config envs/habitat/configs/tasks/objectnav_hm3d.yaml \
    --visualize 2 \
    --print_images 1
```

### Training

Train the navigation agent:

```bash
python main.py \
    --split train \
    --eval 0 \
    --num_training_frames 400000 \
    --exp_name my_experiment \
    --log_interval 10
```

### Visualization

Enable visualization to see real-time navigation:

- `--visualize 1`: Show observations and predicted semantic map
- `--visualize 2`: Show observations with semantic predictions and map
- `--print_images 1`: Save visualization frames to disk

Output is saved to `tmp/dump/{exp_name}/` directory.

## Evaluation Metrics

The system tracks comprehensive navigation metrics:

### Success Rate
Percentage of episodes where the agent reaches within `success_dist` of the target object.

### SPL (Success weighted by Path Length)
```
SPL = (1/N) * Σ Sᵢ * (Lᵢ / max(Pᵢ, Lᵢ))
```
Where:
- `Sᵢ`: Success indicator for episode i
- `Lᵢ`: Shortest path length to goal
- `Pᵢ`: Actual path length taken

### SCA (Success weighted by Collision Avoidance)
```
SCA = Success * (1 - collisions/forward_actions)
```
Penalizes episodes with collisions.

### SEA (Success weighted by Exploration Area)
```
SEA = min(Success * (0.002 * √(explored_area/path_length)), 1)
```
Rewards efficient exploration strategies.

Results are logged in `gt.txt` and JSON files in the dump directory.

## Supported Object Categories

The system supports navigation to 15 object categories from the COCO dataset:

| Category | ID | Category | ID | Category | ID |
|----------|----|---------|----|----------|---|
| chair | 0 | bed | 3 | tv | 5 |
| sofa | 1 | toilet | 4 | dining-table | 6 |
| plant | 2 | | | | |

Additional categories: oven (7), sink (8), refrigerator (9), book (10), clock (11), vase (12), cup (13), stairs (14)

Target categories for evaluation: chair, bed, plant, toilet, tv_monitor, sofa

Scene datasets:
- **Training**: 25 scenes (Allensville, Beechwood, Benevolence, ...)
- **Validation**: 5 scenes (Collierville, Corozal, Darden, Markleeville, Wiconisco)

## Visualization

The system provides rich visualization capabilities:

- **RGB Observations**: Current agent view
- **Semantic Predictions**: Object detection overlays
- **Top-Down Map**: Multi-channel semantic map showing:
  - Obstacles (black)
  - Explored areas (gray)
  - Agent location and trajectory (green)
  - Target object locations (colored by category)
  - Frontier goals (yellow markers)
  - Planned path

Example visualizations are available in the `docs/` folder:
- `docs/example.gif`: Example navigation episode
- `docs/overview.jpg`: System overview
- `docs/legend.png`: Map visualization legend

## Dataset

The system uses the HM3D (Habitat-Matterport 3D) dataset:
- Photorealistic 3D reconstructions of real indoor environments
- RGB-D observations from realistic camera sensors
- Object-goal navigation tasks with target object annotations
- Support for both training and evaluation splits

Refer to `docs/INSTRUCTIONS.md` for dataset setup details.

## Implementation Details

### Map Representation

Maps are represented as multi-channel tensors:
- **Channel 0**: Obstacle map (1 = obstacle, 0 = free space)
- **Channel 1**: Explored area map (1 = explored, 0 = unexplored)
- **Channel 2**: Current agent location
- **Channel 3**: Past agent locations (trajectory)
- **Channels 4+**: Semantic category channels (one per object category)

Map coordinates:
- Global map: Full environment representation (480×480 cells at 5cm resolution)
- Local map: Agent-centric view (240×240 cells) extracted from global map
- Automatic boundary management as agent moves

### Frontier Detection

Frontiers are detected using morphological operations:
1. Dilate explored area to find boundaries
2. Subtract obstacles from dilated area
3. Find connected components of frontier pixels
4. Filter by minimum area threshold (4+ pixels)
5. Rank by combined area and distance score

### Action Space

The agent has 4 low-level actions:
- **Move Forward**: Advance 0.25m in current direction
- **Turn Left**: Rotate 30° counter-clockwise
- **Turn Right**: Rotate 30° clockwise
- **Stop**: End episode (when goal is reached)

High-level planner decomposes long-term goals into sequences of low-level actions.

## Performance Tips

- Use `--num_processes` to parallelize evaluation across multiple environments
- Enable `--auto_gpu_config` for automatic GPU memory management
- Adjust `--vision_range` to trade off between map detail and computation
- Use `--global_downscaling` to reduce memory for large environments

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{voronav2024,
  title={VoroNav: Vision-Language Object Navigation},
  author={[Authors]},
  booktitle={[Conference]},
  year={2024}
}
```

## References

This project builds upon several open-source components:

- **Habitat**: [https://aihabitat.org/](https://aihabitat.org/)
  - AI Habitat simulator for embodied AI research
  
- **Grounding DINO**: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
  - Open-vocabulary object detection
  
- **Segment Anything (SAM)**: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
  - Foundation model for image segmentation

- **HM3D Dataset**: [https://aihabitat.org/datasets/hm3d/](https://aihabitat.org/datasets/hm3d/)
  - Habitat-Matterport 3D Research Dataset

## License

This project contains code from multiple sources with different licenses. Please refer to individual component licenses:
- GroundingDINO: Apache License 2.0
- Segment Anything: Apache License 2.0
- Habitat-related components: MIT License

## Acknowledgments

We thank the authors of Habitat, Grounding DINO, and Segment Anything for their excellent open-source contributions that made this work possible.

