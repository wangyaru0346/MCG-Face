# Multi-Constraint Guided Reconstruction of Complete 3D Facial Geometry from Arbitrarily Incomplete Scans

DOI: https://doi.org/10.5281/zenodo.19595194

This repository contains the official implementation of the research paper:  
**"Multi-Constraint Guided Reconstruction of Complete 3D Facial Geometry from Arbitrarily Incomplete Scans"** Currently **under review** at **The Visual Computer**.

## Introduction
This project introduces **MCG-Face**, a systemic framework designed for high-fidelity 3D facial reconstruction. Our method addresses the challenge of reconstructing faces with arbitrary missing data or irregular defects by leveraging **Graph Convolutional Networks (GCN)** and a **Multi-Constraint Guided** optimization strategy. It ensures both global structural robustness and local geometric accuracy.

## Repository Structure
```text
.
├── config/                # Configuration files (.py and .cfg)
├── examples/              # Sample 3D data (e.g., mask_610.obj) for testing
├── scripts/
│   └── evaluation/
│       ├── pipeline.py    # Prepare representative evaluation folders and metadata
│       └── calc_metrics.py # Compute CD, HD, RMSE, mean/std, and missing-ratio breakdowns
├── ablation_study.py      # Script for reproducing ablation experiments
├── baselines.py           # Implementation of comparison methods
├── extra_baselines.py     # Additional comparative models
├── funcs.py               # Geometric utility functions and loss calculations
├── models.py              # MCG-Face network architectures (GCN-based)
├── my_dataset.py          # Data loading logic for FaceScape dataset
├── reconstruction.py      # Inference script for face reconstruction
├── train.py               # Main training script
├── requirements.txt       # Environment dependencies
└── LICENSE                # MIT License
```
## Installation

### 1. Requirements
- **OS**: Linux
- **Python**: 3.10
- **CUDA**: 12.1

### 2. Setup Environment
```bash
conda create -n mcgface python=3.10
conda activate mcgface
pip install -r requirements.txt
```
### 3. Install PyTorch3D
Due to the sensitivity of PyTorch3D versions, we recommend installing the specific wheel used in our experiments:
```
pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl
```
## Dataset & Checkpoints
### FaceScape Dataset
We utilize the FaceScape dataset for our research.

- **Access**: Due to license restrictions, we cannot distribute the raw data. Please apply for authorized access at the FaceScape Official Site.

- **Processing**: Once authorized, place your processed ```.pt``` files in the ```dataset/``` directory as required by ```my_dataset.py```.
### Pre-trained Models
Due to size limitations on GitHub, the ```checkpoints/``` directory (including the crucial norm.pt and trained weights) is hosted on a cloud drive.

- **Baidu Netdisk**: [https://pan.baidu.com/s/157O3JuLoVzvJYRjanNk3xw?pwd=6688] (Extraction Code: [6688])

**Note**: Please ensure the ```checkpoints/``` folder is placed in the project root before running the scripts.
## Usage
### Training
To train the model using the multi-constraint guidance framework:
```
python train.py --config config/config.cfg
```
### Reconstruction
To reconstruct a 3D face from a partial or defective input:
```
python reconstruction.py --input examples/mask_610.obj
```
### Evaluation
To run the ablation studies described in the paper:
```
python ablation_study.py
```
Additional scripts for the revised quantitative evaluation are provided in ```scripts/evaluation/```.

Prepare representative evaluation folders
The script ```pipeline.py``` prepares representative evaluation folders and metadata for the revised quantitative comparison. It organizes the required files for each test case, including ground truth meshes, incomplete inputs, reconstructed results, and baseline outputs when available.
```
python scripts/evaluation/pipeline.py \
    --processed_dir data/processed_masks \
    --source_result_dir results_comparison_more30 \
    --output_dir results_comparison_more40 \
    --num_cases 40 \
    --start_index 801
```
Expected output structure:
```text
results_comparison_more40/
├── result_mask_801/
│   ├── 0_ground_truth.obj
│   ├── 1_input_hole.obj
│   ├── 2_reconstructed.obj
│   ├── baseline_stage1.obj
│   ├── baseline_symmetry.obj
│   ├── baseline_mean.obj
│   └── baseline_geometric.obj
├── result_mask_802/
│   └── ...
└── missing_metadata.csv
```
The geometric baseline generation in ```pipeline.py``` is provided as a fallback and can be replaced by a project-specific geometric filling implementation.

Compute quantitative metrics
The script ```calc_metrics.py``` computes Chamfer Distance (CD), Hausdorff Distance (HD), and RMSE for reconstructed 3D facial meshes. It reports per-sample metrics, mean and standard deviation for each method, and missing-ratio / missing-pattern breakdowns for MCG-Face.
```
python scripts/evaluation/calc_metrics.py \
    --result_dir results_comparison_more40 \
    --scale 100.0
```
The script exports:
```
metrics_per_sample.csv
metrics_mean_std.csv
ours_missing_ratio_breakdown.csv
ours_missing_pattern_breakdown.csv
```
Please modify dataset paths and result paths according to your local environment. Restricted raw facial scans are not redistributed in this repository.
## Citation
If you find our work or code helpful, please cite:
```
@article{Wang2026Robust3DFace,
  title={Robust 3D Face Reconstruction under Arbitrary Missing Data via Multi-Constraint Guidance},
  author={Wang, Yaru and Wu, Dongsheng and Cheng, Yifan and Niu, Qianqian and Wang, Xin and Zhou, Zhijie},
  journal={Submitted to The Visual Computer},
  year={2026},
  doi={10.5281/zenodo.19595194},
  note={Under review}
}
```
## Contact
Yaru Wang (First Author): [18766270346@163.com]

Dongsheng Wu (Principal Corresponding Author): [wuds@sylu.edu.cn]

Yifan Chen  (Corresponding Author): [yifan.chen@port.ac.uk]
