# Robust 3D Face Reconstruction under Arbitrary Missing Data via Multi-Constraint Guidance

DOI: https://doi.org/10.5281/zenodo.19595194

This repository contains the official implementation of the research paper:  
**"Robust 3D Face Reconstruction under Arbitrary Missing Data via Multi-Constraint Guidance"** Currently **under review** at **The Visual Computer**.

## Introduction
This project introduces **MCG-Face**, a systemic framework designed for high-fidelity 3D facial reconstruction. Our method addresses the challenge of reconstructing faces with arbitrary missing data or irregular defects by leveraging **Graph Convolutional Networks (GCN)** and a **Multi-Constraint Guided** optimization strategy. It ensures both global structural robustness and local geometric accuracy.

## Repository Structure
```text
.
├── config/                # Configuration files (.py and .cfg)
├── examples/              # Sample 3D data (e.g., mask_610.obj) for testing
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
