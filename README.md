# Code for SyMFood: Synergistic Multi-Modal Prompting for Fine-Grained Zero-Shot Food Detection
## Overview
This repository serves as the official implementation for the paper:  
**"SyMFood: Synergistic Multi-Modal Prompting for Fine-Grained Zero-Shot Food Detection"**, published in *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2026*.

**Authors:** Xinlong Wang, Weiqing and Min, Shoulong Liu, Guorui Sheng, Shuqiang Jiang.

---

## 📢 News
* **[2026-02]** Our paper has been accepted by IEEE TCSVT! The code and models are now released.

## Data Preparation
SyMFood/
├── data/
├── dataset/
│   ├── images/
│   ├── zsd_annotations/
│   └── visual_prompts/
├── zsd_detr/
└── ...

## 🛠️ Environment Requirements

We strongly recommend using **Anaconda** to manage the environment. The dependencies are specified in `environment.yaml`.\

- Python 3.8
- PyTorch 1.12.1 + CUDA 11.3
- MMDetection (2.x) + MMCV (2.x)
- Detectron2 / Detrex (for feature extraction components)

## Create the environment
```bash
# Clone the repository
git clone [https://github.com/YourUsername/SyMFood.git](https://github.com/YourUsername/SyMFood.git)
cd SyMFood

# Create the environment from the yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate symfood  # Please check the name inside your yaml file

## 🚀 Training
We provide two ways to train the model. Please ensure you have properly set up the environment and dataset paths in zsd_detr/configs/config.py before running.

Option 1: Training with Shell Script (Recommended)
You can simply run the provided shell script:
'''bash
sh train.sh

Option 2: Training with Python Command
Alternatively, you can manually run the training script with specific arguments:
'''bash
python train_net.py --config-file zsd_detr/configs/config.py --num-gpus 1

Inference / Evaluation
To evaluate the model (Zero-Shot Detection performance) using the trained weights, run the following command:
'''bash
python train_net.py --eval_only True --config-file zsd_detr/configs/config.py --num-gpus 1


## Notes
- If you are using Docker, ensure CUDA 11.3 (or compatible) is available in the container.
- Update dataset and pretrained backbone paths in the config files before running any scripts.
- Full code and instructions will be released once the paper is accepted.

### Citation
@ARTICLE{11414166,
  author={Wang, Xinlong and Min, Weiqing and Liu, Shoulong and Sheng, Guorui and Jiang, Shuqiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={SyMFood: Synergistic Multi-Modal Prompting for Fine-Grained Zero-Shot Food Detection}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Visualization;Semantics;Zero shot learning;Videos;Grounding;Vectors;Object detection;Feature extraction;Training;Iterative methods;Food Computing;Zero-Shot Learning;Zero-Shot Detection;Cross-Modal Fusion;Food Detection},
  doi={10.1109/TCSVT.2026.3668226}
}
