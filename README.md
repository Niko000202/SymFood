# SyMFood: Synergistic Multi-Modal Prompting for Fine-Grained Zero-Shot Food Detection

This repository serves as the official implementation for the paper:  
**"SyMFood: Synergistic Multi-Modal Prompting for Fine-Grained Zero-Shot Food Detection"**, published in *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2026*.

**Authors:** Xinlong Wang, Weiqing Min, Shoulong Liu, Guorui Sheng, Shuqiang Jiang.

---

## 📢 News
* **[2026-02]** The code and models are now released.
* **[2026-02]** 🎉🎉Our paper has been accepted by IEEE TCSVT!

## 📂 Data Preparation

### 1. Dataset (MS COCO 2014)
We use the **MS COCO 2014** dataset for training and evaluation. Please download the images and annotations from the [official website](http://cocodataset.org/#download).

Please organize your dataset directory as follows:
```text
SyMFood/
├── data/
├── dataset/
│   ├── images/
│   │   ├── train2014/      <-- MS COCO 2014 Train Images
│   │   └── val2014/        <-- MS COCO 2014 Val Images
│   ├── zsd_annotations/    <-- Our provided annotations
│   └── visual_prompts/
├── zsd_detr/
└── ...

```

2. Pretrained Backbone
We use ResNet-50 pre-trained on ImageNet as the visual backbone, following the Detectron2 format.

Please download the weights and place them in your pretrained models folder (or update the path in config.py):

ResNet-50 (Detectron2 format): [Download Link](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl)



## 🛠️ Environment Requirements

We strongly recommend using **Anaconda** to manage the environment. The dependencies are specified in `environment.yaml`.

* Python 3.8
* PyTorch 1.12.1 + CUDA 11.3
* MMDetection (2.x) + MMCV (2.x)
* Detectron2 / Detrex (for feature extraction components)

### Installation

```bash
# Clone the repository
git clone [https://github.com/YourUsername/SyMFood.git](https://github.com/YourUsername/SyMFood.git)
cd SyMFood

# Create the environment from the yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate symfood

```

## 🚀 Training

We provide two ways to train the model. Please ensure you have properly set up the environment and dataset paths in `zsd_detr/configs/config.py` before running.

### Option 1: Training with Shell Script (Recommended)

You can simply run the provided shell script:

```bash
sh train.sh

```

### Option 2: Training with Python Command

Alternatively, you can manually run the training script with specific arguments:

```bash
python train_net.py --config-file zsd_detr/configs/config.py --num-gpus 1

```

## ⚡ Inference / Evaluation

To evaluate the model (Zero-Shot Detection performance) using the trained weights, run the following command:

```bash
python train_net.py --eval_only True --config-file zsd_detr/configs/config.py --num-gpus 1

```

## ⚠️ Notes

* If you are using Docker, ensure **CUDA 11.3** (or compatible) is available in the container.
* Update dataset and pretrained backbone paths in the config files before running any scripts.

## 📝 Citation

If you find this code or paper useful for your research, please cite our paper:

```bibtex
@ARTICLE{xlwang2026symfood,
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

```

## 📧 Contact

If you have any questions regarding the code or the paper, please feel free to contact xinlongwang@m.ldu.edu.cn.
