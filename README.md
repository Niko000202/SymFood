# Code for SyMFood: Synergistic Multi-Modal Prompting for Fine-Grained Zero-Shot Food Detection
## Overview
This repository contains the code and materials for our TCSVT paper on fine-grained zero-shot food detection. We propose SyMFood (Synergistic Multi-Modal Prompting) to improve detection of unseen food categories by leveraging multi-modal prompts.

## Environment
We provide a reproducible environment file based on our training setup.

- Python 3.8
- PyTorch 1.12.1 + CUDA 11.3
- MMDetection (2.x) + MMCV (2.x)
- Detectron2 / Detrex (for feature extraction components)

### Create the environment
```
conda env create -f environment.yml
conda activate symfood
```
## Notes
- If you are using Docker, ensure CUDA 11.3 (or compatible) is available in the container.
- Update dataset and pretrained backbone paths in the config files before running any scripts.
- Full code and instructions will be released once the paper is accepted.
