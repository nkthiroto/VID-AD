# VID-AD: A Dataset for Image-Level Logical Anomaly Detection under Vision-Induced Distraction

<p align="left">
    <a href='https://arxiv.org/abs/xxxx.xxxxx'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://drive.google.com/file/d/1_UaWAuylvaErnvOq0uxq4gIg_NeSUNdz/view?usp=sharing'>
      <img src='https://img.shields.io/badge/Data-Dataset-green?style=plastic&logo=Google%20Drive&logoColor=green' alt='Dataset'>
    </a>
</p>

## Abstract

<!-- TODO: Replace with actual abstract -->
Logical anomaly detection in industrial inspection remains challenging due to variations in visual appearance (e.g., background clutter, illumination shift, and blur), which often distract vision-centric detectors from identifying rule-level violations. 
However, existing benchmarks rarely provide controlled settings where logical states are fixed while such nuisance factors vary.
To address this gap, we introduce VID-AD, a dataset for logical anomaly detection under vision-induced distraction, containing 10 manufacturing scenarios and five capture conditions, which yields 50 one-class tasks and 10,395 images.
Each scenario is defined by two logical constraints (from quantity, length, type, placement, and relation) and includes both single-constraint and combined violations.
We further propose a language-based anomaly detection framework that relies solely on text descriptions generated from normal images.
Using contrastive learning with positive texts and contradiction-based negative texts synthesized from these descriptions, our method learns embeddings that emphasize logical content rather than low-level appearance.
Extensive experiments demonstrate consistent improvements over baselines across the evaluated settings.

## Overview

<p align="center">
  <img src="figures/Overview.png" width="100%">
</p>

## Dataset

### Download

<!-- TODO: Replace with actual dataset URL -->
The dataset can be downloaded from [here](https://drive.google.com/file/d/1_UaWAuylvaErnvOq0uxq4gIg_NeSUNdz/view?usp=sharing).

### Dataset Structure

```
VID-AD_dataset/
├── {Category}/                    # e.g., Balls, Blocks, Cookies, ...
│   ├── train/
│   │   └── good/                  # Normal training video frames
│   └── test/
│       ├── good/                  # Normal test video frames
│       └── logical_anomalies/     # Anomalous test video frames
│           ├── Single-Aspect-A/
│           ├── Single-Aspect-B/         # (varies by category)
│           └── Dual-Aspects/
├── {Category}_Cable_BG/           # Cable Background
├── {Category}_Mesh_BG/            # Mesh Background
├── {Category}_Blurry_CD/          # Blurry Condition
└── {Category}_Low-light_CD/       # Low-light Condition
```

**10 Scenarios:** Balls, Blocks, Cookies, Dishes, Fruits, Ropes, Stationery, Sticks, Tapes, Tools

**Five Capture Conditions:** Original, Cable_BG, Mesh_BG, Low-light_CD, Blurry_CD

### Dataset Statistics

<p align="center">
  <img src="figures/Dataset_Statistics.png" width="100%">
</p>

<!-- ## Citation

If you find this dataset useful, please cite our paper:

```bibtex
@article{vid_ad2026,
  title={Paper Title (TBD)},
  author={Author1 and Author2 and Author3},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
``` -->

## License

This dataset is released under the [MIT License](https://opensource.org/licenses/MIT).
