# Automatic Prostate Cancer Segmentation in Multi-Modal MRI Images using CNN (Deep Learning)

This repository contains the code developed for the **Medical Image Processing** course project @polito, focused on segmenting prostate tumors in multi-modal MRI images.

## Dataset

The dataset used comes from a [scientific challenge](https://doi.org/10.1016/S1470-2045(24)00220-1).
The structure used in this project is as follows:
```bash
ProstateMRI/
│── ann_dir/        # Directory containing annotations
│   ├── train/      # Training set annotations
│   ├── val/        # Validation set annotations
│   ├── test/       # Test set annotations
│
│── img_dir/        # Directory containing images
│   ├── train/      # Training set images
│   ├── val/        # Validation set images
│   ├── test/       # Test set images
```
Annotations are **8-bit PNG images**, where pixel values represent non-tumoral tissue when 0, while tumoral when 1.

The dataset includes three MRI modalities:

- **ADC** (Apparent Diffusion Coefficient)
- **HBV** (High b-value)
- **T2w** (T2-weighted)

The implemented pipeline used an **RGB image** created by stacking the three modalities into three channels: **ADC, HBV, and T2w**.

## Code

The segmentation task was implemented using [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation), an OpenMMLab toolbox for semantic segmentation.

The model was trained and tested on Google Colab. They can be performed using `training.ipynb` and `test.ipynb`.
Additional scripts used for the final pipeline are also uploaded.

During the preliminary phase of the project, various approaches were tested, such as different network architecture, alternative pre-processing, or post-processing techniques. These additional scripts are not included in the repository. However, if you have to carry out similar tasks and need assistance, feel free to contact me.
