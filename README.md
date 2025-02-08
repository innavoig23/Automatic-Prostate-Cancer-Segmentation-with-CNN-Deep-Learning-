# Automatic Prostate Cancer Segmentation in Multi-Modal MRI Images using CNN (Deep Learning)

This repository contains the code developed for the [**Medical Image Processing**](https://didattica.polito.it/pls/portal30/gap.pkg_guide.viewGap?p_cod_ins=01VSOMV&p_a_acc=2026&p_header=S&p_lang=&multi=N) course project at Politecnico di Torino, focused on segmenting prostate cancer in multi-modal MRI images.

## Dataset

The dataset used in this project originates from a [scientific challenge](https://doi.org/10.1016/S1470-2045(24)00220-1). The directory structure is as follows:
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
Annotations are **8-bit PNG images**, where pixel values indicate:
- **0**: Non-tumoral tissue
- **1**: Tumoral tissue

The dataset includes three MRI modalities:

- **ADC** (Apparent Diffusion Coefficient)
- **HBV** (High b-value)
- **T2w** (T2-weighted)

These modalities were combined into an **RGB image** by stacking ADC, HBV, and T2w as separate channels.

## Codes, Pipelines, and Report

The segmentation task was implemented using [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation), an OpenMMLab toolbox for semantic segmentation.

The model was trained and tested on Google Colab using `training.ipynb` and `test.ipynb`.
Additional scripts used for the final pipeline are also uploaded.

During the preliminary phase of the project, various approaches were tested, such as different network architecture, alternative pre-processing, or post-processing techniques. These additional scripts are not included in the repository. However, if you have to carry out similar tasks and need assistance, feel free to contact me.

The pre-processing and post-processing pipelines are illustrated below:
### Pre-Processing Pipeline
![Pre-Processing Pipeline](Pipelines/pipelinePreprocessing.svg)

### Post-Processing Pipeline
![Post-Processing Pipeline](Pipelines/pipelinePostprocessing.svg)

The project report is available. However, since the course was taught in Italian, the report is also in Italian.
