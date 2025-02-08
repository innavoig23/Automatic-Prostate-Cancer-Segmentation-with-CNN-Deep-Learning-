# Automatic Prostate Cancer Segmentation in Multi-Modal MRI Images using CNN (Deep Learning)

This repository contains the code developed for the **Medical Image Processing** course project @polito, focused on segmenting prostate tumors in multi-modal MRI images.

## Dataset

The dataset used comes from a scientific challenge: 

```bibtex
@article{Saha2024,
  title = {Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international,  paired,  non-inferiority,  confirmatory study},
  volume = {25},
  ISSN = {1470-2045},
  url = {http://dx.doi.org/10.1016/S1470-2045(24)00220-1},
  DOI = {10.1016/s1470-2045(24)00220-1},
  number = {7},
  journal = {The Lancet Oncology},
  publisher = {Elsevier BV},
  author = {Saha,  Anindo and Bosma,  Joeran S and Twilt,  Jasper J and van Ginneken,  Bram and Bjartell,  Anders and Padhani,  Anwar R and Bonekamp,  David and Villeirs,  Geert and Salomon,  Georg and Giannarini,  Gianluca and Kalpathy-Cramer,  Jayashree and Barentsz,  Jelle and Maier-Hein,  Klaus H and Rusu,  Mirabela and Rouvière,  Olivier and van den Bergh,  Roderick and Panebianco,  Valeria and Kasivisvanathan,  Veeru and Obuchowski,  Nancy A and Yakar,  Derya and Elschot,  Mattijs and Veltman,  Jeroen and F\"{u}tterer,  Jurgen J and de Rooij,  Maarten and Huisman,  Henkjan and Saha,  Anindo and Bosma,  Joeran S. and Twilt,  Jasper J. and van Ginneken,  Bram and Noordman,  Constant R. and Slootweg,  Ivan and Roest,  Christian and Fransen,  Stefan J. and Sunoqrot,  Mohammed R.S. and Bathen,  Tone F. and Rouw,  Dennis and Immerzeel,  Jos and Geerdink,  Jeroen and van Run,  Chris and Groeneveld,  Miriam and Meakin,  James and Karag\"{o}z,  Ahmet and B\^one,  Alexandre and Routier,  Alexandre and Marcoux,  Arnaud and Abi-Nader,  Clément and Li,  Cynthia Xinran and Feng,  Dagan and Alis,  Deniz and Karaarslan,  Ercan and Ahn,  Euijoon and Nicolas,  Fran\c{c}ois and Sonn,  Geoffrey A. and Bhattacharya,  Indrani and Kim,  Jinman and Shi,  Jun and Jahanandish,  Hassan and An,  Hong and Kan,  Hongyu and Oksuz,  Ilkay and Qiao,  Liang and Rohé,  Marc-Michel and Yergin,  Mert and Khadra,  Mohamed and Şeker,  Mustafa E. and Kartal,  Mustafa S. and Debs,  Noëlie and Fan,  Richard E. and Saunders,  Sara and Soerensen,  Simon J.C. and Moroianu,  Stefania and Vesal,  Sulaiman and Yuan,  Yuan and Malakoti-Fard,  Afsoun and Mačiūnien,  Agnė and Kawashima,  Akira and de Sousa Machadov,  Ana M.M. de M.G. and Moreira,  Ana Sofia L. and Ponsiglione,  Andrea and Rappaport,  Annelies and Stanzione,  Arnaldo and Ciuvasovas,  Arturas and Turkbey,  Baris and de Keyzer,  Bart and Pedersen,  Bodil G. and Eijlers,  Bram and Chen,  Christine and Riccardo,  Ciabattoni and Alis,  Deniz and Courrech Staal,  Ewout F.W. and J\"{a}derling,  Fredrik and Langkilde,  Fredrik and Aringhieri,  Giacomo and Brembilla,  Giorgio and Son,  Hannah and Vanderlelij,  Hans and Raat,  Henricus P.J. and Pikūnienė,  Ingrida and Macova,  Iva and Schoots,  Ivo and Caglic,  Iztok and Zawaideh,  Jeries P. and Wallstr\"{o}m,  Jonas and Bittencourt,  Leonardo K. and Khurram,  Misbah and Choi,  Moon H. and Takahashi,  Naoki and Tan,  Nelly and Franco,  Paolo N. and Gutierrez,  Patricia A. and Thimansson,  Per Erik and Hanus,  Pieter and Puech,  Philippe and Rau,  Philipp R. and de Visschere,  Pieter and Guillaume,  Ramette and Cuocolo,  Renato and Falcão,  Ricardo O. and van Stiphout,  Rogier S.A. and Girometti,  Rossano and Briediene,  Ruta and Grigienė,  Rūta and Gitau,  Samuel and Withey,  Samuel and Ghai,  Sangeet and Penzkofer,  Tobias and Barrett,  Tristan and Tammisetti,  Varaha S. and Løgager,  Vibeke B. and Černý,  Vladimír and Venderink,  Wulphert and Law,  Yan M. and Lee,  Young J. and Bjartell,  Anders and Padhani,  Anwar R. and Bonekamp,  David and Villeirs,  Geert and Salomon,  Georg and Giannarini,  Gianluca and Kalpathy-Cramer,  Jayashree and Barentsz,  Jelle and Maier-Hein,  Klaus H. and Rusu,  Mirabela and Obuchowski,  Nancy A. and Rouvière,  Olivier and van den Bergh,  Roderick and Panebianco,  Valeria and Kasivisvanathan,  Veeru and Yakar,  Derya and Elschot,  Mattijs and Veltman,  Jeroen and F\"{u}tterer,  Jurgen J. and de Rooij,  Maarten and Huisman,  Henkjan},
  year = {2024},
  month = jul,
  pages = {879–887}
}
```

### Dataset Structure
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

Annotations are **8-bit PNG images**, where pixel values represent non-tumoral tissue when 0, while tumoral when 1.

The dataset includes three MRI modalities:

- **ADC** (Apparent Diffusion Coefficient)
- **HBV** (High b-value)
- **T2w** (T2-weighted)

The implemented pipeline used an **RGB composite image** created by stacking the three modalities into three channels: **ADC, HBV, and T2w**.

## Code

The segmentation task was implemented using **MMSegmentation**, an OpenMMLab toolbox for semantic segmentation:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

The model was trained and tested on Google Colab. They can be performed using `training.ipynb` and `test.ipynb`.
Additional scripts used for the final pipeline are also uploaded.


During the preliminary phase of the project, various approaches were tested, such as different network architecture, alternative pre-processing, or post-processing techniques. These additional scripts are not included in the repository. However, if you have to carry out similar tasks and need assistance, feel free to contact me.
