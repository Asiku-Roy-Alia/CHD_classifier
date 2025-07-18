# Fetal Congenital Heart Disease (CHD) Diagnosis using Deep Learning

This repository contains the complete end‑to‑end pipeline for prenatal diagnosis of Congenital Heart Diseases (CHD) using Convolutional Neural Networks (CNNs). It uses 2‑D  sonography for _segmentation_ and _classification_ tasks, covering everything from data exploration to clinical deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
    - [Data Sources](#data-sources)
    - [Expected Data Structure](#expected-data-structure)
- [Usage](#usage)
    1. [Data Exploration](#1-data-exploration)
    2. [Data Preprocessing](#2-data-preprocessing)
    3. [Model Training](#3-model-training)
        - [Segmentation Model (Trimester 2)](#segmentation-model-trimester-2)
        - [Classification Model (Trimester 1)](#classification-model-trimester-1)
    4. [Deployment and Inference](#4-deployment-and-inference)
    5. [Performance Analysis & Visualization](#5-performance-analysis-and-visualization)
- [Models and Architectures](#models-and-architectures)
- [License](#license)
- [Contributing](#contributing)

---

## Project Overview

This project implements an **end‑to‑end deep‑learning pipeline** for prenatal CHD diagnosis:

- **Segmentation (Trimester 2)** – Efficiently delineates key fetal cardiac structures from 2‑D ultrasound images.
- **Classification (Trimester 1)** – Predicts CHD classes _(Normal, ASD, VSD)_ from 2‑D/3‑D sonography, optionally using segmentation features.
- **Deployment** – A clinical inference system outputs diagnoses, probability scores, and PDF reports.

Key features include robust data handling, modern model architectures, ensemble & TTA strategies, and detailed performance analytics.

## Repository Structure

```
├── LICENSE
├── README.md
├── deployment/                 # Inference & clinical deployment
│   └── deploy.py
├── documentation/              # Scripts for analysis & plotting
│   └── graphs.py
├── exploration/                # Dataset audits & metadata extraction
│   ├── explore_trimester_1.py
│   ├── explore_trimester_2.py
│   ├── extract_metadata_trim_1.py
│   └── extract_metadata_trim_2.py
├── models/                     # Model definitions & training pipelines
│   ├── classification_model.py
│   └── segmentation_model.py
└── preprocessing/              # Cleaning, augmentation, auto‑labeling
    ├── auto_labeler.py
    ├── harvest_annotations.py
    ├── preprocess_trimester_1.py
    └── preprocess_trimester_2.py
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Asiku-Roy-Alia/CHD_classifier.git
cd fetal-chd-diagnosis
```

### 2. Create a virtual environment _(recommended)_

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Pick the right CUDA / CPU wheel for your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy pandas matplotlib seaborn scikit-learn tqdm timm opencv-python-headless \
            albumentations pydicom reportlab
```

> **Heads‑up:**
> 
> - Use `opencv-python-headless` on servers without GUI support.
>     
> - `pydicom` is required for DICOM images.
>     
> - `reportlab` enables PDF report generation during deployment.
>     

---

## Data Preparation

### Data Sources

You will need two fetal ultrasound datasets:

1. **Trimester 1** – Classification images _(Normal / ASD / VSD)_ from [Figshare](https://figshare.com/articles/figure/First_Trimester_Fetal_Echocardiography_Data_Set_for_Classification/21215492?file=37624184)
    
2. **Trimester 2** – Segmentation images & masks of cardiac structures from [here](https://figshare.com/articles/figure/Second_Trimester_Fetal_Echocardiography_Data_Set_for_Image_Segmentation/21215597)
    

Place them under `data/` as follows.

### Expected Data Structure

```text
data/
├── first_trimester/
│   ├── images/                       # .png /.jpg images
│   └── predicted_masks/              # auto‑labeled masks
└── second_trimester/
    ├── images/                       # Ultrasound frames
    ├── masks/                        # Ground‑truth segmentation masks

```

---

## Usage

### 1. Data Exploration

Audit dataset structure & extract metadata:

```bash
# Trimester 1
python exploration/explore_trimester_1.py \
  --data_root data/first_trimester/images \
  --generate_tree_file --audit_counts

python exploration/extract_metadata_trim_1.py \
  --data_root data/first_trimester/images \
  --dataset_name "Trimester_1_Classification_Data" \
  --output_dir documentation/from_code

# Trimester 2
python exploration/explore_trimester_2.py \
  --data_root data/second_trimester/images \
  --generate_tree_file

python exploration/extract_metadata_trim_2.py \
  --data_root data/second_trimester/images \
  --dataset_name "Trimester_2_Segmentation_Data" \
  --output_dir documentation/from_code
```

### 2. Data Preprocessing

```bash
# Harvest ground‑truth masks (Trimester 2)
python preprocessing/harvest_annotations.py \
  --annotations_root data/second_trimester/masks \
  --output_root processed_annotations/second_trimester_masks

# Auto‑label Trimester 1 images with a pre‑trained segmentation model
python preprocessing/auto_labeler.py \
  --unlabeled_data_root data/first_trimester/images \
  --model_path segmentation_checkpoints/best_model.pth \
  --output_root data/first_trimester/predicted_masks

# Preprocess datasets
python preprocessing/preprocess_trimester_1.py \
  --data_root data/first_trimester/raw_images \
  --output_root data/first_trimester/processed_images

python preprocessing/preprocess_trimester_2.py \
  --data_root data/second_trimester/raw_images \
  --output_root data/second_trimester/processed_images
```

### 3. Model Training

#### Segmentation Model (Trimester 2)

```bash
python models/segmentation_model.py --mode train \
  --images_dir data/second_trimester/images \
  --masks_dir  data/second_trimester/masks \
  --num_epochs 100 --batch_size 8 --learning_rate 1e-4 \
  --num_workers 4 --seg_target_size 256 256 \
  --num_classes 12 --checkpoint_save_dir segmentation_checkpoints
```

#### Classification Model (Trimester 1)

```bash
python models/classification_model.py \
  --annotations_file data/first_trimester/fetal_chd_annotations.json \
  --images_dir data/first_trimester/images \
  --use_segmentation \
  --segmentation_dir data/first_trimester/predicted_masks \
  --model_name efficientnet_b4 \
  --batch_size 16 --learning_rate 1e-4 --num_epochs 100 \
  --num_folds 5 --dropout_rate 0.5 \
  --checkpoint_save_base_dir classification_checkpoints \
  --evaluation_results_base_dir classification_evaluation_results \
  --pretrained_backbone --use_class_weights
```

### 4. Deployment and Inference

```bash
# Single image diagnosis
python deployment/deploy.py \
  --segmentation_model segmentation_checkpoints/best_model.pth \
  --classification_models classification_checkpoints/fold_1/best_model.pth \
                         classification_checkpoints/fold_2/best_model.pth \
  --input  data/second_trimester/images/Video1/00000029.png \
  --output inference_results/single_image_example \
  --device cuda --confidence_threshold 0.7 \
  --ensemble_method soft_voting --tta --clinical_mode --save_visualizations

# Batch inference
python deployment/deploy.py \
  --segmentation_model segmentation_checkpoints/best_model.pth \
  --classification_models classification_checkpoints/fold_1/best_model.pth \
                         classification_checkpoints/fold_2/best_model.pth \
  --input  data/first_trimester/images \
  --output inference_results/batch_example \
  --device cuda --confidence_threshold 0.7 \
  --ensemble_method soft_voting --tta --clinical_mode --save_visualizations
```

### 5. Performance Analysis and Visualization

```bash
python documentation/graphs.py \
  --data_base_dir classification_checkpoints \
  --num_folds 5 \
  --output_dir analysis_results/classification_performance_graphs
```

_(Swap `--data_base_dir` to `segmentation_checkpoints` for segmentation metrics.)_

---

## Models and Architectures

|Task|Architecture|Highlights|
|---|---|---|
|**Segmentation**|**EfficientAttentionUNet**|U‑Net + SE blocks, Attention Gates, ASPP for multiscale context|
|**Classification**|**MultiModalCHDClassifier**|EfficientNet‑B4 backbone + optional ResNet‑34 on masks, attention pooling, auxiliary confidence head|

## License

Distributed under the terms of the **LICENSE** file in this repo.

