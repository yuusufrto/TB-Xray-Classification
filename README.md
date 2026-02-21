# TB-Xray-Classification
# Tuberculosis Detection from Chest X-Rays

This project implements a deep learning pipeline for tuberculosis (TB) classification using chest radiography images.

## Dataset
TB Chest Radiography Database (Kaggle)

- 3,500 Normal images
- 700 Tuberculosis images
- Total: 4,200 images

## Methodology

- Stratified 70/15/15 train-validation-test split
- Class imbalance handled using weighted loss (5:1 ratio)
- Image resizing to 224×224
- Normalization to [0,1]
- Transfer learning using MobileNetV3-Small
- GPU: Tesla T4 (Google Colab)

## Experimental Conditions

- Baseline (clean images)
- Robustness evaluation using controlled image degradation (planned)
