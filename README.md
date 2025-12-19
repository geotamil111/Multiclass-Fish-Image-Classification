# 🐟 Multiclass Fish Image Classification

## Project Overview
This project focuses on classifying fish images into multiple categories using Deep Learning techniques. A Convolutional Neural Network (CNN) and Transfer Learning models are used to train and evaluate performance. The best-performing model is deployed using a Streamlit web application for real-time fish image prediction.

---

## Problem Statement
To build a deep learning model that can accurately identify the type of fish from an input image and deploy it as a user-friendly web application.

---

## Skills Gained
- Python  
- Deep Learning  
- TensorFlow / Keras  
- Image Preprocessing & Augmentation  
- Transfer Learning  
- Model Evaluation  
- Streamlit Deployment  
- GitHub Project Management  

---

## Domain
Image Classification

---

## Dataset
The dataset consists of fish images organized into folders by species. The data is provided as a ZIP file and extracted locally.

### Dataset Structure
data/
├── train/
│ ├── fish_class_1/
│ ├── fish_class_2/
│ └── ...
├── val/
└── test/

---

## Project Workflow

### 1. Data Preprocessing
- Rescale images to [0, 1]
- Resize images to 224 × 224
- Apply data augmentation (rotation, zoom, horizontal flip)

### 2. Model Training
- Custom CNN model built from scratch
- Transfer Learning using:
  - VGG16
  - ResNet50
  - MobileNet
  - InceptionV3
  - EfficientNetB0
- Fine-tuning and saving the best model in `.h5` format

### 3. Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training and validation accuracy/loss visualization

### 4. Model Deployment
- Streamlit web application for:
  - Uploading fish images
  - Predicting fish category
  - Displaying confidence score

---

## Streamlit Application
Run the Streamlit app using:
```bash
streamlit run app.py
