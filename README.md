# Tomato Leaf Disease Classification  
End-to-End Computer Vision & Machine Learning Project

## Overview
This project is an end-to-end system for classifying tomato leaf diseases using image processing and machine learning.

The system covers the full pipeline, from image preprocessing and feature extraction to model training, evaluation, and deployment through a web application.  
The goal is to provide a simple and practical solution for identifying tomato plant diseases based on leaf images.

---

## Problem Statement
Tomato plants are vulnerable to various leaf diseases that often show similar visual symptoms. Manual identification is time-consuming and depends heavily on experience.

This project aims to:
- Automatically classify tomato leaf diseases from images
- Reduce human error in visual inspection
- Provide a lightweight and interpretable machine learning solution

---

## Project Scope
This project includes:
- Image preprocessing and segmentation
- Feature extraction from leaf images
- Machine learning model training and evaluation
- Deployment as a web-based classification system

The entire system was developed end-to-end by the author.

---

## Methodology

### 1. Image Preprocessing and Segmentation
- Image resizing and brightness adjustment
- Noise reduction using Gaussian blur
- Color-based segmentation in HSV color space
- Morphological operations to clean segmentation results
- Extraction of leaf regions from the background

This step ensures that the model focuses only on relevant leaf areas.

---

### 2. Feature Extraction
Two types of features are used:

**Color Features (HSV Histogram)**
- Captures color distribution of leaf surfaces
- Useful for detecting discoloration caused by diseases

**Texture Features (GLCM)**
- Contrast
- Correlation
- Energy
- Homogeneity

These features help distinguish surface patterns between healthy and diseased leaves.

---

### 3. Model Training
- Algorithm: Support Vector Machine (SVM)
- Feature scaling using StandardScaler
- Hyperparameter tuning with GridSearchCV
- 80:20 train-test split
- 5-fold cross validation

This approach was chosen to balance performance, interpretability, and computational efficiency.

---

## Model Evaluation
Model performance is evaluated using:
- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Cross-validation scores

Both training and testing accuracy are compared to check model stability and overfitting.

---

## Web Application
The trained model is deployed as a web application that allows users to:
- Upload tomato leaf images
- Receive disease classification results
- View prediction history and details

The web app integrates preprocessing, feature extraction, and prediction into a single workflow.

---

## Project Structure

---

## Technologies Used
- Python
- OpenCV
- NumPy, Pandas
- scikit-learn
- scikit-image
- Mahotas
- Flask
- HTML & CSS
- Matplotlib & Seaborn

---

## Key Takeaways
This project demonstrates:
- End-to-end machine learning system development
- Practical computer vision pipeline design
- Feature engineering for image-based classification
- Model deployment into a real web application
- Clean separation between model logic and application layer

---

## Future Improvements
- Comparison with deep learning models (CNN)
- Data augmentation
- Mobile or API-based deployment
- Improved model explainability

---

## Author
**[Puteri Amelia Azli]**  
Machine Learning | Computer Vision | Data Science
