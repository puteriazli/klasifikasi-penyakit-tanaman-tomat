# Tomato Leaf Disease Classification  
End-to-End Computer Vision & Machine Learning Project

## Overview
This project is an end-to-end system for classifying tomato leaf diseases using image processing and machine learning.

The system covers the full pipeline, from image preprocessing and feature extraction to model training, evaluation, and deployment through a web application.  
The goal is to provide a simple and practical solution for identifying tomato plant diseases based on leaf images.

---

## Demo
A short video demonstration of the system can be found here:  
👉 **YouTube Demo:** https://youtu.be/QrW2xr9Nk54?si=qgHvAV1qWj_3svk7

The demo shows the full workflow, including image upload, disease classification, and result display through the web application.

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

  <img width="484" height="484" alt="image" src="https://github.com/user-attachments/assets/827373e0-a487-481f-aad4-9b7e6ecbb419" />
  
- Confusion matrix

  <img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/221cb743-7f2b-4466-8515-a082d8a31be1" />

- Classification report (precision, recall, F1-score)

  <img width="388" height="161" alt="{E0B3C7EA-F13D-4D33-AFF8-7DDCBD251E27}" src="https://github.com/user-attachments/assets/842e55ec-ee52-43a4-a958-264a044c5270" />
  
- Cross-validation scores
  
  <img width="585" height="457" alt="image" src="https://github.com/user-attachments/assets/017c7b10-3456-45a7-8310-553c419930f7" />

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
│   LICENSE
│
├───algoritma machine learning
│       model_svm.ipynb
│
└───website
    └───tomato-disease-classification
        │   app.py
        │   requirements.txt
        │
        ├───model
        │       scaler.pkl
        │       svm_model.pkl
        │
        ├───static
        │   ├───css
        │   │       details.css
        │   │       history.css
        │   │       index.css
        │   │       more.css
        │   │       results.css
        │   │
        │   ├───feature extraction
        │   │       dataset_tomat_features.npz
        │   │
        │   ├───images
        │   │       logo.png
        │   │
        │   └───uploads
        │           uploads.txt
        │
        └───templates
                details.html
                history.html
                index.html
                more.html
                results.html

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

## 🖥️ Web Application Screenshots

### Home Page
<img width="960" height="564" alt="Home Page" src="https://github.com/user-attachments/assets/00cd0ad0-4fe8-46b9-acf0-fcd903944fd5" />

---

### About Page
<img width="960" height="564" alt="About Page" src="https://github.com/user-attachments/assets/35bfc9c9-80ee-4fe0-ba71-184948a91b4d" />

<img width="960" height="564" alt="About Page - Additional Section" src="https://github.com/user-attachments/assets/7a15a599-5bb6-4838-bbfb-06c6f5159c3d" />

<img width="960" height="564" alt="About Page - Additional Section" src="https://github.com/user-attachments/assets/7e7d6521-5247-4d59-b7b0-09db497d11ac" />

---

### Prediction Results
<img width="1920" height="1128" alt="Prediction Results Page" src="https://github.com/user-attachments/assets/6176cf4d-d15a-485f-9e16-2b071da7d6dd" />

---

### Processing Results
<img width="1920" height="1128" alt="Processing Results Page" src="https://github.com/user-attachments/assets/a38af4b2-da57-4e12-a9f4-921caafa93e8" />

---

### Prediction History
<img width="1920" height="1128" alt="Prediction History Page" src="https://github.com/user-attachments/assets/36d77199-14c5-4391-8dc1-149b68ec7f5e" />

---

### Full History
<img width="1920" height="1128" alt="Full Prediction History Page" src="https://github.com/user-attachments/assets/0255f47e-f1cd-4ee3-bc84-56ee708f4b5a" />

---

### History by Category
<img width="1920" height="1128" alt="Category-Based History Page" src="https://github.com/user-attachments/assets/ad9fdc13-6116-43c5-8268-82bc78147de5" />

<img width="1920" height="1128" alt="Category-Based History Page - Additional View" src="https://github.com/user-attachments/assets/beec027e-8ecd-41de-a9f9-86609e63187a" />

---

### More Information
<img width="1920" height="1128" alt="More Information Page" src="https://github.com/user-attachments/assets/e9047fff-86fa-480d-adf2-3fa6f80cf1aa" />

---

### Image Detail View
<img width="1920" height="1128" alt="Image Detail Page" src="https://github.com/user-attachments/assets/269a227f-57b5-4f2f-929d-4292f2e70206" />

## Author
**Puteri Amelia Azli**  
Machine Learning | Computer Vision | Data Science
