# Klasifikasi Penyakit Tanaman Tomat Menggunakan SVM Berbasis Citra Daun

## 📌 Deskripsi Proyek
Proyek ini merupakan sistem klasifikasi penyakit tanaman tomat berdasarkan **citra daun** menggunakan pendekatan **Machine Learning dan Computer Vision**. Metode utama yang digunakan adalah **Support Vector Machine (SVM)** untuk mengklasifikasikan kondisi daun tomat ke dalam 4 kelas penyakit maupun kondisi sehat. 4 kelas tersebut yakni sehat, busuk daun, jamur daun, dan septoria.

Model machine learning diintegrasikan ke dalam **aplikasi web**, sehingga pengguna dapat melakukan prediksi penyakit hanya dengan mengunggah gambar daun tomat melalui antarmuka web.

Proyek ini dirancang sebagai implementasi end-to-end mulai dari preprocessing citra, pelatihan model, evaluasi, hingga deployment sederhana berbasis web.

---

## 🎯 Tujuan
- Mengklasifikasikan penyakit daun tanaman tomat berdasarkan citra
- Menerapkan algoritma SVM pada permasalahan klasifikasi citra
- Mengintegrasikan model machine learning ke dalam aplikasi web
- Menyediakan sistem prediksi yang mudah digunakan oleh pengguna

---

## 🧠 Metodologi

### 1. Preprocessing Citra
- Resize citra ke ukuran 512x512 piksel agar seragam 
- Tingkatkan brightness citra sebesar 5 unit
- Aplikasikan Gaussian Blur untuk mengurangi noise citra menggunakan kernel berukuran 7x7
- Konversi warna RGB ke HSV

### 2. Segmentasi Citra
- Tentukan rentang warna citra (hijau, kuning, dan coklat)
- Aplikasikan Bitwise Mask
- Aplikasikan teknik morfologi ellips menggunakan kernel berukuran 5x5
- Aplikasikan teknik kontur dengan ukuran area 0.1 - 3.0

### 3. Ekstraksi Fitur
- Ekstraksi fitur: warna (HSV) dan tekstur (GLCM)
- Hasil ektraksi fitur: 28 fitur berupa 24 fitur warna dan 4 fitur tekstur
- Representasi 28 fitur numerik sebagai input model

### 4. Klasifikasi
- Algoritma: Support Vector Machine (SVM)
- Pelatihan model: 80% data latih dan 20% evaluasi pada data uji
- Optimalisasi model menggunakan GridSearch dengan 18 parameter
- Parameter regulasi:0.1, 1, 10
- Parameter gamma: auto dan scale
- Parameter Kernel: Linear, RBF, Polynomial

### 5. Integrasi Web
- Model disimpan dalam bentuk file (`.pkl`)
- Model diintegrasikan menggunakan framework Flask
- Aplikasi web memuat model dan melakukan prediksi saat pengguna mengunggah citra

---

## 🛠️ Teknologi yang Digunakan
- Bahasa Pemrograman: Python, HTML, CSS, JavaScript :contentReference[oaicite:0]{index=0}  
- Computer Vision 
- Machine Learning: SVM 
- Web Framework: Flask 
- Visualisasi: Matplotlib, Seaborn

---

## 📂 Struktur Folder
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

## ▶️ Cara Menjalankan Proyek

### 1. Buka terminal komputer (CMD), arahkan ke folder yang diinginkan, ketik:
1. git clone https://github.com/puteriazli/klasifikasi-penyakit-tanaman-tomat
2. cd klasifikasi-penyakit-tanaman-tomat/website/tomato-disease-classification
3. pip install -r requirements.txt
4. python app.py

### 2. Jalankan projek di browser
1. Buka browser dan akses: http://localhost:5000
2. Buka menu Tentang Saya untuk melihat petunjuk penggunaan website

## 🖥️ Screenshot Web App
### Menu Beranda

### Menu Tentang Website

### Menu Hasil Prediksi

### Menu Hasil Pengolahan

### Menu Riwayat

### Menu Semua Riwayat

### Menu Riwayat Perkategori

### Menu Selengkapnya

### Menu Detail Gambar


## 📊 Evaluasi Model
### Confusion Matrix

### Classification Report

### Accuracy Score


## 📚 Sumber Dataset
Dataset citra daun tomat diperoleh dari:
[Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/ahmadzargar/tomato-leaf-disease-dataset-segmented)
Dataset citra daun tanaman tomat yang berisi berbagai jenis penyakit dan kondisi sehat. Namun, data yang digunakan dalam projek ini ada 4, yaitu sehat, busuk daun, jamu daun, dan septoria
