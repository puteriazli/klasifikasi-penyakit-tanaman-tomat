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
- Bahasa Pemrograman: Python, HTML, CSS, JavaScript  
- Computer Vision: OpenCV, 
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
<img width="960" height="564" alt="{5B2A8E04-8E78-4FCA-B11B-41D9FA2A2D6F}" src="https://github.com/user-attachments/assets/00cd0ad0-4fe8-46b9-acf0-fcd903944fd5" />

### Menu Tentang Website
<img width="960" height="564" alt="{B85BCBDC-8013-42FB-A998-414DC04BB6ED}" src="https://github.com/user-attachments/assets/35bfc9c9-80ee-4fe0-ba71-184948a91b4d" />
<img width="960" height="564" alt="{E026627A-6869-475E-A5B6-6F38FD6F9385}" src="https://github.com/user-attachments/assets/7a15a599-5bb6-4838-bbfb-06c6f5159c3d" />
<img width="960" height="564" alt="{9465AE71-7217-4227-B323-9325726B855A}" src="https://github.com/user-attachments/assets/7e7d6521-5247-4d59-b7b0-09db497d11ac" />

### Menu Hasil Prediksi

### Menu Hasil Pengolahan

### Menu Riwayat

### Menu Semua Riwayat

### Menu Riwayat Perkategori

### Menu Selengkapnya

### Menu Detail Gambar


## 📊 Evaluasi Model
### Confusion Matrix
<img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/221cb743-7f2b-4466-8515-a082d8a31be1" />

### Classification Report
<img width="388" height="161" alt="{E0B3C7EA-F13D-4D33-AFF8-7DDCBD251E27}" src="https://github.com/user-attachments/assets/842e55ec-ee52-43a4-a958-264a044c5270" />

### Accuracy Score
<img width="484" height="484" alt="image" src="https://github.com/user-attachments/assets/827373e0-a487-481f-aad4-9b7e6ecbb419" />

## 📚 Sumber Dataset
Dataset citra daun tomat diperoleh dari:
[Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/ahmadzargar/tomato-leaf-disease-dataset-segmented)
Dataset citra daun tanaman tomat yang berisi berbagai jenis penyakit dan kondisi sehat. Namun, data yang digunakan dalam projek ini ada 4, yaitu sehat, busuk daun, jamu daun, dan septoria
