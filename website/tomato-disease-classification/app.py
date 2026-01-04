from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import joblib
import json
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from math import radians
import uuid

app = Flask(__name__)

# Load model dan scaler
svm_model = joblib.load("model/svm_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Folder untuk upload gambar
UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE = "history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ File history.json rusak. Akan dibuat ulang.")
    return []

def save_history(history_data):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_data, f, indent=4)

history_data = load_history()

def preprocess_segment(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (512, 512))

    # Ukuran citra
    height, width, _ = resized_image.shape
    
    #avg_brightness = np.mean(resized_image)
    #target_brightness = 130
    #contrast = 1.5
    #brightness_diff = target_brightness - avg_brightness
    #adjusted_image = cv2.convertScaleAbs(resized_image, alpha=contrast, beta=brightness_diff)
    #adjusted_image = np.clip(adjusted_image, 0, 255)
    
    brightness = 5
    adjusted_image = cv2.convertScaleAbs(resized_image, alpha=1, beta=brightness)
    
    # Brightness rata-rata
    brightness_avg = np.mean(cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2HSV)[:, :, 2])
    
    blurred_image = cv2.GaussianBlur(adjusted_image, (7, 7), 0)
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2HSV)

    lower_green = np.array([35, 30, 30])
    upper_green = np.array([90, 255, 255])
    lower_brown = np.array([10, 30, 10])
    upper_brown = np.array([80, 255, 255])
    lower_yellow = np.array([10, 50, 40])
    upper_yellow = np.array([40, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_combined = cv2.bitwise_or(mask_green, cv2.bitwise_or(mask_brown, mask_yellow))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(mask_cleaned)
    min_contour_area = 2500
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if area > min_contour_area and 0.1 < aspect_ratio < 3.0:
            cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)

    segmented = cv2.bitwise_and(resized_image, resized_image, mask=mask_cleaned)

    # Jumlah pixel daun dan latar
    leaf_pixel_count = int(np.sum(mask_cleaned == 255))
    background_pixel_count = int(np.sum(mask_cleaned == 0))

    preprocessing_info = {
        "image_size": f"{width} x {height}",
        "brightness_avg": round(float(brightness_avg), 2),
        "leaf_pixel_count": leaf_pixel_count,
        "background_pixel_count": background_pixel_count
    }

    return segmented, preprocessing_info

bins = 8
def extract_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    hist_features /= np.sum(hist_features)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    angles = [radians(0), radians(45), radians(90), radians(135)]
    glcm = graycomatrix(gray, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
    glcm_features = np.array([
        np.mean(graycoprops(glcm, prop)) 
        for prop in ["contrast", "correlation", "energy", "homogeneity"]
    ])
    
    return np.concatenate((hist_features, glcm_features))

@app.route("/")
def index():
    return render_template("index.html")

def save_image(image, filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return url_for("static", filename="uploads/" + filename)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return redirect(request.url)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    if image is None:
        return render_template("results.html", description="Error", prediction="Gambar tidak valid atau gagal diproses.", filename=file.filename)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (512, 512))
    # Simpan resized
    resized_url = save_image(resized_image, "resized_" + file.filename)
    brightness = 5
    adjusted_image = cv2.convertScaleAbs(resized_image, alpha=1, beta=brightness)
    blurred_image = cv2.GaussianBlur(adjusted_image, (7, 7), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2HSV)
    #hsv_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # Supaya bisa disimpan dengan benar
    hsv_url = save_image(hsv_image, "hsv_" + file.filename)
    # Buat mask
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    lower_brown = np.array([10, 30, 10])
    upper_brown = np.array([80, 255, 255])
    mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
    lower_yellow = np.array([10, 50, 40])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_combined = cv2.bitwise_or(mask_green, cv2.bitwise_or(mask_brown, mask_yellow))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(mask_cleaned)
    min_contour_area = 2500
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if area > min_contour_area and 0.1 < aspect_ratio < 3.0:
            cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Segmentasi (optional, kalau perlu)
    segmented = cv2.bitwise_and(resized_image, resized_image, mask=mask_combined)
    mask_url = save_image(segmented, "mask_" + file.filename)

    # Ekstrak fitur dan prediksi
    features = extract_features(segmented)
    scaled_features = scaler.transform(features.reshape(1, -1))
    prediction = svm_model.predict(scaled_features)[0]

    preprocessing_images = {
    "resized": "resized_" + file.filename,
    "hsv": "hsv_" + file.filename,
    "mask": "mask_" + file.filename
    }

    preprocessing_info = {
        "image_size": f"{resized_image.shape[1]} x {resized_image.shape[0]}",
        "brightness_avg": float(np.mean(hsv_image[:, :, 2])),
        "leaf_pixel_count": int(np.sum(mask_combined == 255)),
        "background_pixel_count": int(np.sum(mask_combined == 0))
    }

    # Simpan ke history
    history_data.append({
        "id": str(uuid.uuid4()),
        "filename": file.filename,
        "category": prediction,
        "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "file_path": url_for("static", filename="uploads/" + file.filename),
        "description": request.form.get("description", "Tidak ada deskripsi"),
        "features": features.tolist(),
        "preprocessing_info": preprocessing_info,
        "preprocessing_images": preprocessing_images  # Pastikan preprocessing_images ada di sini
        })
    save_history(history_data)

    return render_template(
        "results.html",
        filename=file.filename,
        description=request.form.get("description", ""),
        prediction=prediction,
        feature_names=[
            "H_bin_0", "H_bin_1", "H_bin_2", "H_bin_3", "H_bin_4", "H_bin_5", "H_bin_6", "H_bin_7",
            "S_bin_0", "S_bin_1", "S_bin_2", "S_bin_3", "S_bin_4", "S_bin_5", "S_bin_6", "S_bin_7",
            "V_bin_0", "V_bin_1", "V_bin_2", "V_bin_3", "V_bin_4", "V_bin_5", "V_bin_6", "V_bin_7",
            "contrast", "correlation", "energy", "homogeneity"
        ],
        features=features.tolist(),
        preprocessing_info=preprocessing_info,
        preprocessing_images=preprocessing_images
    )

@app.route("/history")
def history():
    history_data = load_history()

    for item in history_data:
        try:
            date_obj = datetime.strptime(item["date"], "%d/%m/%Y %H:%M:%S")
        except ValueError:
            try:
                date_obj = datetime.strptime(item["date"], "%d-%m-%Y %H:%M:%S")
            except ValueError:
                continue

        item["display_date"] = date_obj.strftime("%d/%m/%Y %H:%M:%S")
        item["filter_date"] = date_obj.strftime("%Y-%m-%d")  # untuk filtering

    latest_three = sorted(history_data, key=lambda x: x["display_date"], reverse=True)[:5]
    all_history = sorted(history_data, key=lambda x: x["display_date"], reverse=True)

    categorized_history = {}
    for category in ["sehat", "busuk daun", "jamur daun", "septoria"]:
        categorized_history[category] = [item for item in history_data if item["category"] == category]

    for category in categorized_history:
        categorized_history[category].sort(key=lambda x: x["display_date"], reverse=True)
        categorized_history[category] = {
            "latest": categorized_history[category][:2],
            "more": categorized_history[category][2:]
        }

    return render_template(
        "history.html",
        latest_three=latest_three,
        categorized_history=categorized_history,
        all_history=all_history
    )

@app.route("/delete_history/<item_id>", methods=["DELETE"])
def delete_history(item_id):
    global history_data
    # Cari data berdasarkan id
    target_item = next((item for item in history_data if item["id"] == item_id), None)

    if not target_item:
        return jsonify({"success": False, "message": "Data tidak ditemukan"}), 404

    # Hapus dari list
    history_data = [item for item in history_data if item["id"] != item_id]
    save_history(history_data)

    # Hapus file fisik jika ada
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], target_item["filename"])
    if os.path.exists(file_path):
        os.remove(file_path)

    return jsonify({"success": True})

from datetime import datetime

@app.route("/more/<category>")
def more(category):
    history_data = load_history()

    filtered_history = []
    for item in history_data:
        if item["category"] == category:
            try:
                date_obj = datetime.strptime(item["date"], "%d/%m/%Y %H:%M:%S")
            except ValueError:
                try:
                    date_obj = datetime.strptime(item["date"], "%d-%m-%Y %H:%M:%S")
                except ValueError:
                    continue

            # ✅ Dua format tanggal: satu untuk display, satu untuk filter
            item["display_date"] = date_obj.strftime("%d/%m/%Y %H:%M:%S")
            item["filter_date"] = date_obj.strftime("%Y-%m-%d")

            filtered_history.append(item)

    # Urutkan terbaru dulu
    filtered_history.sort(
        key=lambda x: datetime.strptime(x["display_date"], "%d/%m/%Y %H:%M:%S"),
        reverse=True
    )

    return render_template("more.html", category=category, history=filtered_history)

@app.route("/details/<filename>")
def details(filename):
    history = load_history()
    image_data = next((item for item in history if item.get("filename") == filename), None)

    if not image_data:
        return "Gambar tidak ditemukan", 404

    image_data["feature_names"] = [
        "H_bin_0", "H_bin_1", "H_bin_2", "H_bin_3", "H_bin_4", "H_bin_5", "H_bin_6", "H_bin_7",
        "S_bin_0", "S_bin_1", "S_bin_2", "S_bin_3", "S_bin_4", "S_bin_5", "S_bin_6", "S_bin_7",
        "V_bin_0", "V_bin_1", "V_bin_2", "V_bin_3", "V_bin_4", "V_bin_5", "V_bin_6", "V_bin_7",
        "contrast", "correlation", "energy", "homogeneity"
    ]

    return render_template("details.html", **image_data)

if __name__ == "__main__":
    app.run(debug=True)
