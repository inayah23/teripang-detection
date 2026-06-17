from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================

model = YOLO("model/best.pt")

# ==========================
# FOLDER
# ==========================

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/result"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==========================
# HALAMAN UTAMA
# ==========================

@app.route("/")
def index():

    return render_template("index.html")

# ==========================
# IDENTIFIKASI
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return render_template("index.html")

    file = request.files["image"]

    if file.filename == "":

        return render_template("index.html")

    # simpan gambar input

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        "input.jpg"
    )

    file.save(upload_path)

    # lokasi hasil

    result_path = os.path.join(
        RESULT_FOLDER,
        "hasil.jpg"
    )

    # deteksi YOLO

    results = model.predict(
        source=upload_path,
        save=False,
        verbose=False
    )

    # gambar hasil

    img = results[0].plot()

    cv2.imwrite(
        result_path,
        img
    )

    # tampilkan

    return render_template(

        "hasil.html",

        input_image="uploads/input.jpg",

        image="result/hasil.jpg"

    )

# ==========================
# RUN
# ==========================

if __name__ == "__main__":

    app.run(debug=True)