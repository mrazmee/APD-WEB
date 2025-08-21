import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==============================
# KONFIGURASI MODEL
# ==============================
MODEL_UTAMA_PATH = "Model/Utama/best (1).pt"
MODEL_SEPATU_PATH = "Model/Sepatu/best.pt"
NAMA_KELAS_ORANG = "Person"
APD_WAJIB = ["Hardhat", "Mask", "Safety Vest", "boot"]

# Load model
model_utama = YOLO(MODEL_UTAMA_PATH)
model_sepatu = YOLO(MODEL_SEPATU_PATH)

# ==============================
# Fungsi bantu
# ==============================
def is_inside(box_inner, box_outer):
    cx = (box_inner[0] + box_inner[2]) / 2
    cy = (box_inner[1] + box_inner[3]) / 2
    return box_outer[0] < cx < box_outer[2] and box_outer[1] < cy < box_outer[3]

def analisis_gambar(img):
    hasil_utama = model_utama(img, verbose=False)[0]
    hasil_sepatu = model_sepatu(img, verbose=False)[0]

    deteksi_orang, semua_deteksi_non_orang = [], []
    for box in hasil_utama.boxes:
        nama_kelas = model_utama.names[int(box.cls)]
        koordinat = box.xyxy[0].cpu().numpy()
        if nama_kelas == NAMA_KELAS_ORANG:
            deteksi_orang.append(koordinat)
        else:
            semua_deteksi_non_orang.append({"box": koordinat, "class": nama_kelas})

    for box in hasil_sepatu.boxes:
        nama_kelas = model_sepatu.names[int(box.cls)]
        koordinat = box.xyxy[0].cpu().numpy()
        semua_deteksi_non_orang.append({"box": koordinat, "class": nama_kelas})

    laporan = []
    nomor_orang = 0
    for person_box in deteksi_orang:
        nomor_orang += 1
        px1, py1, px2, py2 = map(int, person_box)
        kelas_terdeteksi = set()
        for deteksi in semua_deteksi_non_orang:
            if is_inside(deteksi["box"], person_box):
                kelas_terdeteksi.add(deteksi["class"])

        pelanggaran_ditemukan = []
        for apd in APD_WAJIB:
            kelas_pos = apd
            kelas_neg = f"NO-{apd}"
            kewajiban_terpenuhi = (kelas_pos in kelas_terdeteksi) and \
                                  (kelas_neg not in kelas_terdeteksi)
            if apd == "boot":
                kewajiban_terpenuhi = kelas_pos in kelas_terdeteksi
            if not kewajiban_terpenuhi:
                pelanggaran_ditemukan.append(f"Tidak Pakai {apd}")

        if not pelanggaran_ditemukan:
            warna, teks = (0, 255, 0), "Patuh"
        else:
            warna, teks = (0, 0, 255), f"Tidak Patuh: Person {nomor_orang}"
            laporan.append(f"Person {nomor_orang}: {', '.join(pelanggaran_ditemukan)}")

        cv2.rectangle(img, (px1, py1), (px2, py2), warna, 2)
        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = py1 - 10 if py1 - 10 > th else py1 + th + 10
        cv2.rectangle(img, (px1, ty - th - 5), (px1 + tw, ty + 5), warna, -1)
        cv2.putText(img, teks, (px1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if not deteksi_orang:
        laporan.append("Tidak ada pekerja terdeteksi")

    return img, laporan

# ==============================
# STREAMLIT APP
# ==============================
st.title("🚧 Deteksi Kepatuhan APD (YOLOv8)")

st.markdown(
    """
    <div style="background-color:rgba(0,0,0,0.2); padding:15px; border-radius:10px; border:1px solid #444; font-size:16px; line-height:1.6; color:#fff;">
        <em>
        Aplikasi website deteksi pelanggaran APD berbasis YOLOv8 adalah sistem berbasis web yang mampu mendeteksi secara otomatis penggunaan alat pelindung diri (APD) melalui gambar statis. 
        Dengan menggunakan model deteksi objek YOLOv8, aplikasi ini dapat mengidentifikasi apakah seseorang dalam gambar mengenakan APD sesuai standar, seperti:
        </em>
        <ul style="margin-top:10px;">
            <li>👷 Helm</li>
            <li>👷 Rompi Keselamatan</li>
            <li>🥾 Sepatu</li>
            <li>😷 Masker</li>
        </ul>
        <em>
        Pengguna cukup mengunggah gambar ke dalam sistem, dan aplikasi akan menganalisis serta menampilkan hasil deteksi, termasuk potensi pelanggaran APD. 
        Aplikasi ini bertujuan untuk membantu meningkatkan keselamatan kerja secara praktis dan efisien melalui analisis visual berbasis AI.
        </em>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="color: #FFD700; font-weight: bold; margin-top:20px; margin-bottom:10px; font-size:16px;">
        ⚠️ Saat ini aplikasi hanya mendukung deteksi berbasis <u>gambar statis</u>. 
        Fitur deteksi video/real-time belum dibuat.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = np.array(image)[:, :, ::-1].copy()  # convert to BGR for OpenCV

    hasil_img, laporan = analisis_gambar(img_cv)

    st.subheader("Hasil Deteksi")
    st.image(cv2.cvtColor(hasil_img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)

    st.subheader("Laporan Pelanggaran")
    if laporan:
        import pandas as pd

        data_laporan = []
        for l in laporan:
            person, detail = l.split(":", 1)
            data_laporan.append({
                "Person": person.strip(),
                "Status": "❌ Tidak Patuh",
                "Detail": detail.strip()
            })

        df_laporan = pd.DataFrame(data_laporan)
        st.table(df_laporan)  # atau pakai st.dataframe(df_laporan) kalau mau scroll
    else:
        st.success("✅ Semua pekerja lengkap APD")





