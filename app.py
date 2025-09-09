import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==============================
# KONFIGURASI MODEL
# ==============================
MODEL_PATH = "Model/Utama/best (1).pt"  # Hanya menggunakan 1 model
NAMA_KELAS_ORANG = "Person"
APD_WAJIB = ["Hardhat", "Mask", "Safety Vest"]  # Boot sudah dihapus

# Load model (hanya 1 model)
model = YOLO(MODEL_PATH)

# ==============================
# Fungsi bantu
# ==============================
def is_inside(box_inner, box_outer):
    """Mengecek apakah box_inner berada di dalam box_outer"""
    cx = (box_inner[0] + box_inner[2]) / 2
    cy = (box_inner[1] + box_inner[3]) / 2
    return box_outer[0] < cx < box_outer[2] and box_outer[1] < cy < box_outer[3]

def analisis_gambar(img):
    """Analisis gambar menggunakan satu model saja"""
    # Jalankan deteksi dengan satu model
    hasil = model(img, verbose=False)[0]

    deteksi_orang, semua_deteksi_non_orang = [], []
    
    # Pisahkan deteksi person dan non-person
    for box in hasil.boxes:
        nama_kelas = model.names[int(box.cls)]
        koordinat = box.xyxy[0].cpu().numpy()
        
        if nama_kelas == NAMA_KELAS_ORANG:
            deteksi_orang.append(koordinat)
        else:
            semua_deteksi_non_orang.append({"box": koordinat, "class": nama_kelas})

    laporan = []
    nomor_orang = 0
    
    # Analisis setiap person yang terdeteksi
    for person_box in deteksi_orang:
        nomor_orang += 1
        px1, py1, px2, py2 = map(int, person_box)
        
        # Kumpulkan semua kelas yang terdeteksi dalam bounding box person
        kelas_terdeteksi = set()
        for deteksi in semua_deteksi_non_orang:
            if is_inside(deteksi["box"], person_box):
                kelas_terdeteksi.add(deteksi["class"])

        # Evaluasi kepatuhan APD
        pelanggaran_ditemukan = []
        for apd in APD_WAJIB:
            kelas_pos = apd  # contoh: "Hardhat"
            kelas_neg = f"NO-{apd}"  # contoh: "NO-Hardhat"
            
            # Logika evaluasi: harus ada kelas positif DAN tidak ada kelas negatif
            kewajiban_terpenuhi = (kelas_pos in kelas_terdeteksi) and \
                                  (kelas_neg not in kelas_terdeteksi)
            
            if not kewajiban_terpenuhi:
                pelanggaran_ditemukan.append(f"Tidak Pakai {apd}")

        # Tentukan status dan warna bounding box
        if not pelanggaran_ditemukan:
            warna, teks = (0, 255, 0), "Patuh"  # Hijau untuk patuh
        else:
            warna, teks = (0, 0, 255), f"Tidak Patuh: Person {nomor_orang}"  # Merah untuk tidak patuh
            laporan.append(f"Person {nomor_orang}: {', '.join(pelanggaran_ditemukan)}")

        # Gambar bounding box dan label
        cv2.rectangle(img, (px1, py1), (px2, py2), warna, 2)
        
        # Hitung ukuran teks untuk background
        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = py1 - 10 if py1 - 10 > th else py1 + th + 10
        
        # Gambar background untuk teks
        cv2.rectangle(img, (px1, ty - th - 5), (px1 + tw, ty + 5), warna, -1)
        
        # Tulis teks
        cv2.putText(img, teks, (px1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Jika tidak ada orang terdeteksi
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
            <li>👷 Helm (Hardhat)</li>
            <li>👷 Rompi Keselamatan (Safety Vest)</li>
            <li>😷 Masker (Mask)</li>
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

# File uploader
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load dan convert gambar
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = np.array(image)[:, :, ::-1].copy()  # convert to BGR for OpenCV

    # Analisis gambar
    hasil_img, laporan = analisis_gambar(img_cv)

    # Tampilkan hasil
    st.subheader("Hasil Deteksi")
    st.image(cv2.cvtColor(hasil_img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)

    st.subheader("Laporan Pelanggaran")
    if laporan:
        import pandas as pd

        data_laporan = []
        for l in laporan:
            if ":" in l:
                person, detail = l.split(":", 1)
                data_laporan.append({
                    "Person": person.strip(),
                    "Status": "❌ Tidak Patuh",
                    "Detail": detail.strip()
                })
            else:
                st.error(l.strip())

        if data_laporan:
            df_laporan = pd.DataFrame(data_laporan)
            st.table(df_laporan)
    else:
        st.success("✅ Semua pekerja lengkap APD")

# Tambahan info model
st.markdown("---")
st.markdown(
    """
    <div style="background-color:rgba(0,100,0,0.1); padding:10px; border-radius:5px; border-left:4px solid #00AA00;">
        <strong>ℹ️ Informasi Model:</strong><br>
        • Menggunakan model YOLOv8 dengan 7 kelas deteksi<br>
        • Kelas: Person, Mask, NO-Mask, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest<br>
        • Evaluasi kepatuhan berdasarkan keberadaan APD positif dan tidak adanya APD negatif<br>
        • APD yang dievaluasi: Hardhat, Mask, Safety Vest
    </div>
    """,
    unsafe_allow_html=True
)
