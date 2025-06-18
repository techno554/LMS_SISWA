
# 🎓 LMS Prediksi Kelulusan Pelatihan - Streamlit App

Aplikasi berbasis Streamlit untuk memprediksi kelulusan peserta pelatihan menggunakan data dari file Excel.

## 🚀 Fitur
- Upload file Excel peserta pelatihan
- Tampilkan data awal
- Preprocessing otomatis
- Pelatihan model prediksi menggunakan RandomForest
- Evaluasi model
- Prediksi seluruh peserta
- Download hasil prediksi

## 🧾 Format Dataset
Dataset Excel harus mengandung kolom-kolom numerik seperti:
- total_login
- materi_selesai
- skor_kuis_rata2
- partisipasi_forum
- durasi_total_akses
- interaksi_mingguan
- jumlah_tugas_dikumpulkan
- frekuensi_kuis
- aktivitas_mobile
- status_kelulusan (target klasifikasi)

## 📦 Cara Menjalankan di Lokal
```bash
pip install streamlit scikit-learn pandas openpyxl xlsxwriter
streamlit run streamlit_lms_prediksi.py
```

## ☁️ Deploy ke Streamlit Cloud
1. Fork atau upload project ini ke GitHub.
2. Buka: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Login & klik **'New App'**
4. Pilih repo dan branch yang berisi file `streamlit_lms_prediksi.py`
5. Jalankan! 🎉

---

© 2025 LMS Prediksi Kelulusan | Dibuat dengan ❤️ oleh OpenAI ChatGPT
