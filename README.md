
# 📊 Aplikasi Prediksi Kelulusan Pelatihan Excel

Aplikasi ini menggunakan **regresi logistik** untuk memprediksi status kelulusan peserta pelatihan Excel berbasis data partisipasi dan kehadiran. Dibuat menggunakan **Streamlit**.

## 🚀 Fitur Utama

- Upload dataset Excel dan pilih sheet
- Validasi kolom penting: `Partisipasi`, `Kehadiran`, dan `status_kelulusan`
- Preprocessing otomatis (konversi persen, encoding label)
- Prediksi kelulusan secara manual dan batch
- Filter berdasarkan partisipasi, kehadiran, dan status kelulusan
- Visualisasi grafik batang dan pie
- Koefisien model dan evaluasi model
- Unduh hasil prediksi dan log sebagai file Excel

## 📦 Instalasi

1. Clone repositori atau download file
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run regresi_kelulusan_excel.py
   ```

## 📁 Format Dataset

Pastikan dataset Anda memiliki kolom berikut:

| ID_Peserta | Nama | Partisipasi | Kehadiran | status_kelulusan |
|------------|------|-------------|-----------|------------------|
| MHS001     | Andi | 85.0%       | 90.0      | Lulus            |
| MHS002     | Budi | 70.0%       | 65.0      | Tidak Lulus      |

## 🌐 Deploy ke Streamlit Cloud

1. Push semua file ke GitHub.
2. Buka [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Hubungkan ke repo Anda.
4. Jalankan aplikasi.

---

© 2025 – Dibuat untuk keperluan analisis kelulusan pelatihan berbasis data.
