
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import io
import matplotlib.pyplot as plt

st.title("📊 LMS - Model Regresi: Prediksi Status Kelulusan Pelatihan Excel")

st.write("""
Model ini memprediksi status kelulusan peserta pelatihan Excel berdasarkan:
- Persentase Partisipasi
- Persentase Kehadiran
""")

# Upload dataset
uploaded_file = st.file_uploader("Unggah File Dataset (Excel)", type=["xlsx"])
if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Pilih Sheet", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Validasi kolom
    required_cols = ['Partisipasi', 'Kehadiran', 'status_kelulusan']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset harus mengandung kolom: {', '.join(required_cols)}")
        st.stop()

    # Preprocessing
    try:
        df['Partisipasi'] = df['Partisipasi'].str.replace('%', '').astype(float)
    except:
        st.warning("Kolom 'Partisipasi' tidak dalam format persen. Pastikan isinya seperti '80.00%'")
        st.stop()

    df['status_kelulusan'] = df['status_kelulusan'].map({'Lulus': 1, 'Tidak Lulus': 0})

    X = df[['Partisipasi', 'Kehadiran']]
    y = df['status_kelulusan']

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Input manual
    st.sidebar.header("Input Manual untuk Prediksi")
    partisipasi = st.sidebar.slider("Persentase Partisipasi", 0.0, 100.0, 50.0)
    kehadiran = st.sidebar.slider("Persentase Kehadiran", 0.0, 100.0, 50.0)

    input_data = np.array([[partisipasi, kehadiran]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Hasil Prediksi Input Manual:")
    if pred == 1:
        st.success(f"✅ Peserta Diprediksi LULUS (Probabilitas: {prob:.2%})")
    else:
        st.error(f"❌ Peserta Diprediksi TIDAK LULUS (Probabilitas: {prob:.2%})")

    # Prediksi batch
    df['Prediksi'] = model.predict(X)
    df['Prob_Lulus'] = model.predict_proba(X)[:, 1]

    # Filter tampilan
    st.subheader("🔎 Filter Data")
    status_filter = st.selectbox("Tampilkan Status Kelulusan", options=["Semua", "Lulus", "Tidak Lulus"])
    min_part = st.slider("Minimal Partisipasi", 0.0, 100.0, 0.0)
    min_kehadiran = st.slider("Minimal Kehadiran", 0.0, 100.0, 0.0)

    df_filtered = df[(df['Partisipasi'] >= min_part) & (df['Kehadiran'] >= min_kehadiran)]
    if status_filter == "Lulus":
        df_filtered = df_filtered[df_filtered['status_kelulusan'] == 1]
    elif status_filter == "Tidak Lulus":
        df_filtered = df_filtered[df_filtered['status_kelulusan'] == 0]

    st.subheader("📄 Tabel Data dan Prediksi")
    st.dataframe(df_filtered[['ID_Peserta', 'Nama', 'Partisipasi', 'Kehadiran', 'status_kelulusan', 'Prediksi', 'Prob_Lulus']])

    # Visualisasi
    st.subheader("📊 Visualisasi Kelulusan")
    chart_data = df['status_kelulusan'].value_counts().rename({0: 'Tidak Lulus', 1: 'Lulus'})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    chart_data.plot(kind='bar', ax=ax[0], color=['red', 'green'])
    ax[0].set_title("Jumlah Kelulusan")
    ax[0].set_ylabel("Jumlah Peserta")
    chart_data.plot(kind='pie', labels=chart_data.index, autopct='%1.1f%%', ax=ax[1])
    ax[1].set_ylabel("")
    ax[1].set_title("Distribusi Kelulusan")
    st.pyplot(fig)

    # Koefisien model
    st.subheader("🧠 Koefisien Model Regresi Logistik")
    coef_df = pd.DataFrame({
        'Fitur': ['Partisipasi', 'Kehadiran'],
        'Koefisien': model.coef_[0]
    })
    st.table(coef_df)

    # Download hasil
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Prediksi')
        writer.save()
        st.download_button(
            label="📥 Unduh Hasil Prediksi dalam Excel",
            data=output.getvalue(),
            file_name="hasil_prediksi_kelulusan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Evaluasi model
    st.subheader("📊 Evaluasi Model")
    y_pred = model.predict(X)
    st.text(classification_report(y, y_pred, target_names=['TIDAK LULUS', 'LULUS']))

    # Simpan log
    log_data = df[['ID_Peserta', 'Nama', 'Partisipasi', 'Kehadiran', 'Prediksi', 'Prob_Lulus']]
    log_output = io.BytesIO()
    with pd.ExcelWriter(log_output, engine='xlsxwriter') as writer:
        log_data.to_excel(writer, index=False, sheet_name='Log_Prediksi')
        writer.save()
        st.download_button(
            label="📁 Unduh Log Prediksi",
            data=log_output.getvalue(),
            file_name="log_riwayat_prediksi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.warning("Silakan unggah file dataset Excel untuk memulai prediksi.")
