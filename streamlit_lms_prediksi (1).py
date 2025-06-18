
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import io

st.set_page_config(page_title="Prediksi Kelulusan Pelatihan", layout="wide")
st.title("🎓 Prediksi Kelulusan Peserta Pelatihan")

uploaded_file = st.file_uploader("📥 Upload Dataset Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("🧾 Data Awal")
    st.dataframe(df.head())

    # Mencoba deteksi otomatis kolom numerik
    df_clean = df.select_dtypes(include=[np.number]).copy()
    if 'status_kelulusan' in df.columns:
        df_clean['status_kelulusan'] = df['status_kelulusan']

    if 'status_kelulusan' in df_clean.columns:
        st.subheader("📊 Prediksi Model")
        X = df_clean.drop(columns=['status_kelulusan'])
        y = df_clean['status_kelulusan']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        st.write("Hasil Evaluasi Model:")
        st.json(report)

        # Prediksi seluruh data
        df['prediksi_kelulusan'] = model.predict(X)

        st.subheader("📋 Hasil Prediksi")
        if 'Nama' in df.columns:
            st.dataframe(df[['Nama', 'prediksi_kelulusan']])
        else:
            st.dataframe(df[['prediksi_kelulusan']])

        # Unduh hasil
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button("📤 Download Hasil Prediksi", output.getvalue(), file_name="hasil_prediksi.xlsx")
    else:
        st.warning("❗ Kolom 'status_kelulusan' tidak ditemukan. Harap sertakan untuk melatih model.")
