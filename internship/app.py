import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
st.title("Analisis Data Magang dengan Machine Learning")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Pratinjau Data")
    st.dataframe(df.head())

    st.subheader("Informasi Dataset")
    st.write("Jumlah baris dan kolom:", df.shape)
    st.write("Tipe data:")
    st.write(df.dtypes)
    st.write("Cek nilai kosong:")
    st.write(df.isnull().sum())

    # Label Encoding
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        st.subheader("Encoding Kolom Kategorikal")
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        st.success("Label encoding selesai.")
    
    # Drop kolom id/label jika ada
    drop_cols = [col for col in df.columns if 'id' in col.lower() or 'label' in col.lower()]
    X = df.drop(columns=drop_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

    st.write("Ukuran data latih:", X_train.shape)
    st.write("Ukuran data uji:", X_test.shape)

    # Jika label tersedia, latih model
    if drop_cols:
        y = df[drop_cols[0]]
        y_train, y_test = train_test_split(y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.subheader("Hasil Evaluasi Model")
        st.write("Akurasi:", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Kolom label tidak ditemukan. Tidak bisa melatih model.")
