import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Saham Netflix", layout="centered")

st.title("🎬 Prediksi Harga Saham Netflix (NFLX)")
st.write("Aplikasi berbasis LSTM untuk memprediksi harga penutupan saham besok.")

# --- LOAD MODEL ---
@st.cache_resource
def load_trained_model():
    # Pastikan nama file sesuai dengan yang kamu download
    model = load_model('model_netflix_lstm.h5')
    return model

try:
    model = load_trained_model()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .h5 ada di folder yang sama. Error: {e}")

# --- INPUT USER ---
st.sidebar.header("Konfigurasi Data")
days_back = st.sidebar.slider("Lihat data historis berapa hari?", 30, 365, 90)

# --- PROSES UTAMA ---
if st.button('Mulai Prediksi'):
    with st.spinner('Sedang mengambil data terbaru dari Yahoo Finance...'):
        
        # 1. Ambil data Real-Time hari ini
        # Kita ambil data cukup banyak ke belakang untuk keperluan scaling yang akurat
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=500)
        
        df = yf.download('NFLX', start=start_date, end=end_date)
        
        if len(df) == 0:
            st.error("Gagal mengambil data. Cek koneksi internet.")
        else:
            # Fix struktur data yfinance (jika multiindex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Tampilkan Grafik Data Terakhir
            st.subheader(f"Grafik Harga {days_back} Hari Terakhir")
            st.line_chart(df['Close'].tail(days_back))
            
            # --- PERSIAPAN DATA UNTUK PREDIKSI ---
            data = df.filter(['Close'])
            dataset = data.values
            
            # Scaling (Wajib sama persis dengan proses training)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            
            # Ambil 60 hari terakhir (Window)
            # Ini adalah input yang dibutuhkan model untuk menebak "besok"
            last_60_days = scaled_data[-60:]
            
            # Ubah bentuk ke 3D [1, 60, 1]
            X_test = []
            X_test.append(last_60_days)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # --- PREDIKSI MODEL ---
            pred_price = model.predict(X_test)
            
            # Kembalikan ke harga asli (USD)
            pred_price_usd = scaler.inverse_transform(pred_price)
            final_price = pred_price_usd[0][0]
            
            # --- TAMPILKAN HASIL ---
            st.markdown("---")
            st.header("🔮 Hasil Prediksi")
            st.metric(label="Perkiraan Harga Penutupan Besok", value=f"USD {final_price:.2f}")
            
            # Info tambahan
            last_actual_price = df['Close'].iloc[-1]
            diff = final_price - last_actual_price
            
            # PERBAIKAN DI SINI: Ganti '$' dengan 'USD' atau escape '\$'
            if diff > 0:
                st.write(f"Model memprediksi harga akan **NAIK** sebesar **USD {diff:.2f}** dari harga penutupan terakhir (USD {last_actual_price:.2f}).")
            else:
                st.write(f"Model memprediksi harga akan **TURUN** sebesar **USD {abs(diff):.2f}** dari harga penutupan terakhir (USD {last_actual_price:.2f}).")