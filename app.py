import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Dashboard Prediksi Konsumsi Listrik Kota Makassar",
    layout="wide"
)

# css
st.markdown("""
<style>
.title { font-size: 2.2rem; font-weight: 700; text-align:center; margin-bottom:0.5rem }
.subtitle { text-align:center; font-size:1.1rem; margin-bottom:2rem }
.desc-box { border-radius:12px; padding:1.5rem; box-shadow:0 1px 6px rgba(0,0,0,0.08); margin-bottom:1.5rem }
.model-box { border-left:4px solid #0073e6; padding:1rem; border-radius:6px; margin-top:1rem; font-size:0.95rem }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title'>Dashboard Prediksi Konsumsi Listrik Kota Makassar</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analisis Konsumsi Listrik Bulanan Wilayah Utara & Selatan Menggunakan Model LSTM dan GRU</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🏠 Beranda", "📈 Prediksi"])

# fungsi utilitas
@st.cache_resource
def load_model_cached(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def predict_future(model, scaled_data, n_future, seq_len, min_val, max_val):
    """Prediksi berulang time series"""
    seq = scaled_data[-seq_len:].copy()
    preds = []

    for _ in range(n_future):
        pred = model.predict(seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        preds.append(pred)
        seq = np.vstack([seq[1:], [pred]])

    # inverse scaling
    preds_inv = np.array(preds) * (max_val - min_val) + min_val
    return preds_inv.tolist()

def render_plot(data, df_pred, title):
    """Plot all in one reusable function"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["tanggal"], y=data["konsumsi_kWh"],
        mode="lines+markers", name="Data Aktual"
    ))

    for col in df_pred.columns[1:]:
        fig.add_trace(go.Scatter(
            x=df_pred["Tanggal"], y=df_pred[col],
            mode="lines+markers", name=col, line=dict(dash="dash")
        ))

    fig.update_layout(title=title, xaxis_title="Tahun", yaxis_title="kWh", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# TAB 1 
with tab1:
    st.markdown("""
    <div class='desc-box'>
        Aplikasi ini dirancang untuk <b>menganalisis dan memprediksi konsumsi listrik</b> 
        menggunakan model <b>LSTM</b> dan <b>GRU</b>.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Fitur Utama Dashboard")
    st.markdown("""
    <div class='desc-box'>
    <ol>
        <li>Dataset konsumsi listrik aktual.</li>
        <li>Prediksi hingga 12 bulan ke depan.</li>
        <li>Pilihan model: <b>LSTM</b>, <b>GRU</b>, atau <b>Model Terbaik</b>.</li>
        <li>Perbandingan performa melalui MAPE.</li>
        <li>Grafik tren dan ekspor data prediksi.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# TAB 2 
with tab2:
    # Sidebar
    wilayah = st.sidebar.selectbox("Pilih Wilayah", ["Selatan", "Utara"])
    n_future = st.sidebar.slider("Jumlah Bulan Prediksi", 1, 12, 6)
    model_option = st.sidebar.selectbox("Pilih Model Prediksi", ["Model Terbaik", "LSTM", "GRU"])
    compare_models = st.sidebar.checkbox("Tampilkan Perbandingan LSTM vs GRU")

    # File paths
    base_dir = "."
    data_path = os.path.join(base_dir, "data", "data_listrik_bulanan_sltn.xlsx" if wilayah == "Selatan" else "data_listrik_bulanan_utara_winsor2.xlsx")
    result_dir = os.path.join(base_dir, "PLN_Data_Selatan" if wilayah == "Selatan" else "PLN_Data_Utara", "results_models_lstm_gru")

    # Load data
    data = pd.read_excel(data_path)
    st.subheader(f"📊 Data Konsumsi Listrik Wilayah {wilayah}")
    df_last = data.tail(12)
    st.dataframe(df_last, use_container_width=True)

    # Load model metadata
    rekap = pd.read_csv(os.path.join(result_dir, "rekap_semua_model.csv"))
    rekap["MAPE_Inv"] = rekap["MAPE_Inv"].str.replace("%", "").astype(float)
    best_row = rekap.iloc[rekap["MAPE_Inv"].idxmin()]

    lstm_row = rekap[rekap["Model"].str.contains("LSTM")].sort_values("MAPE_Inv").iloc[0]
    gru_row  = rekap[rekap["Model"].str.contains("GRU")].sort_values("MAPE_Inv").iloc[0]

    # Load model
    lstm_model = load_model_cached(os.path.join(result_dir, f"LSTM_batch{int(lstm_row['Batch'])}_epoch{int(lstm_row['Epochs'])}.h5"))
    gru_model  = load_model_cached(os.path.join(result_dir, f"GRU_batch{int(gru_row['Batch'])}_epoch{int(gru_row['Epochs'])}.h5"))

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['konsumsi_kWh']])
    min_val, max_val = scaler.data_min_[0], scaler.data_max_[0]

    # Generate future date
    future_dates = pd.date_range(data['tanggal'].iloc[-1] + pd.offsets.MonthBegin(), periods=n_future, freq='MS')

    # PERBANDINGAN MODEL
    if compare_models:
        preds_lstm = predict_future(lstm_model, scaled_data, n_future, 12, min_val, max_val)
        preds_gru  = predict_future(gru_model, scaled_data, n_future, 12, min_val, max_val)

        df_pred = pd.DataFrame({
            "Tanggal": future_dates,
            "Prediksi LSTM (kWh)": preds_lstm,
            "Prediksi GRU (kWh)": preds_gru
        })

        render_plot(data, df_pred, "Perbandingan Prediksi LSTM vs GRU")

    # SATU MODEL
    else:
        chosen_model_name = (
            best_row["Model"].split("_")[0] if model_option == "Model Terbaik" else model_option
        )
        chosen_model = lstm_model if chosen_model_name == "LSTM" else gru_model

        preds = predict_future(chosen_model, scaled_data, n_future, 12, min_val, max_val)

        df_pred = pd.DataFrame({
            "Tanggal": future_dates,
            f"Prediksi {chosen_model_name} (kWh)": preds
        })

        render_plot(data, df_pred, f"Prediksi Konsumsi Listrik Menggunakan {chosen_model_name}")

    # Export CSV 
    csv_name = f"prediksi_{wilayah.lower()}_{n_future}bulan.csv"
    st.download_button("⬇️ Unduh Hasil Prediksi (.csv)", df_pred.to_csv(index=False), file_name=csv_name)