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

st.markdown("""
<style>
.title {
    font-size: 2.2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.desc-box {
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

.model-box {
    border-left: 4px solid #0073e6;
    padding: 1rem;
    border-radius: 6px;
    margin-top: 1rem;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# HEADER UTAMA
st.markdown("<div class='title'>Dashboard Prediksi Konsumsi Listrik Kota Makassar</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analisis Konsumsi Listrik Bulanan Wilayah Utara & Selatan Menggunakan Model LSTM dan GRU</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🏠 Beranda", "📈 Prediksi"])

# TAB 1: BERANDA
with tab1:
    st.markdown("""
    <div class='desc-box'>
        Aplikasi ini dirancang untuk <b>menganalisis dan memprediksi konsumsi listrik bulanan Kota Makassar</b>
        dengan memanfaatkan dua model <b><i>Deep Learning</i></b>, yaitu 
        <b><i>Long Short-Term Memory</i> (LSTM)</b> dan <b><i>Gated Recurrent Unit</i> (GRU)</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
    """
    <h4 style='font-size:1.05rem; font-weight:600; margin-top:0; margin-bottom:0rem;'>
        Fitur Utama Dashboard
    </h4>
    """,
    unsafe_allow_html=True
    )   

    st.markdown("""
    <div class='desc-box'>
    <ol>
        <li>Menampilkan data konsumsi listrik aktual per wilayah.</li>
        <li>Prediksi konsumsi listrik hingga 12 bulan ke depan.</li>
        <li>Pilihan model prediksi: <b>LSTM</b>, <b>GRU</b>, atau <b>Model Terbaik</b>.</li>
        <li>Perbandingan performa antar model berdasarkan <b>MAPE</b>.</li>
        <li>Visualisasi tren konsumsi dan hasil prediksi.</li>
        <li>Ekspor hasil prediksi ke file <code>.csv</code> untuk analisis lanjutan.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# TAB 2: PREDIKSI
with tab2:
    icon_path = "icon/filter.png"
    base_dir = "."
    data_dir = os.path.join(base_dir, "data")
    result_selatan = os.path.join(base_dir, "PLN_Data_Selatan", "results_lstm_gru")
    result_utara = os.path.join(base_dir, "PLN_Data_Utara", "results_models_lstm_gru")

    dashboard_result_dir = os.path.join(base_dir, "results_dashboard")
    os.makedirs(dashboard_result_dir, exist_ok=True)

    # SIDEBAR
    col1, col2 = st.sidebar.columns([1, 7])
    with col1:
        st.image(icon_path, width=26)
    with col2:
        st.markdown(
            """
            <div style='display:flex; align-items:center; height:26px;'>
                <span style='font-size:1.05rem; font-weight:600;'>Filter Prediksi</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    wilayah = st.sidebar.selectbox("Pilih Wilayah", ["Selatan", "Utara"])
    n_future = st.sidebar.slider("Jumlah Bulan Prediksi", 1, 12, 6)
    model_option = st.sidebar.selectbox("Pilih Model Prediksi", ["Model Terbaik", "LSTM", "GRU"])
    compare_models = st.sidebar.checkbox("Tampilkan Perbandingan LSTM vs GRU")

    st.sidebar.markdown("""
    <hr style='border:0.5px solid #ccc; margin:13rem 0 0.3rem 0;'>
    <div style='text-align:center; color:#555; font-size:0.85rem; margin-top:0.3rem;'>
    Skripsi 2025 — H071221066<br>Universitas Hasanuddin
    </div>
    """, unsafe_allow_html=True)

    # DATA PATH
    data_path = os.path.join(
        data_dir, 
        "data_listrik_bulanan_sltn.xlsx" if wilayah == "Selatan" else "data_listrik_bulanan_utara_winsor2.xlsx"
    )
    result_dir = result_selatan if wilayah == "Selatan" else result_utara

    @st.cache_resource
    def load_model_cached(model_path):
        return tf.keras.models.load_model(model_path, compile=False)

    # LOAD DATA
    data = pd.read_excel(data_path)
    desired_order = ["tahun", "bulan", "tanggal", "konsumsi_kWh"]
    existing_cols = [col for col in desired_order if col in data.columns]
    data = data[existing_cols + [col for col in data.columns if col not in existing_cols]]

    st.subheader(f"📊 Data Konsumsi Listrik Wilayah {wilayah}")
    st.caption("Berikut 12 bulan terakhir konsumsi listrik yang digunakan sebagai dasar prediksi:")

    df_last = data.tail(12).copy()
    if "tahun" in df_last.columns:
        df_last["tahun"] = df_last["tahun"].astype(str)
        
    if "konsumsi_kWh" in df_last.columns:
        df_last["konsumsi_kWh"] = df_last["konsumsi_kWh"].apply(lambda x: f"{x:,.0f}")

    st.dataframe(df_last, use_container_width=True)
    st.divider()

    rekap_path = os.path.join(result_dir, "rekap_semua_model.csv")
    rekap = pd.read_csv(rekap_path)

    rekap["MAPE_Inv"] = (
        rekap["MAPE_Inv"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .astype(float)
    )

    best_model_row = rekap.loc[rekap["MAPE_Inv"].idxmin()]
    best_model_type = best_model_row["Model"]

    lstm_row = rekap[rekap["Model"].str.contains("LSTM")].sort_values("MAPE_Inv").iloc[0]
    gru_row  = rekap[rekap["Model"].str.contains("GRU")].sort_values("MAPE_Inv").iloc[0]

    lstm_model_path = os.path.join(result_dir, f"LSTM_batch{int(lstm_row['Batch'])}_epoch{int(lstm_row['Epochs'])}.h5")
    gru_model_path  = os.path.join(result_dir, f"GRU_batch{int(gru_row['Batch'])}_epoch{int(gru_row['Epochs'])}.h5")

    lstm_model = load_model_cached(lstm_model_path)
    gru_model  = load_model_cached(gru_model_path)

    # INFO MODEL TERBAIK
    st.subheader("🏆 Model Terbaik Wilayah")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", best_model_row["Model"])
    col2.metric("Batch", int(best_model_row["Batch"]))
    col3.metric("Epochs", int(best_model_row["Epochs"]))
    col4.metric("MAPE", f"{best_model_row['MAPE_Inv']:.2f}%")
    st.divider()

    # NORMALISASI
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['konsumsi_kWh']].astype(float))

    seq_len = 12
    min_kwh = scaler.data_min_[0]
    max_kwh = scaler.data_max_[0]

    last_date = data['tanggal'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=n_future, freq='MS')

    # PREDIKSI 
    if compare_models:
        preds_lstm, preds_gru = [], []
        seq_lstm = scaled_data[-seq_len:].copy()
        seq_gru  = scaled_data[-seq_len:].copy()

        for _ in range(n_future):
            pred_l = lstm_model.predict(seq_lstm.reshape(1, seq_len, 1), verbose=0)[0, 0]
            pred_g = gru_model.predict(seq_gru.reshape(1, seq_len, 1), verbose=0)[0, 0]

            preds_lstm.append(pred_l)
            preds_gru.append(pred_g)

            seq_lstm = np.vstack([seq_lstm[1:], np.array([[pred_l]])])
            seq_gru  = np.vstack([seq_gru[1:], np.array([[pred_g]])])

        preds_lstm_inv = [p * (max_kwh - min_kwh) + min_kwh for p in preds_lstm]
        preds_gru_inv  = [p * (max_kwh - min_kwh) + min_kwh for p in preds_gru]

        compare_df = pd.DataFrame({
            "Tanggal": future_dates,
            "Prediksi_LSTM (kWh)": preds_lstm_inv,
            "Prediksi_GRU (kWh)": preds_gru_inv
        })

        st.markdown(f"""
        <div class='model-box'>
        🔹 <b>LSTM</b>: Batch = {int(lstm_row['Batch'])}, Epoch = {int(lstm_row['Epochs'])}, MAPE = {float(lstm_row['MAPE_Inv']):.2f}%<br>
        🔸 <b>GRU</b>: Batch = {int(gru_row['Batch'])}, Epoch = {int(gru_row['Epochs'])}, MAPE = {float(gru_row['MAPE_Inv']):.2f}% 
        </div>
        """, unsafe_allow_html=True)

        # Tabel
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        df_show = compare_df.copy()
        df_show["Prediksi_LSTM (kWh)"] = df_show["Prediksi_LSTM (kWh)"].apply(lambda x: f"{x:,.0f}")
        df_show["Prediksi_GRU (kWh)"] = df_show["Prediksi_GRU (kWh)"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_show, use_container_width=True)
        fig = go.Figure()

        # Data aktual
        fig.add_trace(go.Scatter(
            x=data["tanggal"],
            y=data["konsumsi_kWh"],
            mode="lines+markers",
            name="Data Aktual",
            hovertemplate="<b>%{x|%d-%b-%Y}</b><br>Total: <b>%{y:,.0f} kWh</b><extra></extra>"
        ))

        # Prediksi LSTM
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=preds_lstm_inv,
            mode="lines+markers",
            name="Prediksi LSTM",
            line=dict(dash="dash"),
            hovertemplate="<b>%{x|%d-%b-%Y}</b><br>LSTM: <b>%{y:,.0f} kWh</b><extra></extra>"
        ))

        # Prediksi GRU
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=preds_gru_inv,
            mode="lines+markers",
            name="Prediksi GRU",
            line=dict(dash="dot"),
            hovertemplate="<b>%{x|%d-%b-%Y}</b><br>GRU: <b>%{y:,.0f} kWh</b><extra></extra>"
        ))

        fig.update_layout(
            title="Perbandingan Prediksi LSTM vs GRU",
            xaxis_title="Tahun",
            yaxis_title="Konsumsi (kWh)",
            hoverlabel=dict(bgcolor="white", font_size=14),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Simpan CSV
        csv_filename = f"prediksi_perbandingan_{wilayah.lower()}_{n_future}bulan.csv"
        csv_path = os.path.join(dashboard_result_dir, csv_filename)
        compare_df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            st.download_button(
                "⬇️ Unduh Hasil Perbandingan (.csv)",
                f,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True
            )
    else:
        chosen_model_name = (
            "LSTM" if model_option == "LSTM"
            else "GRU" if model_option == "GRU"
            else ("LSTM" if "LSTM" in best_model_type else "GRU")
        )
        chosen_row = lstm_row if chosen_model_name == "LSTM" else gru_row
        chosen_model = lstm_model if chosen_model_name == "LSTM" else gru_model

        seq = scaled_data[-seq_len:].copy()
        preds = []

        for _ in range(n_future):
            pred = chosen_model.predict(seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
            preds.append(pred)
            seq = np.vstack([seq[1:], np.array([[pred]])])

        preds_inv = [p * (max_kwh - min_kwh) + min_kwh for p in preds]
        pred_df = pd.DataFrame({
            "Tanggal": future_dates,
            f"Prediksi_{chosen_model_name} (kWh)": preds_inv
        })

        st.markdown(f"""
        <div class='model-box'>
        Model <b>{chosen_model_name}</b> dijalankan dengan konfigurasi:
        <ul>
            <li>Batch size: <b>{int(chosen_row['Batch'])}</b></li>
            <li>Epoch: <b>{int(chosen_row['Epochs'])}</b></li>
            <li>MAPE: <b>{float(chosen_row['MAPE_Inv']):.2f}%</b></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Tampilkan DataFrame
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        df_show = pred_df.copy()
        col_name = df_show.columns[1]
        df_show[col_name] = df_show[col_name].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_show, use_container_width=True)
        fig = go.Figure()

        # Data aktual
        fig.add_trace(go.Scatter(
            x=data["tanggal"],
            y=data["konsumsi_kWh"],
            mode="lines+markers",
            name="Data Aktual",
            hovertemplate="<b>%{x|%d-%b-%Y}</b><br>Total: <b>%{y:,.0f} kWh</b><extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=pred_df["Tanggal"],
            y=pred_df[col_name],
            mode="lines+markers",
            name=f"Prediksi {chosen_model_name}",
            line=dict(dash="dash"),
            hovertemplate="<b>%{x|%d-%b-%Y}</b><br>"
                        f"Prediksi {chosen_model_name}: <b>%{{y:,.0f}} kWh</b><extra></extra>"
        ))

        fig.update_layout(
            title=f"Prediksi Konsumsi Listrik Menggunakan {chosen_model_name}",
            xaxis_title="Tahun",
            yaxis_title="Konsumsi (kWh)",
            hoverlabel=dict(bgcolor="white", font_size=14),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Simpan CSV
        csv_filename = f"prediksi_{wilayah.lower()}_{chosen_model_name.lower()}_{n_future}bulan.csv"
        csv_path = os.path.join(dashboard_result_dir, csv_filename)
        pred_df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            st.download_button(
                "⬇️ Unduh Hasil Prediksi (.csv)",
                f,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True
            )