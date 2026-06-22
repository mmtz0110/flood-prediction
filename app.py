import logging
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Flood Prediction Dashboard", page_icon="🌊", layout="wide")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSS = """
<style>
.main{
    background-color:#f7f9fc;
}
.metric-card{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
}
.hero{
    padding:30px;
    border-radius:20px;
    background: linear-gradient(135deg,#0f172a,#2563eb);
    color:white;
    text-align:center;
}
.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}
</style>
"""


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading dataset %s", path)
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    logger.info("Loading model %s", path)
    return joblib.load(path)


def render_header() -> None:
    try:
        st.image("assets/banner.png", use_container_width=True)
    except Exception:
        logger.debug("Banner image not found")

    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
    <div class="hero">
    <h1>🌊 Flood Prediction Dashboard</h1>
    <p>Analisis Pola Banjir Menggunakan Machine Learning</p>
    <p>SDG 13 • Climate Action • Flood Mitigation</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        try:
            st.image("assets/logo.png", width=120)
        except Exception:
            logger.debug("Logo image not found")

        selected = option_menu(
            menu_title="Menu",
            options=["Dashboard", "Dataset", "Visualisasi", "Prediksi", "Evaluasi"],
            icons=["house", "table", "bar-chart", "cloud-rain", "graph-up"],
            menu_icon="cast",
            default_index=0,
        )

    return selected


def show_dashboard(df: pd.DataFrame) -> None:
    st.subheader("📊 Ringkasan Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jumlah Data", len(df))
    col2.metric("Jumlah Fitur", max(0, len(df.columns) - 1))
    col3.metric("Missing Value", int(df.isnull().sum().sum()))
    col4.metric("Target", "Flood")

    st.markdown("---")
    st.subheader("Preview Dataset")
    st.dataframe(df.head(20), use_container_width=True)


def show_dataset(df: pd.DataFrame) -> None:
    st.subheader("Dataset Explorer")
    st.dataframe(df)
    st.write(df.describe())


def show_visualisasi(df: pd.DataFrame) -> None:
    st.subheader("📈 Distribusi Data")
    target_col = df.columns[-1]

    fig = px.histogram(df, x=target_col, color=target_col, title="Distribusi Data Banjir")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df.corr(numeric_only=True), cmap="RdBu_r", annot=True, ax=ax)
    st.pyplot(fig2)


def show_prediksi(df: pd.DataFrame, model) -> None:
    st.subheader("🌧️ Prediksi Risiko Banjir")

    with st.form("prediksi"):
        fitur = {}
        cols = st.columns(2)

        for i, col in enumerate(df.columns[:-1]):
            default = float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
            fitur[col] = cols[i % 2].number_input(col, value=default)

        submit = st.form_submit_button("Prediksi")

    if submit:
        input_df = pd.DataFrame([fitur])
        try:
            hasil = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Model prediction error: {e}")
            return

        if int(hasil) == 1:
            st.error("⚠️ Potensi Banjir Tinggi")
        else:
            st.success("✅ Risiko Banjir Rendah")


def show_evaluasi(df: pd.DataFrame, model) -> None:
    st.subheader("🎯 Feature Importance")
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            st.info("Model does not expose `feature_importances_`.")
            return

        importance = pd.DataFrame({"Feature": df.columns[:-1], "Importance": importances})
        importance = importance.sort_values(by="Importance", ascending=False)

        fig = px.bar(importance.head(10), x="Importance", y="Feature", orientation="h", title="Top 10 Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating evaluation: {e}")


def main() -> None:
    st.title("🌊 Flood Prediction Dashboard")

    # Load resources
    try:
        df = load_data("data_banjir_combine_final.csv")
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return

    try:
        model = load_model("model.pkl")
    except Exception:
        logger.warning("Model not found or failed to load; prediction/evaluation disabled.")
        model = None

    render_header()
    selected = render_sidebar()

    if selected == "Dashboard":
        show_dashboard(df)
    elif selected == "Dataset":
        show_dataset(df)
    elif selected == "Visualisasi":
        show_visualisasi(df)
    elif selected == "Prediksi":
        if model is None:
            st.warning("Model tidak tersedia untuk prediksi.")
        else:
            show_prediksi(df, model)
    elif selected == "Evaluasi":
        if model is None:
            st.warning("Model tidak tersedia untuk evaluasi.")
        else:
            show_evaluasi(df, model)

    st.markdown("---")
    st.markdown(
        """
    <div class="footer">
    Flood Prediction Dashboard © 2026<br>
    Machine Learning Project | Teknik Informatika<br>
    Nusa Putra University
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
