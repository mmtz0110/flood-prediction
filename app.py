import logging
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Flood Prediction Dashboard", layout="wide")

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

    st.markdown("<div class="hero"></div>",unsafe_allow_html=True,)


def render_sidebar():
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

        st.markdown("---")
        train_btn = st.button("Latih Model (Train model)")

    return selected, train_btn


def show_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Ringkasan Dataset")

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
    st.subheader("Distribusi Data")
    target_col = df.columns[-1]

    fig = px.histogram(df, x=target_col, color=target_col, title="Distribusi Data Banjir")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df.corr(numeric_only=True), cmap="RdBu_r", annot=True, ax=ax)
    st.pyplot(fig2)


def show_prediksi(df: pd.DataFrame, model) -> None:
    st.subheader("Prediksi Risiko Banjir")
    # define features to use for prediction
    categorical_features = ["landcover_class"]
    numeric_features = [
        "avg_rainfall",
        "max_rainfall",
        "avg_temperature",
        "elevation",
        "ndvi",
        "slope",
        "soil_moisture",
        "month",
    ]

    # allow simple vs advanced input modes
    mode = st.radio("Pilih mode input:", ["Sederhana (direkomendasikan)", "Lanjutan"], index=0)

    def quantile_value(col: str, level: str):
        # level in {low, medium, high}
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            return 0.0
        if level == "Low":
            q = df[col].quantile(0.25)
        elif level == "Medium":
            q = df[col].quantile(0.5)
        else:
            q = df[col].quantile(0.75)
        return float(q)

    input_data = {}

    with st.form("prediksi"):
        if mode.startswith("Sederhana"):
            c1, c2 = st.columns(2)
            rain = c1.selectbox("Curah hujan", ["Low", "Medium", "High"], index=1)
            temp = c1.selectbox("Suhu rata-rata", ["Low", "Medium", "High"], index=1)
            elev = c2.selectbox("Ketinggian", ["Low", "Medium", "High"], index=1)
            soil = c2.selectbox("Kelembaban tanah", ["Low", "Medium", "High"], index=1)
            slope_sel = c1.selectbox("Kemiringan tanah", ["Flat", "Gentle", "Steep"], index=1)
            month_sel = c2.selectbox("Bulan", list(range(1, 13)), index=0)
            # landcover simplified
            land_opt = list(df["landcover_class"].dropna().unique()) if "landcover_class" in df.columns else ["Built-up", "Tree cover"]
            land = c1.selectbox("Tipe lahan (landcover)", land_opt)

            # map simple selections to numeric features
            input_data["avg_rainfall"] = quantile_value("avg_rainfall", rain)
            input_data["max_rainfall"] = quantile_value("max_rainfall", rain)
            input_data["avg_temperature"] = quantile_value("avg_temperature", temp)
            input_data["elevation"] = quantile_value("elevation", elev)
            input_data["soil_moisture"] = quantile_value("soil_moisture", soil)
            # map slope
            slope_map = {"Flat": 0.2, "Gentle": 3.0, "Steep": 7.0}
            input_data["slope"] = slope_map.get(slope_sel, 1.0)
            input_data["month"] = int(month_sel)
            input_data["ndvi"] = quantile_value("ndvi", "Medium")
            input_data["landcover_class"] = land
        else:
            # advanced: show numeric fields
            cols = st.columns(2)
            for i, feat in enumerate(numeric_features):
                default = float(df[feat].mean()) if feat in df.columns else 0.0
                input_data[feat] = cols[i % 2].number_input(feat, value=default)

            for cat in categorical_features:
                if cat in df.columns:
                    options = list(df[cat].dropna().unique())
                    input_data[cat] = st.selectbox(cat, options)
                else:
                    input_data[cat] = "Unknown"

        submit = st.form_submit_button("Prediksi")

    if submit:
        input_df = pd.DataFrame([input_data])

        if model is None:
            st.error("Model belum tersedia. Aplikasi sedang melatih model otomatis atau tekan 'Latih Model' di sidebar.")
            return

        try:
            hasil = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Model prediction error: {e}")
            return

        if int(hasil) == 1:
            st.error("Potensi Banjir Tinggi")
        else:
            st.success("Risiko Banjir Rendah")


def train_and_save_model(df: pd.DataFrame, model_path: str = "model.pkl"):
    """Simple training pipeline: one-hot encode categorical, scale numeric, train RandomForest."""
    # ensure target exists
    if "banjir" not in df.columns:
        raise ValueError("Target column 'banjir' not found in dataset")

    # features
    cat_feats = ["landcover_class"]
    num_feats = [
        "avg_rainfall",
        "max_rainfall",
        "avg_temperature",
        "elevation",
        "ndvi",
        "slope",
        "soil_moisture",
        "month",
    ]

    available_num = [c for c in num_feats if c in df.columns]
    available_cat = [c for c in cat_feats if c in df.columns]

    X = df[available_num + available_cat].copy()
    y = df["banjir"].astype(int)

    # simple preprocessing
    transformers = []
    if available_num:
        transformers.append(("num", StandardScaler(), available_num))
    if available_cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), available_cat))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline(
        steps=[("preproc", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # save
    joblib.dump(pipeline, model_path)

    return pipeline, acc, classification_report(y_test, y_pred)


def show_evaluasi(df: pd.DataFrame, model) -> None:
    st.subheader("Feature Importance")
    try:
        # Try to extract feature importances from pipeline or estimator
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            importances = model.named_steps["clf"].feature_importances_
        else:
            importances = getattr(model, "feature_importances_", None)

        if importances is None:
            st.info("Model does not expose `feature_importances_`.")
            return

        # build feature names consistent with training preprocessing
        num_feats = [
            "avg_rainfall",
            "max_rainfall",
            "avg_temperature",
            "elevation",
            "ndvi",
            "slope",
            "soil_moisture",
            "month",
        ]
        cat_feats = ["landcover_class"]

        available_num = [c for c in num_feats if c in df.columns]
        available_cat = [c for c in cat_feats if c in df.columns]

        feature_names = []
        feature_names.extend(available_num)
        # expand categorical as one-hot names using observed categories
        for cat in available_cat:
            cats = list(df[cat].dropna().unique())
            cats = [str(x) for x in cats]
            cats_sorted = sorted(cats)
            feature_names.extend([f"{cat}__{v}" for v in cats_sorted])

        # trim or pad feature names to match importances length
        if len(feature_names) > len(importances):
            feature_names = feature_names[: len(importances)]
        elif len(feature_names) < len(importances):
            # pad with generic names
            feature_names.extend([f"f_{i}" for i in range(len(importances) - len(feature_names))])

        importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance = importance.sort_values(by="Importance", ascending=False)

        fig = px.bar(importance.head(10), x="Importance", y="Feature", orientation="h", title="Top 10 Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating evaluation: {e}")


def main() -> None:
    st.title("Flood Prediction Dashboard")

    # Load resources
    try:
        df = load_data("data_banjir_combine_final.csv")
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return

    try:
        model = load_model("model.pkl")
    except Exception:
        logger.warning("Model not found or failed to load; training will start automatically.")
        model = None

    render_header()
    selected, train_requested = render_sidebar()

    # Auto-train if model not available (on app start)
    if model is None:
        with st.spinner("Model tidak ditemukan — melatih model secara otomatis, mohon tunggu..."):
            try:
                pipeline, acc, report = train_and_save_model(df)
                st.success(f"Auto-training selesai — akurasi: {acc:.3f}")
                st.text("Classification report:")
                st.text(report)
                model = pipeline
            except Exception as e:
                st.error(f"Auto-training gagal: {e}")
                model = None

    # if user requested training from sidebar, allow re-training
    if train_requested:
        with st.spinner("Melatih model... Ini mungkin memakan waktu beberapa detik"):
            try:
                pipeline, acc, report = train_and_save_model(df)
                st.success(f"Training selesai — akurasi: {acc:.3f}")
                st.text("Classification report:")
                st.text(report)
                model = pipeline
            except Exception as e:
                st.error(f"Training gagal: {e}")

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
