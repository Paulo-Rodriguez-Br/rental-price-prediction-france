import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from RentalCleaner import RentalCleaner
from RentalRegression import RentalRegression
from RentalStatsViews import RentalStatsViews

FILE_PATH = Path(__file__).resolve().parent / "scraping_outputs" / "rental_database.parquet"

UPLOAD_DIR = Path("uploaded_data")
UPLOAD_DIR.mkdir(exist_ok=True)

def train_model_for_app(regression: RentalRegression, tune: bool = False) -> None:
    regression.set_features()
    regression.train_test_split_data()
    regression.build_preprocessor()
    regression.build_model()
    regression.build_pipeline()

    if tune and regression.model_type == "random forest":
        regression.tune_hyperparameters()
    else:
        regression.fit()

    regression.store_results()
    
    regression.compute_permutation_importance()

def run_pipeline(path_to_file: str):

    RentalRegression.results_history.clear()
    RentalRegression.features_importance_history.clear()

    models_dict = {}

    cleaner = RentalCleaner(file_path=path_to_file)
    cleaner.run_cleaner()

    df_clean_initial = cleaner.dataframe.copy()

    lin_reg = RentalRegression(dataframe=df_clean_initial)
    train_model_for_app(lin_reg)
    models_dict[lin_reg.model_name] = lin_reg

    rf_default = RentalRegression(dataframe=df_clean_initial, model_type="random forest")
    train_model_for_app(rf_default)
    models_dict[rf_default.model_name] = rf_default

    rf_less = RentalRegression(
        dataframe=df_clean_initial,
        model_type="random forest",
        model_name="Random Forest without Rooms",
        numeric_features=["Surface"],
    )
    train_model_for_app(rf_less)
    models_dict[rf_less.model_name] = rf_less

    rf_no_prop = RentalRegression(
        dataframe=df_clean_initial,
        model_type="random forest",
        model_name="Random Forest without Property",
        categorical_features=["Dept"],
    )
    train_model_for_app(rf_no_prop)
    models_dict[rf_no_prop.model_name] = rf_no_prop

    cleaner.remove_outliers(lower_quantile=0.025, upper_quantile=0.975)
    df_clean_final = cleaner.dataframe.copy()

    rf_no_outliers = RentalRegression(
        dataframe=df_clean_final,
        model_type="random forest",
        model_name="Random Forest without Outliers (2.5/97.5)",
    )
    train_model_for_app(rf_no_outliers)
    models_dict[rf_no_outliers.model_name] = rf_no_outliers

    rf_tuned = RentalRegression(
        dataframe=df_clean_final,
        model_type="random forest",
        model_name="Random Forest without Outliers",
    )
    train_model_for_app(rf_tuned, tune=True)
    models_dict[rf_tuned.model_name] = rf_tuned

    df_res = pd.DataFrame(RentalRegression.results_history)
    df_imp = pd.DataFrame(RentalRegression.features_importance_history)

    return df_clean_initial, df_clean_final, df_res, df_imp, models_dict

def main():

    st.set_page_config(page_title="Rental Models", layout="wide")
    st.title("🏠 Rental Princing Prediction Model")

    defaults = {
        "pipeline_ready": False,
        "df_clean_initial": None,
        "df_clean_final": None,
        "df_results": None,
        "df_importance": None,
        "models_dict": None,
        "file_path": FILE_PATH,
        "active_page": "Onboarding",
        "shap_fig1": None,
        "shap_fig2": None,
        "shap_model_key": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.pipeline_ready:
        st.sidebar.title("📌 Navigation")
        st.session_state.active_page = st.sidebar.radio(
            "Select a section",
            [
                "🏁 Regression Dashboard",
                "📈 Exploratory Views",
                "🔮 Price Prediction",
            ],
        )

    if not st.session_state.pipeline_ready:

        st.header("🚀 Onboarding")

        uploaded = st.file_uploader("Upload dataset (CSV or Parquet)", type=["csv", "parquet"])

        if uploaded:
            temp_path = UPLOAD_DIR / "uploaded_dataset.parquet"

            if uploaded.name.endswith(".csv"):
                pd.read_csv(uploaded).to_parquet(temp_path, index=False)
            else:
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())

            st.session_state.file_path = str(temp_path)
            st.success("Custom file loaded!")

        st.info(f"Using file: {st.session_state.file_path}")

        if st.button("Run pipeline", type="primary"):
            with st.spinner("Running full pipeline..."):
                df_init, df_final, df_res, df_imp, models = run_pipeline(
                    st.session_state.file_path
                )

            st.session_state.df_clean_initial = df_init
            st.session_state.df_clean_final = df_final
            st.session_state.df_results = df_res
            st.session_state.df_importance = df_imp
            st.session_state.models_dict = models
            st.session_state.pipeline_ready = True
            st.session_state.active_page = "🏁 Regression Dashboard"

            st.success("Pipeline complete!")
            st.rerun()

        return

    # ----------------------------------------------
    # PAGE 1 — REGRESSION DASHBOARD
    # ----------------------------------------------
    if st.session_state.active_page == "🏁 Regression Dashboard":

        df_res = st.session_state.df_results
        df_imp = st.session_state.df_importance
        models_dict = st.session_state.models_dict
        df_final = st.session_state.df_clean_final

        st.header("📊 Regression Models Results")

        st.subheader("Cleaned Dataset")
        st.dataframe(df_final.head(50))

        st.subheader("Model Performance")
        st.dataframe(df_res)

        st.subheader("Feature Importance")

        model_name = st.selectbox(
            "Select model",
            sorted(df_imp["Model"].unique()),
            key="fi_select",
        )

        df_m = df_imp[df_imp["Model"] == model_name].sort_values(
            "Importance_mean", ascending=False
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df_m["Feature"], df_m["Importance_mean"])
        ax.set_title(f"Permutation Feature Importance — {model_name}")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance (mean decrease in R²)")
        ax.set_xticklabels(df_m["Feature"], rotation=45)
        st.pyplot(fig)

        # RESIDUAL PLOTS
        st.subheader("Residual Plots")

        model_name_resid = st.selectbox(
            "Select model for residuals",
            sorted(models_dict.keys()),
            key="resid_select",
        )

        if st.button("Show residual plot"):
            reg = models_dict[model_name_resid]
            fig_resid, _ = reg.plot_residuals(return_fig=True)
            st.pyplot(fig_resid)

        # -----------------------------------------
        # SHAP
        # -----------------------------------------
        st.subheader("🔍 SHAP Analysis")
        
        st.warning(
            "⚠️ SHAP computation is computationally expensive and may temporarily freeze "
            "the user interface. It is recommended to run this analysis only once per model."
        )
        
        rf_models = {
            name: m
            for name, m in models_dict.items()
            if m.model_type.lower() == "random forest"
        }
        
        if rf_models:
            shap_model_name = st.selectbox(
                "Select Random Forest model for SHAP analysis",
                sorted(rf_models.keys()),
                key="shap_select",
            )
        
            reg_shap = rf_models[shap_model_name]
        
            cached = (
                st.session_state.get("shap_model_key") == shap_model_name
                and st.session_state.get("shap_fig1") is not None
                and st.session_state.get("shap_fig2") is not None
            )
        
            if cached:
                st.success(f"Loaded cached SHAP plots for model: {shap_model_name}")
                st.pyplot(st.session_state.shap_fig1)
                st.pyplot(st.session_state.shap_fig2)
        
            else:
                st.info(
                    "SHAP plots have not been generated yet for this model. "
                    "Click the button below to compute them."
                )
        
                if st.button(
                    "Generate SHAP plots (may freeze UI)",
                    key="shap_generate_btn",
                ):
                    with st.spinner("Computing SHAP values... Please wait."):
                        fig1, fig2 = reg_shap.run_shap()
        
                    # Cache results
                    st.session_state.shap_model_key = shap_model_name
                    st.session_state.shap_fig1 = fig1
                    st.session_state.shap_fig2 = fig2
        
                    st.pyplot(fig1)
                    st.pyplot(fig2)


    # ----------------------------------------------
    # PAGE 2 — EXPLORATORY VIEWS
    # ----------------------------------------------
    if st.session_state.active_page == "📈 Exploratory Views":

        st.header("🔍 Exploratory Data Views")

        views = RentalStatsViews(st.session_state.df_clean_initial)

        if st.button("Generate exploratory views"):
            figs = views.execute_views()
            for fig in figs:
                st.pyplot(fig)

    # ----------------------------------------------
    # PAGE 3 — PRICE PREDICTION (ADDED)
    # ----------------------------------------------
    if st.session_state.active_page == "🔮 Price Prediction":

        st.header("🔮 Rental Price Prediction (Tuned Model)")

        MODEL_KEY = "Random Forest without Outliers (tuned)"

        models_dict = st.session_state.models_dict
        df_final = st.session_state.df_clean_final

        if MODEL_KEY not in models_dict:
            st.error("Tuned model not available. Run the pipeline first.")
            st.stop()

        model = models_dict[MODEL_KEY]

        st.subheader("Property characteristics")

        col1, col2 = st.columns(2)

        with col1:
            dept = st.selectbox(
                "Department",
                sorted(df_final["Dept"].dropna().unique())
            )
            dept = str(dept)

            rooms = st.number_input(
                "Rooms",
                min_value=1,
                max_value=10,
                value=2
            )

        with col2:
            surface = st.number_input(
                "Surface (m²)",
                min_value=10,
                max_value=500,
                value=50
            )

            property_type = st.selectbox(
                "Property Type",
                sorted(df_final["Property Type"].dropna().unique())
            )

        if st.button("Predict price", type="primary"):

            observation = {
                "Dept": dept,
                "Rooms": rooms,
                "Surface": surface,
                "Property Type": property_type,
            }

            log_price = model.predict_single(observation)
            price = np.exp(log_price)

            st.success(f"💰 Estimated rent: **{price:,.0f} € / month**")

if __name__ == "__main__":
    main()
