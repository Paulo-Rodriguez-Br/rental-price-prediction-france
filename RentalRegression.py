from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RentalRegression:
    """
    Regression pipeline for rental price modeling.
    """

    results_history: ClassVar[List[Dict]] = []
    features_importance_history: ClassVar[List[Dict]] = []

    dataframe: Optional[pd.DataFrame] = None

    model: Optional[object] = None
    model_type: Optional[str] = "linear regression"
    preprocessor: Optional[object] = None
    pipeline: Optional[Pipeline] = None

    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None

    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    target_column: str = "log (Price)"

    model_name: str = ""
    is_trained: bool = field(default=False, init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize logger and logging directory.
        """
        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()

        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "regression.log"

        self.logger = logging.getLogger(f"{__name__}.RentalRegression")
        self.logger.setLevel(logging.INFO)

        if not any(
            isinstance(h, logging.FileHandler)
            for h in self.logger.handlers
        ):
            handler = logging.FileHandler(
                log_file,
                mode="a",
                encoding="utf-8",
            )
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(
            "RentalRegression initialized (model_type=%s).",
            self.model_type,
        )

    def set_features(self) -> None:
        """
        Define numeric and categorical features.
        """
        if self.numeric_features is None:
            self.numeric_features = ["Rooms", "Surface"]

        if self.categorical_features is None:
            self.categorical_features = ["Dept", "Property Type"]

        self.logger.info(
            "Features defined: numeric=%s | categorical=%s",
            self.numeric_features,
            self.categorical_features,
        )

    def train_test_split_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Split dataset into training and testing sets.
        """
        if self.dataframe is None:
            raise ValueError("dataframe must not be None before splitting.")

        X = self.dataframe[self.numeric_features + self.categorical_features]
        y = self.dataframe[self.target_column]

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        self.logger.info(
            "Train/test split completed. Train=%s | Test=%s",
            self.X_train.shape,
            self.X_test.shape,
        )

    def build_preprocessor(self) -> None:
        """
        Build preprocessing pipeline.
        """
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        self.logger.info("Preprocessor built successfully.")

    def build_model(self) -> None:
        """
        Instantiate regression model.
        """
        if self.model_type == "linear regression":
            self.model = LinearRegression()
            if not self.model_name:
                self.model_name = "Linear Regression"

        elif self.model_type == "random forest":
            self.model = RandomForestRegressor()
            if not self.model_name:
                self.model_name = "Random Forest"

        else:
            raise ValueError("Unknown model_type.")

        self.logger.info("Model built: %s", self.model_name)

    def build_pipeline(self) -> None:
        """
        Assemble preprocessing and model into a pipeline.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not defined.")
        if self.model is None:
            raise ValueError("Model is not defined.")

        self.pipeline = Pipeline(
            steps=[
                ("preprocessing", self.preprocessor),
                ("model", self.model),
            ]
        )

        self.logger.info("Pipeline built successfully.")

    def fit(self) -> None:
        """
        Fit the pipeline to training data.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not built.")
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data is not defined.")

        self.pipeline.fit(self.X_train, self.y_train)
        self.is_trained = True

        self.logger.info("Model trained successfully.")

    def tune_hyperparameters(
        self,
        n_iter: int = 20,
        cv: int = 3,
        random_state: int = 42,
    ) -> None:
        """
        Perform randomized hyperparameter tuning for Random Forest.
        """
        if self.model_type != "random forest":
            raise ValueError(
                "Hyperparameter tuning only applies to Random Forest."
            )

        if self.pipeline is None:
            raise ValueError("Pipeline is not built.")

        param_distributions = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2", 0.5],
        }

        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

        search.fit(self.X_train, self.y_train)

        self.pipeline = search.best_estimator_
        self.model = self.pipeline.named_steps["model"]
        self.model_name += " (tuned)"
        self.is_trained = True

        self.logger.info("Hyperparameter tuning completed.")
        self.logger.info("Best parameters: %s", search.best_params_)

    def cross_validate(self, cv: int = 5):
        """
        Compute R² using K-Fold cross-validation.
        """
        if self.pipeline is None:
            raise ValueError(
                "Pipeline must be built before running cross-validation."
            )

        X = self.dataframe[self.numeric_features + self.categorical_features]
        y = self.dataframe[self.target_column]

        scores = cross_val_score(
            self.pipeline,
            X,
            y,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )

        mean_score = scores.mean()
        std_score = scores.std()

        self.logger.info(
            "Cross-validation R2: mean=%.4f | std=%.4f (cv=%d)",
            mean_score,
            std_score,
            cv,
        )

        return mean_score, std_score

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on train and test sets.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not trained.")

        results: Dict[str, Dict[str, float]] = {}

        y_pred_train = self.pipeline.predict(self.X_train)
        y_pred_test = self.pipeline.predict(self.X_test)

        results["train"] = {
            "MAE": mean_absolute_error(self.y_train, y_pred_train),
            "RMSE": float(
                np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            ),
            "R2": r2_score(self.y_train, y_pred_train),
        }

        results["test"] = {
            "MAE": mean_absolute_error(self.y_test, y_pred_test),
            "RMSE": float(
                np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            ),
            "R2": r2_score(self.y_test, y_pred_test),
        }

        cv_mean, cv_std = self.cross_validate(cv=5)
        results["cv"] = {
            "R2_mean": cv_mean,
            "R2_std": cv_std,
        }

        self.logger.info(
            "Evaluation done. Train R2=%.4f | Test R2=%.4f | CV R2=%.4f",
            results["train"]["R2"],
            results["test"]["R2"],
            results["cv"]["R2_mean"],
        )

        return results

    def plot_residuals(self, return_fig: bool = False):
        """
        Plot residuals vs predictions for the test set.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not trained.")

        y_pred = self.pipeline.predict(self.X_test)
        residuals = self.y_test - y_pred

        df_plot = pd.DataFrame(
            {"y_pred": y_pred, "residuals": residuals}
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df_plot,
            x="y_pred",
            y="residuals",
            alpha=0.4,
            ax=ax,
        )
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predictions (log(Price))")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Predictions (Test set)")
        fig.tight_layout()

        if return_fig:
            return fig, ax

        plt.show()

    def store_results(self) -> None:
        """
        Store evaluation results in the class history.
        """
        scores = self.evaluate()

        row = {
            "Model": self.model_name,
            "Target": self.target_column,
            "MAE_train": scores["train"]["MAE"],
            "RMSE_train": scores["train"]["RMSE"],
            "R2_train": scores["train"]["R2"],
            "MAE_test": scores["test"]["MAE"],
            "RMSE_test": scores["test"]["RMSE"],
            "R2_test": scores["test"]["R2"],
            "R2_CV_mean": scores["cv"]["R2_mean"],
            "R2_CV_std": scores["cv"]["R2_std"],
        }

        RentalRegression.results_history.append(row)
        self.logger.info("Results stored in class history.")

    def compute_permutation_importance(self) -> None:
        """
        Compute permutation feature importance (Random Forest only).
        """
        if not self.is_trained:
            return

        if self.model_type.lower() != "random forest":
            return

        X = self.X_test
        y = self.y_test

        r = permutation_importance(
            self.pipeline,
            X,
            y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        feature_names = (
            self.numeric_features + self.categorical_features
        )

        if len(feature_names) != len(r.importances_mean):
            raise ValueError(
                f"Permutation importance mismatch for model {self.model_name}: "
                f"{len(feature_names)} features vs "
                f"{len(r.importances_mean)} importances"
            )

        for i, fname in enumerate(feature_names):
            RentalRegression.features_importance_history.append(
                {
                    "Model": self.model_name,
                    "Feature": fname,
                    "Importance_mean": r.importances_mean[i],
                    "Importance_std": r.importances_std[i],
                }
            )

    def predict_single(
        self,
        observation: dict | pd.Series | pd.DataFrame,
    ) -> float:
        """
        Predict the target value for a single observation.
        """
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before prediction."
            )

        if isinstance(observation, dict):
            obs_df = pd.DataFrame([observation])
        elif isinstance(observation, pd.Series):
            obs_df = pd.DataFrame([observation.to_dict()])
        elif isinstance(observation, pd.DataFrame):
            obs_df = observation
        else:
            raise TypeError(
                "Observation must be dict, Series, or DataFrame."
            )

        obs_df = obs_df[
            self.numeric_features + self.categorical_features
        ]

        prediction = self.pipeline.predict(obs_df)[0]
        return float(prediction)

    def run_model(self, tune: bool = False) -> None:
        """
        Run the full modeling pipeline.
        """
        self.logger.info(
            "Running full modeling pipeline (tune=%s).",
            tune,
        )

        self.set_features()
        self.train_test_split_data()
        self.build_preprocessor()
        self.build_model()
        self.build_pipeline()

        if tune and self.model_type == "random forest":
            self.tune_hyperparameters()
        else:
            self.fit()

        self.plot_residuals()
        self.store_results()
        self.compute_permutation_importance()

        self.logger.info("Full modeling pipeline finished.")

    def run_shap(
        self,
        n_samples: int = 200,
        top_k_dept: int = 8,
    ):
        """
        Compute SHAP values for the Random Forest model.
        """
        if self.model_type != "random forest":
            raise ValueError("SHAP requires Random Forest.")

        if self.pipeline is None:
            raise ValueError(
                "Pipeline must be trained before SHAP."
            )

        preprocessor = self.pipeline.named_steps["preprocessing"]
        model = self.pipeline.named_steps["model"]

        encoded_cat = (
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(self.categorical_features)
        )
        feature_names = (
            list(self.numeric_features) + list(encoded_cat)
        )

        X_train_pre = preprocessor.transform(self.X_train)

        if X_train_pre.shape[0] > n_samples:
            idx = np.random.choice(
                X_train_pre.shape[0],
                n_samples,
                replace=False,
            )
            X_train_pre = X_train_pre[idx]

        explainer = shap.Explainer(model, algorithm="tree")
        shap_values_full = explainer(
            X_train_pre,
            check_additivity=False,
        )

        shap_df = pd.DataFrame(
            shap_values_full.values,
            columns=feature_names,
        )
        X_df = pd.DataFrame(
            X_train_pre,
            columns=feature_names,
        )

        keep = set(self.numeric_features)

        prop_cols = [
            c for c in feature_names
            if "Property Type_" in c
        ]
        keep.update(prop_cols)

        dept_cols = [
            c for c in feature_names
            if "Dept_" in c
        ]

        dept_importance = (
            shap_df[dept_cols]
            .abs()
            .mean()
            .sort_values(ascending=False)
        )

        top_depts = (
            dept_importance
            .head(top_k_dept)
            .index
            .tolist()
        )
        keep.update(top_depts)

        final_cols = [
            c for c in feature_names if c in keep
        ]

        shap_small = shap_df[final_cols].values
        X_small = X_df[final_cols]

        plt.ioff()
        fig1 = plt.figure()
        shap.summary_plot(
            shap_small,
            X_small,
            feature_names=final_cols,
            show=False,
        )

        fig2 = plt.figure()
        shap.summary_plot(
            shap_small,
            X_small,
            feature_names=final_cols,
            plot_type="bar",
            show=False,
        )

        return fig1, fig2
