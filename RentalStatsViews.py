from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class RentalStatsViews:
    """
    Visualization helper class for rental statistics.
    Compatible with Streamlit and Jupyter/Spyder.
    """

    dataframe: Optional[pd.DataFrame] = None
    filtered_dataframe: Optional[pd.DataFrame] = None
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / "stats_views.log"

        self.logger = logging.getLogger(__name__ + ".RentalStatsViews")
        self.logger.setLevel(logging.INFO)

        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.logger.info("RentalStatsViews initialized.")

    def compare_hist_price_vs_log(
        self,
        price_per_m2_col: str = "Price per m²",
        log_price_per_m2_col: str = "log (Price per m²)",
        bins: int = 100,
    ):
        if self.dataframe is None:
            return None

        df = self.dataframe

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(df[price_per_m2_col].dropna(), bins=bins, ax=axes[0])
        axes[0].set_title(price_per_m2_col)

        sns.histplot(df[log_price_per_m2_col].dropna(), bins=bins, ax=axes[1])
        axes[1].set_title(log_price_per_m2_col)

        plt.tight_layout()
        return fig

    def sns_scatter(self, x_col, y_col, title=None, alpha: float = 0.5, size: int = 40):

        if self.dataframe is None:
            return None

        df = self.dataframe

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            s=size,
            alpha=alpha,
            ax=ax,
        )

        ax.set_title(title or f"{x_col} vs {y_col}")
        plt.tight_layout()
        return fig

    def plot_multiple_boxplots(
        self,
        y_value_col,
        x_category_col,
        title=None,
        limit_top_categories: bool = False,
        top_n: int = 10,
    ):

        if self.dataframe is None:
            return None

        df = self.dataframe

        if limit_top_categories:
            medians = (
                df.groupby(x_category_col)[y_value_col]
                .median()
                .sort_values(ascending=False)
                .head(top_n)
            )
            df = df[df[x_category_col].isin(medians.index)]

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(data=df, x=x_category_col, y=y_value_col, ax=ax)

        ax.set_title(title or f"{y_value_col} by {x_category_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plot_bar_by_category(
        self,
        category_column: str = "Dept",
        value_column: str = "Price per m²",
        aggregation_method: str = "median",
        filter_top_categories: bool = True,
        top_n: int = 15,
        title: Optional[str] = None,
    ):

        if self.dataframe is None:
            return None

        df = self.dataframe

        grouped = df.groupby(category_column)[value_column]

        if aggregation_method == "median":
            aggregated = grouped.median()
        elif aggregation_method == "mean":
            aggregated = grouped.mean()
        elif aggregation_method == "sum":
            aggregated = grouped.sum()
        else:
            raise ValueError("aggregation_method must be 'median', 'mean' or 'sum'.")

        aggregated = aggregated.sort_values(ascending=False)

        if filter_top_categories:
            aggregated = aggregated.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            x=aggregated.index.astype(str),
            y=aggregated.values,
            ax=ax,
        )

        ax.set_title(title or f"{aggregation_method} of {value_column} by {category_column}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def execute_views(self, filter_by_dept: bool = False, dept_code=None) -> List:

        if self.dataframe is None:
            return []

        df = self.dataframe

        if filter_by_dept and dept_code:
            df = df.loc[df["Dept"] == dept_code]

        self.dataframe = df

        figs = []

        figs.append(self.compare_hist_price_vs_log())
        figs.append(self.sns_scatter(x_col="Surface", y_col="log (Price)"))

        figs.append(self.plot_multiple_boxplots(
            y_value_col="log (Price per m²)",
            x_category_col="Property Type",
        ))

        figs.append(self.plot_multiple_boxplots(
            y_value_col="log (Price per m²)",
            x_category_col="Rooms",
        ))

        figs.append(self.plot_multiple_boxplots(
            y_value_col="log (Price per m²)",
            x_category_col="Dept",
            limit_top_categories=True,
            top_n=10,
        ))

        figs.append(self.plot_bar_by_category(filter_top_categories=True, top_n=20))

        return [fig for fig in figs if fig is not None]
