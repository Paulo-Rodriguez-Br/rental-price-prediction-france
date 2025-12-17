from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class RentalCleaner:
    """
    Utility class to clean and transform a rentals dataset.

    Main steps:
    - Filter rows based on valid French ZIP codes and derive departments.
    - Normalize numeric columns (Surface, Price, etc.).
    - Drop extreme or inconsistent values.
    - Create engineered features (Price per m², log transforms).
    - Optionally remove outliers based on quantiles.
    """
    
    file_path: Optional[str] = None
    
    dataframe: Optional[pd.DataFrame] = None
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize the logger, create the logs directory if needed,
        and load the input dataset from the parquet file.
        """
        self.dataframe = pd.read_parquet(self.file_path)
    
        logs_dir = Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    
        log_file = logs_dir / "cleaner.log"
    
        self.logger = logging.getLogger(__name__ + ".RentalCleaner")
        self.logger.setLevel(logging.INFO)
    
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
        self.logger.info(
            "Input dataframe loaded from '%s'. Shape: %s",
            self.file_path,
            self.dataframe.shape,
        )
    
        self.logger.info("RentalCleaner initialized.")

    def clean_city_from_address(
        self,
        address_col: str = "Address",
        keep_only_france: bool = True,
    ) -> None:
        """
        Filter rows that contain a 5-digit ZIP code in the address column
        and create the 'Dept' column from the ZIP code.

        Parameters
        ----------
        address_col : str
            Name of the column that contains the full address.
        keep_only_france : bool
            Reserved for future use (e.g. filtering by country).
        """
        if self.dataframe is None:
            self.logger.warning("clean_city_from_address called with dataframe=None.")
            return

        df = self.dataframe
        n_rows_before = len(df)

        self.logger.info(
            "Starting ZIP-code filtering from column '%s'. Rows before: %s",
            address_col,
            n_rows_before,
        )

        zipcode_mask = df[address_col].astype(str).str.contains(r"\b\d{5}\b", na=False)
        df = df.loc[zipcode_mask].copy()

        df["zipcode"] = df[address_col].astype(str).str.extract(
            r"\b(\d{5})\b", expand=False
        )

        def extract_department_code(zipcode: str | int) -> str:
            zipcode_int = int(zipcode)

            # Paulo  O2/12/2025: Depto Corse
            if 20000 <= zipcode_int <= 20199 or 20900 <= zipcode_int <= 20999:
                return "201"
            if 20200 <= zipcode_int <= 20699:
                return "202"

            zipcode_str = str(zipcode_int)

            # Paulo  O2/12/2025: DOM-TOM
            if zipcode_str[0:2] in ["97", "98"]:
                return zipcode_str[0:3]

            # Paulo  O2/12/2025: Deptos prennent les deux premières chiffres du code postal
            return zipcode_str[0:2]

        df["Dept"] = df["zipcode"].apply(extract_department_code)

        n_rows_after = len(df)
        removed_rows = n_rows_before - n_rows_after
        pct_removed = (removed_rows / n_rows_before) * 100 if n_rows_before > 0 else 0.0

        drop_columns = [address_col, "zipcode", "Durée de location"]
        self.dataframe = df.drop(columns=drop_columns, errors="ignore")

        self.logger.info(
            "ZIP-code filtering done. Rows before=%s, rows after=%s, "
            "removed=%s (%.2f%%).",
            n_rows_before,
            n_rows_after,
            removed_rows,
            pct_removed,
        )
        self.logger.info(
            "clean_city_from_address successfully finished. Final rows: %s",
            len(self.dataframe),
        )

        print("\n--------------------------------------------------")
        print("\n📦 ZIP-code filtering done!\n")
        print(f"   📊 Rows before:  {n_rows_before}")
        print(f"   📉 Rows after:   {n_rows_after}")
        print(f"   🗑️ Removed:      {removed_rows} ({pct_removed:.2f}%)")

    def normalize_number_columns(
        self,
        numeric_columns: Sequence[str] = ("Surface", "Prix", "Pièces"),
    ) -> None:
        """
        Normalize numeric columns by stripping non-digits and casting to Int64.

        Parameters
        ----------
        numeric_columns : sequence of str
            Columns to be cleaned and cast to pandas Int64 dtype.
        """
        if self.dataframe is None:
            self.logger.warning(
                "normalize_number_columns called with dataframe=None."
            )
            return

        df = self.dataframe

        self.logger.info(
            "Normalizing numeric columns: %s", ", ".join(numeric_columns)
        )

        def to_int_clean(series: pd.Series) -> pd.Series:
            return (
                series.astype(str)
                .str.replace(r"[^0-9]", "", regex=True)
                .replace("", pd.NA)
                .astype("Int64")
            )

        df[list(numeric_columns)] = df[list(numeric_columns)].apply(to_int_clean)
        self.dataframe = df

        self.logger.info("Numeric columns normalized successfully.")

    def drop_extreme_values(
        self,
        filter_small_surface: bool = True,
        filter_low_price: bool = True,
        filter_inconsistent_price_per_m2: bool = True,
        min_surface: int | float = 9,
        min_price: int | float = 300,
    ) -> None:
        """
        Drop rows with extreme or inconsistent values.

        Parameters
        ----------
        filter_small_surface : bool
            If True, filter out rows with too small or huge surfaces.
        filter_low_price : bool
            If True, filter out rows with very low prices.
        filter_inconsistent_price_per_m2 : bool
            If True, filter out rows where price is inconsistent with surface.
        min_surface : int or float
            Minimum allowed surface.
        min_price : int or float
            Minimum allowed absolute price.
        """
        if self.dataframe is None:
            self.logger.warning(
                "drop_extreme_values called with dataframe=None."
            )
            return

        df = self.dataframe
        n_rows_before = len(df)

        self.logger.info(
            "Starting extreme values filtering. Initial rows: %s", n_rows_before
        )
        print("\n--------------------------------------------------")

        removed_small_surface_rows = 0
        removed_paris_surface_outliers = 0
        removed_low_price_rows = 0
        removed_inconsistent_price_m2_rows = 0

        if filter_small_surface:
            small_surface_mask = df["Surface"] < min_surface
            removed_small_surface_rows = small_surface_mask.sum()
            df = df.loc[~small_surface_mask].copy()
            pct_small_surface = (
                removed_small_surface_rows / n_rows_before * 100
                if n_rows_before > 0
                else 0.0
            )

            self.logger.info(
                "Removed %s rows due to Surface < %s (%.2f%%).",
                removed_small_surface_rows,
                min_surface,
                pct_small_surface,
            )
            print(
                f"\n🧹 Removed due to Surface < {min_surface}: "
                f"{removed_small_surface_rows} rows ({pct_small_surface:.2f}%)"
            )

            # Paulo  O2/12/2025: Surfaces absurdes à Paris
            paris_extreme_surface_mask = (df["Dept"] == 75) & (df["Surface"] > 3500)
            removed_paris_surface_outliers = paris_extreme_surface_mask.sum()
            df = df.loc[~paris_extreme_surface_mask].copy()

            if removed_paris_surface_outliers > 0:
                pct_paris_outliers = (
                    removed_paris_surface_outliers / n_rows_before * 100
                    if n_rows_before > 0
                    else 0.0
                )
                self.logger.info(
                    "Removed %s rows as Paris surface outliers "
                    "(Surface > 3500) (%.4f%%).",
                    removed_paris_surface_outliers,
                    pct_paris_outliers,
                )
                print(
                    "🏙️ Removed Paris outliers (Surface > 3500): "
                    f"{removed_paris_surface_outliers} rows ({pct_paris_outliers:.4f}%)"
                )

        n_rows_after_surface_filter = len(df)

        if filter_low_price:
            low_price_mask = df["Prix"] < min_price
            removed_low_price_rows = low_price_mask.sum()
            df = df.loc[~low_price_mask].copy()
            pct_low_price = (
                removed_low_price_rows / n_rows_after_surface_filter * 100
                if n_rows_after_surface_filter > 0
                else 0.0
            )

            self.logger.info(
                "Removed %s rows due to Prix < %s (%.2f%%).",
                removed_low_price_rows,
                min_price,
                pct_low_price,
            )
            print(
                f"💸 Removed due to Prix < {min_price}: "
                f"{removed_low_price_rows} rows ({pct_low_price:.2f}%)"
            )

        # Paulo  O2/12/2025: On enlève prix moins que 5 fois la surface (anormaux)
        if filter_inconsistent_price_per_m2:
            inconsistent_price_m2_mask = df["Prix"] < 5 * df["Surface"]
            removed_inconsistent_price_m2_rows = inconsistent_price_m2_mask.sum()
            df = df.loc[~inconsistent_price_m2_mask].copy()
            pct_inconsistent_price_m2 = (
                removed_inconsistent_price_m2_rows / n_rows_before * 100
                if n_rows_before > 0
                else 0.0
            )

            self.logger.info(
                "Removed %s rows due to Prix < 5 × Surface (%.2f%%).",
                removed_inconsistent_price_m2_rows,
                pct_inconsistent_price_m2,
            )
            print(
                "📉 Removed due to Prix < 5 × Surface: "
                f"{removed_inconsistent_price_m2_rows} rows ({pct_inconsistent_price_m2:.2f}%)"
            )

        total_removed_rows = (
            removed_small_surface_rows
            + removed_paris_surface_outliers
            + removed_low_price_rows
            + removed_inconsistent_price_m2_rows
        )
        pct_total_removed = (
            total_removed_rows / n_rows_before * 100 if n_rows_before > 0 else 0.0
        )

        self.dataframe = df

        self.logger.info(
            "Extreme values filtering finished. Total removed: %s (%.2f%%). "
            "Final dataset size: %s rows.",
            total_removed_rows,
            pct_total_removed,
            len(df),
        )

        print("\n--------------------------------------------------")
        print(f"\n✨ Total rows removed: {total_removed_rows} ({pct_total_removed:.2f}%)")
        print(f"📦 Final dataset size: {len(df)} rows\n")
        print("--------------------------------------------------")

    def add_calculated_columns(self) -> None:
        """
        Add engineered features:
        - log (Price)
        - Price per m²
        - log (Price per m²)
        """
        if self.dataframe is None:
            self.logger.warning(
                "add_calculated_columns called with dataframe=None."
            )
            return

        df = self.dataframe

        self.logger.info(
            "Adding calculated columns: log (Price), Price per m², "
            "log (Price per m²)."
        )

        df["log (Price)"] = np.log(df["Price"])
        df["Price per m²"] = df["Price"] / df["Surface"]
        df["log (Price per m²)"] = np.log(df["Price per m²"])

        self.dataframe = df

        self.logger.info("Calculated columns added successfully.")

    def rename_columns(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Rename columns to more convenient English names.

        Parameters
        ----------
        column_mapping : dict or None
            Mapping from old to new column names. If None, a default
            mapping is applied.
        """
        if self.dataframe is None:
            self.logger.warning(
                "rename_columns called with dataframe=None."
            )
            return

        if column_mapping is None:
            column_mapping = {
                "Pièces": "Rooms",
                "Type de bien": "Property Type",
                "Prix": "Price",
            }

        self.logger.info(
            "Renaming columns using mapping: %s", column_mapping
        )

        self.dataframe = self.dataframe.rename(columns=column_mapping)
        self.logger.info("Columns renamed successfully.")

    def remove_outliers(
        self,
        target_column: str = "log (Price)",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        """
        Remove outliers from a numeric column based on quantile thresholds.

        Parameters
        ----------
        target_column : str
            Name of the numeric column on which to apply the filter.
        lower_quantile : float
            Lower quantile (between 0 and 1).
        upper_quantile : float
            Upper quantile (between 0 and 1).
        """
        if self.dataframe is None:
            self.logger.warning(
                "remove_outliers called with dataframe=None."
            )
            return

        df = self.dataframe

        if target_column not in df.columns:
            self.logger.error("Column '%s' does not exist.", target_column)
            raise ValueError(f"A coluna '{target_column}' não existe no dataframe.")

        n_rows_before = len(df)

        self.logger.info(
            "Removing outliers on column '%s' using quantiles [%.3f, %.3f]. "
            "Rows before: %s",
            target_column,
            lower_quantile,
            upper_quantile,
            n_rows_before,
        )

        q_low = df[target_column].quantile(lower_quantile)
        q_high = df[target_column].quantile(upper_quantile)

        outlier_mask = (df[target_column] >= q_low) & (df[target_column] <= q_high)
        df_without_outliers = df.loc[outlier_mask].copy()

        n_rows_after = len(df_without_outliers)
        removed_rows = n_rows_before - n_rows_after
        pct_removed = (
            removed_rows / n_rows_before * 100 if n_rows_before > 0 else 0.0
        )

        self.dataframe = df_without_outliers

        self.logger.info(
            "Outlier removal finished on '%s'. Kept range: [%.2f, %.2f]. "
            "Rows after: %s. Removed: %s (%.2f%%).",
            target_column,
            q_low,
            q_high,
            n_rows_after,
            removed_rows,
            pct_removed,
        )

        print("\n--------------------------------------------------")
        print(
            f"🔎 Remove outliers on '{target_column}' using quantiles "
            f"[{lower_quantile:.3f}, {upper_quantile:.3f}]"
        )
        print(f"   {target_column} min kept: {q_low:.2f}")
        print(f"   {target_column} max kept: {q_high:.2f}")
        print(f"   Rows before: {n_rows_before}")
        print(f"   Rows after:  {n_rows_after}")
        print(f"   Removed:     {removed_rows} ({pct_removed:.2f}%)")
        print("--------------------------------------------------")

    def run_cleaner(self) -> None:
        """
        Run the full cleaning pipeline on the current dataframe.

        Pipeline:
        - clean_city_from_address
        - normalize_number_columns
        - drop_extreme_values
        - rename_columns
        - add_calculated_columns
        - select final columns of interest
        """
        if self.dataframe is None:
            self.logger.warning("run_cleaner called with dataframe=None.")
            return
        
        print("\n🧹 Starting full cleaning pipeline... 🧹")
        self.logger.info("Starting full cleaning pipeline.")

        self.clean_city_from_address()
        self.normalize_number_columns()
        self.drop_extreme_values()
        self.rename_columns()
        self.add_calculated_columns()

        final_columns = [
            "Dept",
            "Property Type",
            "Rooms",
            "Surface",
            "Price",
            "log (Price)",
            "Price per m²",
            "log (Price per m²)",
        ]

        self.dataframe = self.dataframe[final_columns]

        self.logger.info(
            "Cleaning pipeline finished. Final dataframe shape: %s",
            self.dataframe.shape,
        )