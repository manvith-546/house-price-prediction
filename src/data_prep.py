"""
data_prep.py
------------
Loading + cleaning + feature engineering for the Bengaluru house price data.

Works on the real Kaggle file "Bengaluru_House_Data.csv" (13,320 listings).

Cleaning steps
  1. Remove duplicate rows
  2. Drop the 'society' column (≈40% missing, too many unique values)
  3. Drop rows with missing location / size / price
  4. size ("2 BHK", "4 Bedroom", "1 RK")  ->  bhk (number of bedrooms)
  5. total_sqft ("1200", "1100 - 1300", "120Sq. Yards") -> numeric sq.ft
  6. availability ("Ready To Move", "18-Dec")  ->  ready_to_move (1/0)
  7. Tidy text columns, group rare locations (< MIN_LOCATION_COUNT rows) into "other"
  8. Keep only listings the web app accepts (300-30,000 sq.ft, up to 10
     bedrooms, up to 12 bathrooms) - this drops impossible entries
  9. Remove outliers:
       * less than 300 sq.ft per bedroom (data-entry mistakes)
       * more bathrooms than bedrooms + 2
       * price-per-sqft outside the 1.5 x IQR range of its own location
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "Bengaluru_House_Data.csv"
CLEAN_DATA_PATH = ROOT / "data" / "cleaned_house_data.csv"

NUMERIC_FEATURES = ["total_sqft", "bhk", "bath", "balcony", "ready_to_move"]
CATEGORICAL_FEATURES = ["location", "area_type"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "price"  # in lakhs

MIN_LOCATION_COUNT = 10
MIN_SQFT_PER_BHK = 300

# Only keep listings in the range the web app accepts (see predict.LIMITS).
# The real data has a few impossible entries (for example a 2 BHK of
# 1,306,800 sq.ft after converting "30Acres"), which would only confuse the model.
MIN_SQFT, MAX_SQFT = 300, 30000
MAX_BHK = 10
MAX_BATH = 12

# unit conversions to square feet for odd entries in total_sqft
UNIT_TO_SQFT = {
    "sq. meter": 10.7639,
    "sq. yards": 9.0,
    "perch": 272.25,
    "acres": 43560.0,
    "cents": 435.6,
    "guntha": 1089.0,
    "grounds": 2400.0,
}


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'Bengaluru House Price Data' from "
            "https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data "
            "and save the CSV there."
        )
    return pd.read_csv(path)


def _tidy_text(value):
    """Collapse repeated spaces: 'Super built-up  Area' -> 'Super built-up Area'."""
    if pd.isna(value):
        return np.nan
    return " ".join(str(value).split())


def size_to_bhk(value):
    """'2 BHK' -> 2, '4 Bedroom' -> 4, '1 RK' -> 1."""
    if pd.isna(value):
        return np.nan
    match = re.match(r"\s*(\d+)", str(value))
    return int(match.group(1)) if match else np.nan


def sqft_to_number(value):
    """Convert the messy total_sqft text into a float number of square feet."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    # range: "1100 - 1300" -> average
    if " - " in text:
        parts = text.split(" - ")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    # units: "120Sq. Yards", "34.46Sq. Meter", "1Acres"
    lower = text.lower()
    for unit, factor in UNIT_TO_SQFT.items():
        if lower.endswith(unit):
            try:
                return float(lower.replace(unit, "").strip()) * factor
            except ValueError:
                return np.nan
    return np.nan


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe, ready for modelling."""
    log = []

    def step(msg, frame):
        log.append((msg, len(frame)))

    df = df.copy()
    step("Raw rows", df)

    df = df.drop_duplicates()
    step("After removing duplicates", df)

    df = df.drop(columns=["society"], errors="ignore")
    df = df.dropna(subset=["location", "size", "price"])
    step("After dropping rows missing location/size/price", df)

    df["bhk"] = df["size"].apply(size_to_bhk)
    df["total_sqft"] = df["total_sqft"].apply(sqft_to_number)
    df = df.dropna(subset=["bhk", "total_sqft"])
    step("After converting size and total_sqft", df)

    in_range = (df["total_sqft"].between(MIN_SQFT, MAX_SQFT)
                & (df["bhk"] <= MAX_BHK)
                & (df["bath"].isna() | (df["bath"] <= MAX_BATH)))
    df = df[in_range]
    step(f"After keeping sizes the app accepts ({MIN_SQFT}-{MAX_SQFT:,} sqft, "
         f"<= {MAX_BHK} bedrooms)", df)

    df["ready_to_move"] = (df["availability"].astype(str).str.strip().str.lower()
                           .isin(["ready to move", "immediate possession"])).astype(int)
    df["area_type"] = df["area_type"].apply(_tidy_text)
    df["location"] = df["location"].apply(_tidy_text)

    # group rare locations so the model is not confused by 1-2 example places
    counts = df["location"].value_counts()
    rare = counts[counts < MIN_LOCATION_COUNT].index
    df["location"] = df["location"].where(~df["location"].isin(rare), "other")

    # ---- outliers ----
    df = df[df["total_sqft"] / df["bhk"] >= MIN_SQFT_PER_BHK]
    step(f"After removing < {MIN_SQFT_PER_BHK} sqft per bedroom", df)

    df = df[~(df["bath"] > df["bhk"] + 2)]
    step("After removing bath > bhk + 2", df)

    df["price_per_sqft"] = df["price"] * 1e5 / df["total_sqft"]
    q1 = df.groupby("location")["price_per_sqft"].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby("location")["price_per_sqft"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    df = df[(df["price_per_sqft"] >= q1 - 1.5 * iqr) & (df["price_per_sqft"] <= q3 + 1.5 * iqr)]
    step("After removing price-per-sqft outliers (per location, 1.5 x IQR)", df)

    df = df[FEATURES + ["price_per_sqft", TARGET]].reset_index(drop=True)
    df["bhk"] = df["bhk"].astype(int)

    if verbose:
        print("\nData cleaning summary")
        print("-" * 76)
        for msg, n in log:
            print(f"  {msg:<66} {n:>7,}")
        print(f"  {'Unique locations (incl. other)':<66} {df['location'].nunique():>7,}")
        print("-" * 76)
    return df


def load_clean_data(path=RAW_DATA_PATH, verbose: bool = True, save: bool = True) -> pd.DataFrame:
    df = clean_data(load_raw(path), verbose=verbose)
    if save:
        df.to_csv(CLEAN_DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    cleaned = load_clean_data()
    print(cleaned.head())
    print(f"\nSaved cleaned data -> {CLEAN_DATA_PATH}")
