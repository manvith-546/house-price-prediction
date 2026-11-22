"""
market.py
---------
Market statistics from the cleaned dataset, used by the web app:
  * overall numbers (listings, median price, median price per sq.ft)
  * per-locality statistics
  * median price by number of bedrooms
  * "similar houses" for a prediction
"""
from functools import lru_cache

import pandas as pd

from data_prep import CLEAN_DATA_PATH, load_clean_data


@lru_cache(maxsize=1)
def market_data() -> pd.DataFrame:
    """Cleaned dataset (created by training; rebuilt from the raw CSV if missing)."""
    if CLEAN_DATA_PATH.exists():
        return pd.read_csv(CLEAN_DATA_PATH)
    return load_clean_data(verbose=False, save=True)


def clear_cache():
    market_data.cache_clear()
    location_table.cache_clear()


MIN_LISTINGS_FOR_RANKING = 10  # a locality with fewer listings is too small to call "highest" or "lowest"


def overall_stats() -> dict:
    df = market_data()
    named = df[df["location"] != "other"]
    by_loc = named.groupby("location")["price_per_sqft"].median()
    counts = named.groupby("location")["price"].size()
    by_loc = by_loc[counts.reindex(by_loc.index) >= MIN_LISTINGS_FOR_RANKING].sort_values()
    return {
        "listings": int(len(df)),
        "median_price_lakh": float(df["price"].median()),
        "median_ppsf": float(df["price_per_sqft"].median()),
        "median_sqft": float(df["total_sqft"].median()),
        "localities": int(named["location"].nunique()),
        "priciest": by_loc.index[-1] if len(by_loc) else None,
        "priciest_ppsf": float(by_loc.iloc[-1]) if len(by_loc) else None,
        "cheapest": by_loc.index[0] if len(by_loc) else None,
        "cheapest_ppsf": float(by_loc.iloc[0]) if len(by_loc) else None,
    }


@lru_cache(maxsize=1)
def location_table() -> tuple:
    """One row per locality, most expensive (per sq.ft) first. Cached as a tuple of dicts."""
    df = market_data()
    table = (df.groupby("location")
               .agg(listings=("price", "size"),
                    median_price=("price", "median"),
                    median_ppsf=("price_per_sqft", "median"),
                    median_sqft=("total_sqft", "median"))
               .reset_index()
               .sort_values("median_ppsf", ascending=False))
    return tuple(table.to_dict("records"))


def location_ppsf(location: str):
    """Median price per sq.ft for a locality (None if unknown)."""
    for row in location_table():
        if row["location"] == location:
            return float(row["median_ppsf"])
    return None


def bhk_stats(max_bhk: int = 6) -> list:
    """Median price for 1, 2, ... max_bhk+ bedrooms."""
    df = market_data().copy()
    df["group"] = df["bhk"].clip(upper=max_bhk)
    stats = df.groupby("group").agg(listings=("price", "size"), median_price=("price", "median"))
    return [{"bhk": int(b), "label": f"{b}+" if b == max_bhk else str(b),
             "listings": int(r.listings), "median_price": float(r.median_price)}
            for b, r in stats.iterrows()]


def similar_houses(location: str, total_sqft: float, bhk: int, n: int = 5) -> list:
    """Houses in the same locality with the closest size (same BHK first)."""
    df = market_data()
    same_loc = df[df["location"] == location]
    if same_loc.empty:
        same_loc = df[df["location"] == "other"]
    candidates = same_loc.assign(
        bhk_gap=(same_loc["bhk"] - bhk).abs(),
        size_gap=(same_loc["total_sqft"] - total_sqft).abs(),
    ).sort_values(["bhk_gap", "size_gap"])
    cols = ["location", "area_type", "total_sqft", "bhk", "bath", "price", "price_per_sqft"]
    return candidates.head(n)[cols].to_dict("records")
