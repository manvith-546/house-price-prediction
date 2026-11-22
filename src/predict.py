"""
predict.py
----------
Use the trained model to predict the price of a house.

As a command-line tool:
    python src/predict.py --location "Whitefield" --sqft 1200 --bhk 2 --bath 2
    python src/predict.py --location "Indira Nagar" --sqft 1800 --bhk 3 --bath 3 \
                          --balcony 2 --area-type "Built-up Area" --under-construction
    python src/predict.py --list-locations

From Python:
    from predict import predict_price, price_curve
    predict_price(location="Whitefield", total_sqft=1200, bhk=2, bath=2)
    price_curve(location="Whitefield", total_sqft=1200, bhk=2, bath=2)   # price at other sizes
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"

LIMITS = {  # sensible input ranges (keep in sync with MIN_SQFT/MAX_* in data_prep.py)
    "total_sqft": (300, 30000),
    "bhk": (1, 10),
    "bath": (1, 12),
    "balcony": (0, 5),
}


@lru_cache(maxsize=1)
def load_model():
    """Load the saved pipeline and its info (cached after the first call)."""
    if not MODEL_PATH.exists() or not INFO_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Train it first with:  python src/train.py"
        )
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # e.g. saved with a different scikit-learn version
        raise FileNotFoundError(
            f"The saved model could not be loaded ({type(exc).__name__}). "
            "This usually means a different scikit-learn version. "
            "Re-train it with:  python src/train.py"
        ) from exc
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    return model, info


def format_price(lakh: float, symbol: str = "₹") -> str:
    """85.4 -> '₹85.40 Lakh',  152.3 -> '₹1.52 Crore'."""
    if lakh >= 100:
        return f"{symbol}{lakh / 100:.2f} Crore"
    return f"{symbol}{lakh:.2f} Lakh"


def validate_inputs(total_sqft, bhk, bath, balcony):
    """Return a list of error messages (empty list = inputs are fine)."""
    errors = []
    values = {"total_sqft": total_sqft, "bhk": bhk, "bath": bath, "balcony": balcony}
    names = {"total_sqft": "Area (sq.ft)", "bhk": "Bedrooms (BHK)",
             "bath": "Bathrooms", "balcony": "Balconies"}
    for key, value in values.items():
        low, high = LIMITS[key]
        if value is None:
            errors.append(f"{names[key]} is required.")
        elif not (low <= value <= high):
            errors.append(f"{names[key]} must be between {low:,} and {high:,}.")
    if not errors and total_sqft / bhk < 250:
        errors.append(f"{total_sqft:,.0f} sq.ft is too small for {bhk} bedrooms "
                      f"(less than 250 sq.ft per bedroom).")
    return errors


def _build_rows(info, location, sqft_values, bhk, bath, balcony, area_type, ready_to_move):
    """Make the model's input table for one house (or one house at several sizes)."""
    known_location = location in info["locations"]
    rows = pd.DataFrame({
        "location": location if known_location else "other",
        "area_type": " ".join(str(area_type).split()),
        "total_sqft": np.asarray(sqft_values, dtype=float),
        "bhk": int(bhk),
        "bath": float(bath),
        "balcony": float(balcony),
        "ready_to_move": int(bool(ready_to_move)),
    })[info["features"]]
    return rows, known_location


def _check(info, total_sqft, bhk, bath, balcony, area_type):
    errors = validate_inputs(total_sqft, bhk, bath, balcony)
    if " ".join(str(area_type).split()) not in info["area_types"]:
        errors.append(f"Area type must be one of: {', '.join(info['area_types'])}.")
    if errors:
        raise ValueError(" ".join(errors))


def predict_price(location: str, total_sqft: float, bhk: int, bath: int,
                  balcony: int = 1, area_type: str = "Super built-up Area",
                  ready_to_move: bool = True) -> dict:
    """Predict the price of one house. Returns a dictionary with the result."""
    model, info = load_model()
    _check(info, total_sqft, bhk, bath, balcony, area_type)

    row, known_location = _build_rows(info, location, [total_sqft], bhk, bath, balcony,
                                      area_type, ready_to_move)
    price = float(model.predict(row)[0])
    mape = info["metrics_test"]["MAPE_percent"] / 100  # typical % error on test data
    low, high = price * (1 - mape), price * (1 + mape)

    return {
        "price_lakh": round(price, 2),
        "price_text": format_price(price),
        "range_low_lakh": round(low, 2),
        "range_high_lakh": round(high, 2),
        "range_low_text": format_price(low),
        "range_high_text": format_price(high),
        "range_text": f"{format_price(low)} - {format_price(high)}",
        "price_per_sqft": round(price * 1e5 / total_sqft),
        "location_used": row.loc[0, "location"],
        "known_location": known_location,
        "model_name": info["model_name"],
    }


def price_curve(location: str, total_sqft: float, bhk: int, bath: int,
                balcony: int = 1, area_type: str = "Super built-up Area",
                ready_to_move: bool = True, points: int = 25) -> list:
    """Predicted price of the same house at smaller and bigger sizes.

    Returns a list of {"sqft": ..., "price_lakh": ...} sorted by size.
    """
    model, info = load_model()
    _check(info, total_sqft, bhk, bath, balcony, area_type)
    low = max(LIMITS["total_sqft"][0], 250 * bhk, total_sqft * 0.5)
    high = min(LIMITS["total_sqft"][1], max(total_sqft * 1.8, low * 1.5))
    sizes = np.unique(np.round(np.append(np.linspace(low, high, points), total_sqft)))
    rows, _ = _build_rows(info, location, sizes, bhk, bath, balcony, area_type, ready_to_move)
    prices = model.predict(rows)
    return [{"sqft": float(s), "price_lakh": round(float(p), 2)} for s, p in zip(sizes, prices)]


def main():
    parser = argparse.ArgumentParser(description="Predict a Bengaluru house price.")
    parser.add_argument("--location", help='e.g. "Whitefield"')
    parser.add_argument("--sqft", type=float, help="total area in square feet")
    parser.add_argument("--bhk", type=int, help="number of bedrooms")
    parser.add_argument("--bath", type=int, help="number of bathrooms")
    parser.add_argument("--balcony", type=int, default=1, help="number of balconies (default 1)")
    parser.add_argument("--area-type", default="Super built-up Area",
                        help='"Super built-up Area", "Built-up Area", "Plot Area" or "Carpet Area"')
    parser.add_argument("--under-construction", action="store_true",
                        help="use if the house is NOT ready to move in")
    parser.add_argument("--list-locations", action="store_true", help="show known locations")
    args = parser.parse_args()

    try:
        _, info = load_model()
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    if args.list_locations:
        print("\n".join(info["locations"]))
        return

    missing = [n for n in ("location", "sqft", "bhk", "bath") if getattr(args, n) is None]
    if missing:
        parser.error("missing: " + ", ".join("--" + m for m in missing))

    try:
        result = predict_price(args.location, args.sqft, args.bhk, args.bath, args.balcony,
                               args.area_type, ready_to_move=not args.under_construction)
    except ValueError as exc:
        sys.exit(f"Input error: {exc}")

    print("\nHouse details")
    print(f"  Location   : {args.location}"
          + ("" if result["known_location"] else "  (not in training data - using 'other')"))
    print(f"  Area       : {args.sqft:,.0f} sq.ft   |  {args.bhk} BHK  |  {args.bath} bath"
          f"  |  {args.balcony} balcony")
    print(f"  Type       : {args.area_type}, "
          + ("under construction" if args.under_construction else "ready to move"))
    print("\nPredicted price")
    print(f"  {format_price(result['price_lakh'], 'Rs. ')}"
          f"   (likely range {format_price(result['range_low_lakh'], 'Rs. ')}"
          f" - {format_price(result['range_high_lakh'], 'Rs. ')})")
    print(f"  ≈ Rs. {result['price_per_sqft']:,} per sq.ft   [model: {result['model_name']}]")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
