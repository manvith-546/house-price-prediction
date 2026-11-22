"""
train.py
--------
End-to-end training script:

  1. Load + clean the data                    (data_prep.py)
  2. Exploratory data analysis charts         (reports/figures/)
  3. Train/test split (80 / 20)
  4. Compare 6 regression models with 5-fold cross-validation
       Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting
  5. Tune the best model with GridSearchCV
  6. Evaluate on the unseen test set  (R², MAE, RMSE, MAPE)
  7. Save the final pipeline -> models/house_price_model.joblib
     and its details        -> models/model_info.json

Run:  python src/train.py
"""
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data_prep import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, RAW_DATA_PATH,
                       TARGET, load_clean_data)

warnings.filterwarnings("ignore", category=UserWarning)
try:  # make sure ₹ and ² print correctly on Windows terminals
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------- chart style ----------
# same colours as the web app: garden green + marigold
GREEN, MARIGOLD = "#1d6b4d", "#e09f1f"
INK, INK_2, GRID, SURFACE = "#18211e", "#4c5854", "#e1e6e2", "#fcfdfc"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.size": 10,
})


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# =====================================================================
# 1. Exploratory Data Analysis
# =====================================================================
def eda_charts(df: pd.DataFrame):
    print("\nCreating EDA charts ...")

    # Price distribution (raw and log)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df[TARGET], bins=60, color=GREEN, edgecolor=SURFACE, linewidth=0.6)
    axes[0].set(title="Price distribution", xlabel="Price (₹ lakh)", ylabel="Number of houses")
    axes[1].hist(np.log1p(df[TARGET]), bins=60, color=GREEN, edgecolor=SURFACE, linewidth=0.6)
    axes[1].set(title="Log(price) distribution - closer to normal",
                xlabel="log(1 + price)", ylabel="Number of houses")
    save_fig(fig, "01_price_distribution.png")

    # Price vs area
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["total_sqft"], df[TARGET], s=10, alpha=0.35, color=GREEN, linewidths=0)
    ax.set(title="Price vs total area", xlabel="Total area (sq.ft)", ylabel="Price (₹ lakh)")
    save_fig(fig, "02_price_vs_area.png")

    # Most expensive locations (median price per sqft)
    loc = (df[df["location"] != "other"].groupby("location")["price_per_sqft"]
           .median().sort_values().tail(15))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(loc.index, loc.values, color=GREEN, height=0.65)
    for y, v in enumerate(loc.values):
        ax.text(v + 150, y, f"₹{v:,.0f}", va="center", fontsize=8.5, color=INK_2)
    ax.set(title="Top 15 locations by median price per sq.ft", xlabel="₹ per sq.ft")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, loc.max() * 1.15)
    save_fig(fig, "03_top_locations_price_per_sqft.png")

    # Price by number of bedrooms
    bhk_values = sorted(df["bhk"].unique())
    data = [df.loc[df["bhk"] == b, TARGET] for b in bhk_values]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=[str(b) for b in bhk_values], patch_artist=True,
                    showfliers=False, widths=0.55)
    for box in bp["boxes"]:
        box.set(facecolor="#d5e8dd", edgecolor=GREEN)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color(GREEN)
    for med in bp["medians"]:
        med.set(color=INK, linewidth=2)
    ax.set(title="Price by number of bedrooms (BHK)", xlabel="BHK", ylabel="Price (₹ lakh)")
    save_fig(fig, "04_price_by_bhk.png")

    # Correlation heatmap
    cols = ["total_sqft", "bhk", "bath", "balcony", "ready_to_move", "price"]
    corr = df[cols].corr()
    cmap = LinearSegmentedColormap.from_list("div", [MARIGOLD, "#eef0ee", GREEN])
    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation between numeric features")
    save_fig(fig, "05_correlation_heatmap.png")
    print(f"  EDA charts saved in {FIG_DIR.relative_to(ROOT)}/")


# =====================================================================
# 2. Model pipeline
# =====================================================================
def build_preprocessor():
    # Area is used on a log scale: price grows with area *proportionally*,
    # and the model predicts log(price), so log(area) makes the relation linear.
    area = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    other_numeric = [c for c in NUMERIC_FEATURES if c != "total_sqft"]
    return ColumnTransformer([
        ("area", area, ["total_sqft"]),
        ("num", numeric, other_numeric),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_pipeline(regressor):
    # Prices are right-skewed, so the model learns log(price) and converts back.
    model = TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.0005, max_iter=20000),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                           n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                                   max_depth=4, random_state=RANDOM_STATE),
}

PARAM_GRIDS = {
    "Linear Regression": {},
    "Ridge Regression": {"model__regressor__alpha": [0.01, 0.1, 1.0, 10.0]},
    "Lasso Regression": {"model__regressor__alpha": [0.0001, 0.0005, 0.001, 0.01]},
    "Decision Tree": {"model__regressor__max_depth": [8, 12, 16, None],
                      "model__regressor__min_samples_leaf": [2, 5, 10]},
    "Random Forest": {"model__regressor__n_estimators": [200, 400],
                      "model__regressor__max_features": [0.5, 1.0],
                      "model__regressor__min_samples_leaf": [1, 3]},
    "Gradient Boosting": {"model__regressor__n_estimators": [300, 600],
                          "model__regressor__learning_rate": [0.05, 0.1],
                          "model__regressor__max_depth": [3, 4]},
}


def evaluate(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE_lakh": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_lakh": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_percent": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


# =====================================================================
# 3. Main
# =====================================================================
def main():
    start = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(" HOUSE PRICE PREDICTION - TRAINING")
    print("=" * 76)
    print(f"Dataset: {RAW_DATA_PATH.relative_to(ROOT)}")

    df = load_clean_data(RAW_DATA_PATH)
    eda_charts(df)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"\nTrain rows: {len(X_train):,}   Test rows: {len(X_test):,}")

    # ---------- compare models ----------
    print(f"\nComparing models ({CV_FOLDS}-fold cross-validation on training data)")
    print("-" * 76)
    print(f"  {'Model':<20}{'CV R²':>10}{'Test R²':>10}{'MAE (lakh)':>13}{'RMSE (lakh)':>13}{'MAPE %':>9}")
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, reg in MODELS.items():
        pipe = make_pipeline(reg)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring="r2", n_jobs=-1)
        pipe.fit(X_train, y_train)
        m = evaluate(y_test, pipe.predict(X_test))
        rows.append({"Model": name, "CV_R2_mean": cv_scores.mean(), "CV_R2_std": cv_scores.std(),
                     "Test_R2": m["R2"], "Test_MAE_lakh": m["MAE_lakh"],
                     "Test_RMSE_lakh": m["RMSE_lakh"], "Test_MAPE_percent": m["MAPE_percent"]})
        print(f"  {name:<20}{cv_scores.mean():>10.4f}{m['R2']:>10.4f}"
              f"{m['MAE_lakh']:>13.2f}{m['RMSE_lakh']:>13.2f}{m['MAPE_percent']:>9.2f}")
    comparison = pd.DataFrame(rows).sort_values("CV_R2_mean", ascending=False)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # chart: model comparison
    comp = comparison.sort_values("Test_R2")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [GREEN if n == comparison.iloc[0]["Model"] else "#a8cdb9" for n in comp["Model"]]
    ax.barh(comp["Model"], comp["Test_R2"], color=colors, height=0.6)
    for yy, v in enumerate(comp["Test_R2"]):
        ax.text(v + 0.005, yy, f"{v:.3f}", va="center", fontsize=9, color=INK_2)
    ax.set(title="Model comparison - R² on test data (higher is better)", xlabel="R² score",
           xlim=(max(0, comp["Test_R2"].min() - 0.1), 1.0))
    ax.grid(axis="y", visible=False)
    save_fig(fig, "06_model_comparison.png")

    # ---------- tune the best model ----------
    best_name = comparison.iloc[0]["Model"]
    print("-" * 76)
    print(f"Best model by cross-validation: {best_name}")
    grid = PARAM_GRIDS.get(best_name, {})
    best_pipe = make_pipeline(MODELS[best_name])
    best_params = {}
    if grid:
        print(f"Tuning hyper-parameters with GridSearchCV ({CV_FOLDS}-fold) ...")
        search = GridSearchCV(best_pipe, grid, cv=kfold, scoring="r2", n_jobs=-1)
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        best_params = {k.replace("model__regressor__", ""): v for k, v in search.best_params_.items()}
        print(f"  Best parameters: {best_params}")
        print(f"  Tuned CV R²: {search.best_score_:.4f}")
    else:
        best_pipe.fit(X_train, y_train)

    # ---------- final evaluation ----------
    y_pred = best_pipe.predict(X_test)
    final = evaluate(y_test, y_pred)
    print("\nFinal model on unseen TEST data")
    print("-" * 76)
    print(f"  R² score : {final['R2']:.4f}  (the model explains {final['R2']*100:.1f}% of price variation)")
    print(f"  MAE      : ₹{final['MAE_lakh']:.2f} lakh  (average error)")
    print(f"  RMSE     : ₹{final['RMSE_lakh']:.2f} lakh")
    print(f"  MAPE     : {final['MAPE_percent']:.2f}%  (average % error)")

    # chart: actual vs predicted
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, y_pred, s=12, alpha=0.4, color=GREEN, linewidths=0, label="Test houses")
    lim = [0, max(y_test.max(), y_pred.max()) * 1.03]
    ax.plot(lim, lim, color=MARIGOLD, linewidth=2, label="Perfect prediction")
    ax.set(xlim=lim, ylim=lim, title=f"Actual vs predicted price - {best_name}",
           xlabel="Actual price (₹ lakh)", ylabel="Predicted price (₹ lakh)")
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, "07_actual_vs_predicted.png")

    # chart: residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(residuals, bins=60, color=GREEN, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(0, color=MARIGOLD, linewidth=2)
    ax.set(title="Prediction errors (actual - predicted)", xlabel="Error (₹ lakh)",
           ylabel="Number of houses")
    save_fig(fig, "08_residuals.png")

    # chart: which features matter most (permutation importance)
    print("\nCalculating feature importance ...")
    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=5, scoring="r2",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(importance.index, importance.values, color=GREEN, height=0.6)
    for yy, v in enumerate(importance.values):
        ax.text(max(v, 0) + 0.005, yy, f"{max(v, 0):.3f}", va="center", fontsize=9, color=INK_2)
    ax.set(title="Feature importance (drop in R² when a feature is shuffled)",
           xlabel="Importance", xlim=(0, importance.max() * 1.15))
    ax.grid(axis="y", visible=False)
    save_fig(fig, "09_feature_importance.png")
    for feat, val in importance.sort_values(ascending=False).items():
        print(f"  {feat:<15} {val:.4f}")

    # ---------- save ----------
    joblib.dump(best_pipe, MODEL_PATH)
    locations = sorted(l for l in df["location"].unique() if l != "other") + ["other"]
    info = {
        "model_name": best_name,
        "best_params": best_params,
        "metrics_test": final,
        "features": FEATURES,
        "target": "price (₹ lakh)",
        "locations": locations,
        "area_types": sorted(df["area_type"].unique().tolist()),
        "cv_r2": float(comparison.iloc[0]["CV_R2_mean"]),
        "feature_importance": {k: round(float(v), 4) for k, v in
                               importance.sort_values(ascending=False).items()},
        "comparison": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                       for row in comparison.to_dict("records")],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset": RAW_DATA_PATH.name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,
    }
    INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved")
    print("-" * 76)
    print(f"  Model        -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"  Model info   -> {INFO_PATH.relative_to(ROOT)}")
    print(f"  Comparison   -> {(REPORT_DIR / 'model_comparison.csv').relative_to(ROOT)}")
    print(f"  Charts       -> {FIG_DIR.relative_to(ROOT)}/ (9 PNG files)")
    print(f"\nDone in {time.time() - start:.0f} s. Next: python app.py")


if __name__ == "__main__":
    plt.switch_backend("Agg")  # save charts to files, no pop-up windows
    main()
