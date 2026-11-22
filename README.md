# 🏠 Bengaluru House Price Predictor

[![CI](https://github.com/manvith-546/house-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/manvith-546/house-price-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-1d6b4d?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-regression-1d6b4d?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-web%20app-1d6b4d?logo=flask&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-e09f1f)

A machine-learning web app that estimates the price of a house in Bengaluru from its
**location, area, bedrooms, bathrooms and other features**. It is built with
Python, Pandas and Scikit-learn and has a Flask website.

![Estimate page](docs/screenshots/estimate.png)

---

## Contents
- [Features](#features)
- [Quick start in VS Code](#quick-start-in-vs-code)
- [Project structure](#project-structure)
- [How it works](#how-it-works)
- [Results](#results)
- [More screenshots](#more-screenshots)
- [JSON API](#json-api)
- [The dataset](#the-dataset)
- [Tests and CI](#tests-and-ci)
- [Run with Docker (optional)](#run-with-docker-optional)
- [Troubleshooting](#troubleshooting)
- [Ideas for improvement](#ideas-for-improvement)
- [License and credits](#license-and-credits)

---

## Features

**Machine learning**
- Cleans messy real-estate data: `"2 BHK"` → 2, `"1100 - 1300"` sq.ft → 1,200, Sq. Yards / Sq. Meter / Acres converted, duplicates and outliers removed
- Compares **6 regression models** (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting) with 5-fold cross-validation
- Tunes the best model with `GridSearchCV` and tests it on houses it has never seen
- One scikit-learn `Pipeline`: imputing, log-transform, scaling, one-hot encoding and the model are saved together
- 9 charts from training: price distribution, correlations, model comparison, actual vs predicted, feature importance and more

**Web app**
- **Estimate page:** price, likely range, price per sq.ft compared with the locality, a price-vs-size chart, similar houses from the data, and your recent estimates
- **Market page:** median prices by number of bedrooms and a searchable, sortable table of every locality
- **Model page:** accuracy, model comparison, what drives the price, training charts and API guide
- Light and dark theme, works on phones, fonts bundled (no internet needed)
- Shareable links, e.g. `/?location=Whitefield&total_sqft=1200&bhk=2&bath=2`
- JSON API for other apps

**Developer experience**
- Ready for VS Code: one-click run configurations, tasks, test explorer and recommended extensions
- `python app.py` trains the model automatically the first time
- 19 automatic tests and a GitHub Actions workflow that trains and tests on every push
- Jupyter notebook that walks through every step (useful for a demo or viva)

---

## Quick start in VS Code

**You need:** [Python 3.11 or newer](https://www.python.org/downloads/) (tick **"Add Python to PATH"** on Windows),
[VS Code](https://code.visualstudio.com/) and [Git](https://git-scm.com/).

### 1. Get the code
```bash
git clone https://github.com/manvith-546/house-price-prediction.git
cd house-price-prediction
code .
```
(Or download the ZIP from GitHub, unzip it, and use **File → Open Folder** in VS Code.)

When VS Code asks, install the **recommended extensions** (Python, Jupyter).

### 2. Create a virtual environment and install packages
Open a terminal in VS Code (**Terminal → New Terminal**):

<table>
<tr><th>Windows</th><th>Mac / Linux</th></tr>
<tr><td>

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
</td><td>

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
</td></tr>
</table>

Then press **Ctrl+Shift+P → "Python: Select Interpreter"** and pick the one in `venv`.

### 3. Run it
```bash
python app.py
```
Open **http://127.0.0.1:5000**. A trained model is included, so the app starts straight away. If it can't be loaded
(for example a different scikit-learn version), the app trains a new one first, which takes about 3-4 minutes.

### Other ways to run
| What | How |
|---|---|
| Run from VS Code | **Run and Debug** panel (Ctrl+Shift+D) → *Run web app* → ▶ |
| Live reload while editing | *Run web app (auto-reload)* configuration |
| Tasks | **Ctrl+Shift+P → Tasks: Run Task** → Setup / Train model / Run web app / Run tests |
| Tests in the sidebar | **Testing** panel (flask icon) → ▶ |
| One step, Windows | double-click `run_project.bat` |
| One step, Mac/Linux | `./run_project.sh` |

### Command line
```bash
python src/train.py                       # clean data, compare models, save the best one + charts
python src/predict.py --location "Whitefield" --sqft 1200 --bhk 2 --bath 2
python src/predict.py --location "Indira Nagar" --sqft 1800 --bhk 3 --bath 3 \
                      --balcony 2 --area-type "Built-up Area" --under-construction
python src/predict.py --list-locations
python -m unittest discover tests -v      # run the tests
```

---

## Project structure

```
house-price-prediction/
├── app.py                        # Flask web app + JSON API
├── House_Price_Prediction.ipynb  # step-by-step notebook
├── requirements.txt
├── data/
│   └── Bengaluru_House_Data.csv  # real listings from Kaggle (13,320 rows)
├── src/
│   ├── data_prep.py              # loading, cleaning, feature engineering
│   ├── train.py                  # EDA charts, model comparison, tuning, saving
│   ├── predict.py                # prediction functions + command-line tool
│   ├── market.py                 # locality statistics, similar houses
│   └── charts.py                 # SVG chart for the web page
├── models/
│   ├── house_price_model.joblib  # trained pipeline
│   └── model_info.json           # metrics, features, locations, comparison
├── reports/
│   ├── model_comparison.csv
│   └── figures/                  # 9 training charts (PNG)
├── templates/                    # HTML pages (Jinja)
├── static/                       # CSS, JavaScript, fonts, icon
├── tests/test_project.py         # 19 automatic tests
├── docs/screenshots/
├── .github/workflows/ci.yml      # GitHub Actions: train + test on every push
├── .vscode/                      # run configurations, tasks, settings
├── Dockerfile
├── run_project.bat / run_project.sh
└── LICENSE
```

---

## How it works

### 1. Data
The real **Bengaluru House Price Data** from Kaggle: 13,320 listings with 9 columns:
`area_type, availability, location, size, society, total_sqft, bath, balcony, price` (price in ₹ lakh).
After cleaning, 11,198 listings in 248 localities remain.

### 2. Cleaning (`src/data_prep.py`)
1. Remove duplicate rows
2. Drop `society` (about 40% missing)
3. Drop rows without location, size or price
4. `size` → `bhk` (number of bedrooms)
5. `total_sqft` → number: ranges averaged, other units converted
6. `availability` → `ready_to_move` (1 / 0)
7. Keep only listings the web app accepts: 300-30,000 sq.ft, up to 10 bedrooms and 12 bathrooms.
   The real data has a few impossible entries (for example a 2 BHK of 1,306,800 sq.ft after
   converting "30Acres") that would only confuse the model
8. Localities with fewer than 10 houses → `"other"`
9. Outliers removed: under 300 sq.ft per bedroom, bathrooms > bedrooms + 2,
   and price per sq.ft outside 1.5 × IQR **of its own locality**

### 3. Features and pipeline (`src/train.py`)
| Feature | Processing |
|---|---|
| `total_sqft` | median impute → **log** → standard scale |
| `bhk`, `bath`, `balcony`, `ready_to_move` | median impute → standard scale |
| `location`, `area_type` | most-frequent impute → one-hot encode |
| target `price` | model learns **log(price)**, converted back automatically |

Prices are right-skewed and grow in proportion to area, so working on a log scale turns
the relationship into a straight line.

**A bigger house is never cheaper.** Tree models can make the price dip when the area grows a
little. The Gradient Boosting model is therefore given a *monotonic constraint* on the area:
with everything else equal, more area never lowers the price. This keeps the "price at other
sizes" chart sensible and cost no accuracy.

### 4. Model selection
80% of houses are used for training and 20% are kept aside. Each model is scored with
5-fold cross-validation on the training data, the best one is tuned with `GridSearchCV`
(if it has settings to tune), and the final score comes from the untouched test set.

---

## Results

Real Kaggle data: 8,958 training houses and 2,240 test houses (after cleaning). The models are
ranked by 5-fold cross-validation on the training houses:

| Model | CV R² | Test R² | MAE (₹ lakh) | RMSE (₹ lakh) | MAPE |
|---|---|---|---|---|---|
| **Gradient Boosting** ✅ | **0.760** | 0.685 | **20.43** | 52.29 | **20.2%** |
| Random Forest | 0.695 | 0.626 | 22.30 | 56.92 | 21.7% |
| Linear Regression | 0.694 | **0.819** | 20.73 | **39.57** | 21.1% |
| Ridge Regression | 0.682 | 0.815 | 20.74 | 40.04 | 21.0% |
| Decision Tree | 0.628 | 0.597 | 25.13 | 59.11 | 24.8% |
| Lasso Regression | 0.556 | 0.714 | 23.73 | 49.82 | 23.2% |

**Final model** (Gradient Boosting after `GridSearchCV`): CV R² **0.775**; on the untouched test
houses R² **0.698**, average error **₹19.4 lakh** (MAE), RMSE ₹51.1 lakh and about **19%** off on
average (MAPE).

- **R²** – share of price differences the model explains (1.0 = perfect)
- **MAE / RMSE** – average error in ₹ lakh (RMSE punishes big misses more)
- **MAPE** – average error in percent; the web app uses it for the "likely range"

**Reading the numbers honestly**
- **Why Gradient Boosting was chosen:** it has the best cross-validation score, which averages 5
  different splits and is more reliable than one test split. It also has the lowest average error
  (MAE and MAPE).
- **Why Linear Regression has a higher test R²:** prices are heavy-tailed. A few very expensive
  homes in the test set dominate R² and RMSE, and the linear model happens to miss those by less
  on this one split. The typical error is about the same for both.
- **What to expect:** these are asking prices from listings, with no floor, building age or
  amenities, so an average error of about 19% is realistic for this dataset. Treat the estimate as
  a guide, not a valuation.
- **Most important features:** total area, then location, then bathrooms and area type.

---

## More screenshots

| Market | Model |
|---|---|
| ![Market page](docs/screenshots/market.png) | ![Model page](docs/screenshots/model.png) |

| Dark theme | Phone |
|---|---|
| ![Dark theme](docs/screenshots/estimate-dark.png) | ![Phone layout](docs/screenshots/mobile.png) |

---

## JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2, "balcony": 1, "area_type": "Super built-up Area", "ready_to_move": true}'
```
```json
{"price_lakh": 58.07, "price_text": "₹58.07 Lakh", "range_text": "₹46.98 Lakh - ₹69.17 Lakh", "price_per_sqft": 4840, "...": "..."}
```

| Endpoint | Returns |
|---|---|
| `POST /api/predict` | Estimate, likely range, price per sq.ft (only `location`, `total_sqft`, `bhk`, `bath` are required) |
| `GET /api/locations` | Listings, median price, rate and size for every locality |
| `GET /api/info` | Model name, metrics, features, accepted locations and area types |
| `GET /api/health` | `{"status": "ok"}` when the model is loaded |

---

## The dataset

`data/Bengaluru_House_Data.csv` is the public Kaggle dataset
[Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
(13,320 listings, 9 columns). The data belongs to its original author; see the Kaggle page for
its terms before reusing it elsewhere.

To use a newer or different file with the same columns, replace the CSV and run
`python src/train.py`. No code changes are needed.

---

## Tests and CI

```bash
python -m unittest discover tests -v
```
The 19 tests cover data cleaning, predictions (bigger and better-located houses must cost
more), market statistics, the chart, every page and the API.

On GitHub, `.github/workflows/ci.yml` installs the packages, trains the model and runs the
tests on Python 3.11, 3.12 and 3.13 for every push and pull request. The badge at the top
shows the result.

---

## Run with Docker (optional)

```bash
docker build -t house-price-predictor .
docker run -p 5000:5000 house-price-predictor
```

Settings can be changed with environment variables: `HOST`, `PORT`, `FLASK_DEBUG=1`, `SECRET_KEY`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick **Add Python to PATH**, or use `py` instead of `python` |
| `venv\Scripts\activate` is blocked in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt |
| `ModuleNotFoundError` | Activate the venv and run `pip install -r requirements.txt`; select the venv interpreter in VS Code |
| "The saved model could not be loaded" | Your scikit-learn version differs; run `python src/train.py` (or just `python app.py`). Training takes about 3-4 minutes |
| Port 5000 is already in use | `set PORT=5050` (Windows) or `PORT=5050 python app.py` (Mac/Linux) |
| Charts don't open a window | That's expected; they are saved in `reports/figures/` |

---

## Ideas for improvement
- More features: floor, building age, furnishing, parking, distance to metro and IT parks
- Try XGBoost / LightGBM or target encoding for locations
- A map of predicted prices by locality
- Deploy online (Render, Railway or PythonAnywhere)

---

## License and credits
- Code: [MIT License](LICENSE)
- Dataset: [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) on Kaggle
- Fonts: [Fira Sans](https://fonts.google.com/specimen/Fira+Sans) and [Bitter](https://fonts.google.com/specimen/Bitter), SIL Open Font License (`static/fonts/`)
