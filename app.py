"""
app.py - Flask web app for the Bengaluru House Price Predictor.

Run:   python app.py          (trains the model automatically the first time)
Open:  http://127.0.0.1:5000

Pages
    /          Estimate a price (form, price range, price-vs-size chart, similar houses)
    /market    Prices by locality and by number of bedrooms
    /model     Model accuracy, what drives the price, API guide

JSON API
    POST /api/predict     {"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2}
    GET  /api/locations   statistics for every locality
    GET  /api/info        model details
    GET  /api/health      quick status check

Settings (environment variables, all optional)
    HOST (default 127.0.0.1), PORT (default 5000), FLASK_DEBUG (1 = auto-reload), SECRET_KEY
"""
import os
import sys
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import market  # noqa: E402
from charts import price_curve_svg  # noqa: E402
from predict import format_price, load_model, predict_price, price_curve  # noqa: E402

FIGURES_DIR = ROOT / "reports" / "figures"
HISTORY_SIZE = 6

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(24)
app.json.sort_keys = False       # keep API fields in a readable order
app.json.ensure_ascii = False    # show ₹ as ₹ in API responses

DEFAULTS = {"location": "Whitefield", "total_sqft": "1200", "bhk": "2", "bath": "2",
            "balcony": "1", "area_type": "Super built-up Area", "ready_to_move": "1"}

FEATURE_NAMES = {"total_sqft": "Total area", "location": "Location", "area_type": "Area type",
                 "bath": "Bathrooms", "bhk": "Bedrooms", "ready_to_move": "Ready to move",
                 "balcony": "Balconies"}

AREA_TYPE_HELP = {
    "Super built-up Area": "Flat area plus a share of common areas (lobby, lifts, stairs)",
    "Built-up Area": "Flat area including walls and balconies",
    "Carpet Area": "Usable floor area inside the walls",
    "Plot Area": "Land area of an independent house or villa plot",
}


# ---------------------------------------------------------------- helpers
@app.template_filter("money")
def money_filter(lakh):
    return format_price(float(lakh))


@app.template_filter("inr")
def inr_filter(value):
    return f"₹{float(value):,.0f}"


@app.template_filter("num")
def num_filter(value, digits=0):
    return f"{float(value):,.{digits}f}"


def _to_number(value, cast=float):
    try:
        return cast(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _read_inputs(source):
    return {
        "location": str(source.get("location", "")).strip(),
        "total_sqft": _to_number(source.get("total_sqft"), float),
        "bhk": _to_number(source.get("bhk"), int),
        "bath": _to_number(source.get("bath"), int),
        "balcony": _to_number(source.get("balcony", 1), int),
        "area_type": str(source.get("area_type", "Super built-up Area")).strip(),
        "ready_to_move": str(source.get("ready_to_move", "1")).lower() in ("1", "true", "yes", "on"),
    }


def _model_or_none():
    try:
        return load_model()[1], None
    except FileNotFoundError as exc:
        return None, str(exc)


def _remember(form, result):
    entry = {"location": form["location"], "total_sqft": form["total_sqft"], "bhk": form["bhk"],
             "bath": form["bath"], "balcony": form["balcony"], "area_type": form["area_type"],
             "ready_to_move": form["ready_to_move"], "price_text": result["price_text"]}
    history = [h for h in session.get("history", []) if h != entry]
    session["history"] = [entry] + history[:HISTORY_SIZE - 1]


@app.context_processor
def inject_globals():
    return {"nav": request.endpoint}


# ---------------------------------------------------------------- pages
@app.route("/", methods=["GET", "POST"])
def index():
    info, setup_error = _model_or_none()
    if setup_error:
        return render_template("setup.html", setup_error=setup_error), 503

    form, result, error = dict(DEFAULTS), None, None
    source = request.form if request.method == "POST" else request.args
    wants_estimate = request.method == "POST" or ("location" in request.args
                                                  and "total_sqft" in request.args)
    if wants_estimate:
        form.update({k: str(source.get(k, v)).strip() for k, v in DEFAULTS.items()})
        inputs = _read_inputs(source)
        try:
            result = predict_price(**inputs)
        except ValueError as exc:
            error = str(exc)
    elif request.args.get("location"):  # link from the market page: just pre-select the location
        form["location"] = request.args["location"]

    extra = {}
    if result:
        curve = price_curve(**inputs)
        typical = market.location_ppsf(result["location_used"])
        extra = {
            "chart": price_curve_svg(curve, float(inputs["total_sqft"]), result["price_lakh"]),
            "similar": market.similar_houses(result["location_used"], inputs["total_sqft"], inputs["bhk"]),
            "typical_ppsf": typical,
            "vs_typical": (result["price_per_sqft"] / typical - 1) * 100 if typical else None,
        }
        _remember(form, result)

    area_types = ([t for t in AREA_TYPE_HELP if t in info["area_types"]]
                  + [t for t in info["area_types"] if t not in AREA_TYPE_HELP])
    return render_template("predict.html", info=info, form=form, result=result, error=error,
                           area_types=area_types,
                           history=session.get("history", []), area_help=AREA_TYPE_HELP,
                           listings=info["train_rows"] + info["test_rows"], **extra)


@app.post("/history/clear")
def clear_history():
    session.pop("history", None)
    return redirect(url_for("index"))


@app.get("/market")
def market_page():
    info, setup_error = _model_or_none()
    if setup_error:
        return render_template("setup.html", setup_error=setup_error), 503
    table = market.location_table()
    return render_template("market.html", stats=market.overall_stats(), table=table,
                           max_ppsf=max(r["median_ppsf"] for r in table),
                           bhk=market.bhk_stats())


@app.get("/model")
def model_page():
    info, setup_error = _model_or_none()
    if setup_error:
        return render_template("setup.html", setup_error=setup_error), 503
    figures = sorted(p.name for p in FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    importance = info.get("feature_importance", {})
    return render_template("model.html", info=info, figures=figures,
                           importance=[(FEATURE_NAMES.get(k, k), max(v, 0)) for k, v in importance.items()],
                           max_importance=max([v for v in importance.values()] + [1e-9]),
                           host_url=request.host_url.rstrip("/"))


@app.get("/figures/<path:name>")
def figure(name):
    if not name.endswith(".png"):
        abort(404)
    return send_from_directory(FIGURES_DIR, name)


@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404


# ---------------------------------------------------------------- JSON API
@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    inputs = _read_inputs(data)
    if not inputs["location"]:
        return jsonify({"error": "location is required"}), 400
    try:
        return jsonify(predict_price(**inputs))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/api/locations")
def api_locations():
    return jsonify([{**r, "median_price": round(r["median_price"], 2),
                     "median_ppsf": round(r["median_ppsf"]),
                     "median_sqft": round(r["median_sqft"])} for r in market.location_table()])


@app.get("/api/info")
def api_info():
    info, setup_error = _model_or_none()
    if setup_error:
        return jsonify({"error": setup_error}), 503
    return jsonify(info)


@app.get("/api/health")
def api_health():
    info, setup_error = _model_or_none()
    return jsonify({"status": "ok" if info else "model missing",
                    "model": info["model_name"] if info else None}), (200 if info else 503)


# ---------------------------------------------------------------- start-up
def ensure_model():
    """Train the model automatically if it is missing or cannot be loaded."""
    try:
        load_model()
        return
    except FileNotFoundError as exc:
        print(f"\n{exc}\nTraining the model now (takes about a minute)...\n")
    import matplotlib.pyplot as plt
    import train
    plt.switch_backend("Agg")
    train.main()
    load_model.cache_clear()
    market.clear_cache()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ensure_model()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\nHouse Price Predictor running at  http://127.0.0.1:{port}   (Ctrl+C to stop)\n")
    app.run(host=host, port=port, debug=debug)
