"""
generate_dataset.py
-------------------
Creates a realistic *synthetic* Bengaluru house-price dataset that follows the
exact column format of the popular Kaggle "Bengaluru House Price Data"
(https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data):

    area_type, availability, location, size, society, total_sqft, bath, balcony, price

The data is deliberately "messy" (just like the real file) so that the project
can demonstrate real data-cleaning steps:
    * size written as "2 BHK", "4 Bedroom", "1 RK"
    * total_sqft sometimes a range ("1100 - 1300") or in other units ("120Sq. Yards")
    * missing values in bath, balcony, society, size, location
    * duplicate rows and a few unrealistic outliers

Price is in LAKHS of rupees (1 lakh = 1,00,000 INR), same as the Kaggle file.

If you download the real Kaggle CSV, just save it as
data/Bengaluru_House_Data.csv and run src/train.py - no code change needed.

Run:  python data/generate_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 13320  # same size as the real Kaggle dataset
OUT_FILE = Path(__file__).resolve().parent / "Bengaluru_House_Data.csv"

# Locality -> approximate base price per sq.ft (INR) and how common it is.
# (Values are illustrative, chosen to look like a real city price map.)
LOCATIONS = {
    # name: (base_price_per_sqft, popularity_weight)
    "Whitefield": (7800, 540), "Sarjapur  Road": (7400, 400), "Electronic City": (5200, 300),
    "Kanakpura Road": (6600, 270), "Thanisandra": (6500, 235), "Yelahanka": (6700, 210),
    "Uttarahalli": (5600, 185), "Hebbal": (10200, 177), "Marathahalli": (8100, 175),
    "Raja Rajeshwari Nagar": (6000, 171), "Bannerghatta Road": (7500, 152),
    "Hennur Road": (7000, 150), "7th Phase JP Nagar": (8800, 149), "Haralur Road": (7600, 141),
    "Electronic City Phase II": (4900, 132), "Rajaji Nagar": (12600, 107), "Chandapura": (3800, 100),
    "Bellandur": (9000, 96), "Hoodi": (7200, 88), "KR Puram": (6000, 88), "Electronics City Phase 1": (5300, 87),
    "Yeshwanthpur": (10200, 85), "Begur Road": (5800, 84), "Sarjapur": (5500, 81),
    "Kasavanhalli": (7200, 80), "Harlur": (7500, 79), "Hormavu": (6300, 74), "Banashankari": (9500, 74),
    "Ramamurthy Nagar": (6800, 73), "Koramangala": (15500, 72), "Kengeri": (5000, 72),
    "Hosa Road": (5500, 72), "Jakkur": (8200, 70), "Old Airport Road": (11000, 69),
    "Varthur": (6200, 68), "Kothanur": (6200, 66), "Akshaya Nagar": (6400, 62),
    "Budigere": (4500, 61), "Jigani": (3800, 60), "Kaggadasapura": (6800, 59),
    "HSR Layout": (11500, 58), "Indira Nagar": (16000, 57), "Malleshwaram": (14500, 57),
    "Panathur": (7000, 55), "Hennur": (6900, 55), "Jayanagar": (14000, 54),
    "Bommanahalli": (6500, 51), "Thigalarapalya": (5600, 48), "Mysore Road": (5800, 47),
    "Brookefield": (8200, 46), "Kundalahalli": (8000, 45), "Talaghattapura": (5000, 44),
    "Attibele": (3500, 43), "Devanahalli": (4800, 42), "Hosakote": (4000, 41),
    "Anekal": (3300, 40), "Vijayanagar": (10500, 39), "Magadi Road": (6200, 38),
    "Basavangudi": (13000, 37), "Frazer Town": (12000, 35), "Sahakara Nagar": (9500, 34),
    "Domlur": (13000, 33), "BTM Layout": (9000, 33), "CV Raman Nagar": (9000, 32),
    "Mahadevpura": (7500, 31), "Doddanekundi": (7600, 30), "RT Nagar": (9800, 28),
    "Nelamangala": (3500, 27), "Ulsoor": (14000, 25), "Sadashiva Nagar": (22000, 20),
}

# Prefix/suffix pieces used to create rare "long tail" localities
# (the real data has ~1300 location names, most appearing only a few times).
_PREFIX = ["Abbigere", "Basava", "Chikka", "Doddaballa", "Gottigere", "Hulimavu", "Kammana",
           "Lingarajapura", "Muneshwara", "Nagavara", "Padmanabha", "Rachenahalli",
           "Shivaji", "Tumkur", "Vidyaranya", "Yelachena", "Bilekahalli", "Chokkasandra",
           "Gubbalala", "Kodigehalli", "Lakshmipura", "Mallathahalli", "Siddapura", "Vasantha"]
_SUFFIX = [" Nagar", " Layout", " Palya", "halli", " Extension", " Main Road", " Cross"]

AREA_TYPES = ["Super built-up  Area", "Built-up  Area", "Plot  Area", "Carpet  Area"]
AREA_TYPE_P = [0.66, 0.18, 0.15, 0.01]
AREA_TYPE_MULT = {"Super built-up  Area": 1.00, "Built-up  Area": 1.03,
                  "Plot  Area": 1.30, "Carpet  Area": 1.12}

BHK_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
BHK_P = [0.06, 0.40, 0.36, 0.12, 0.03, 0.017, 0.008, 0.005]
SQFT_PER_BHK_BASE = {1: 620, 2: 1150, 3: 1580, 4: 2350, 5: 2900, 6: 3400, 7: 3900, 8: 4400}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _rare_locations(rng, n_names=140):
    all_combos = [p + s for p in _PREFIX for s in _SUFFIX]
    names = rng.choice(all_combos, size=min(n_names, len(all_combos)), replace=False)
    # rare localities get a random base price between 3500 and 11000
    return {name: (int(rng.uniform(3500, 11000)), float(rng.uniform(0.3, 2.5))) for name in sorted(names)}


def generate(n_rows=N_ROWS, seed=SEED):
    rng = np.random.default_rng(seed)
    rare = _rare_locations(rng)
    all_locs = {**LOCATIONS, **rare}
    loc_names = list(all_locs)
    weights = np.array([all_locs[l][1] for l in loc_names], dtype=float)
    weights /= weights.sum()

    location = rng.choice(loc_names, size=n_rows, p=weights)
    area_type = rng.choice(AREA_TYPES, size=n_rows, p=AREA_TYPE_P)
    bhk = rng.choice(BHK_VALUES, size=n_rows, p=BHK_P)

    # Plot (independent house) listings are a bit bigger
    plot = area_type == "Plot  Area"
    bhk = np.where(plot & (bhk < 3) & (rng.random(n_rows) < 0.5), bhk + 1, bhk)

    sqft = np.array([SQFT_PER_BHK_BASE[b] for b in bhk]) * rng.lognormal(0, 0.20, n_rows)
    sqft = np.where(plot, sqft * 1.25, sqft)
    sqft = np.round(sqft).astype(int)

    # bathrooms: usually equal to BHK, sometimes one less / one more
    bath = bhk + rng.choice([-1, 0, 0, 0, 1], size=n_rows)
    bath = np.clip(bath, 1, None).astype(float)

    balcony = rng.choice([0, 1, 2, 3], size=n_rows, p=[0.08, 0.38, 0.38, 0.16]).astype(float)

    ready = rng.random(n_rows) < 0.79
    availability = np.where(
        ready, "Ready To Move",
        [f"{rng.integers(17, 23)}-{rng.choice(MONTHS)}" for _ in range(n_rows)],
    )

    # ---------- price model (in lakhs) ----------
    base_ppsf = np.array([all_locs[l][0] for l in location], dtype=float)
    mult = np.array([AREA_TYPE_MULT[a] for a in area_type])
    mult *= np.where(ready, 1.04, 0.94)              # ready homes cost a bit more
    mult *= 1 + 0.03 * (bath - bhk)                  # extra bathrooms add value
    mult *= 1 + 0.012 * balcony                      # balconies add a little
    mult *= (sqft / 1200.0) ** 0.07                  # bigger homes -> slightly higher rate
    # Real markets are not purely "additive", so add two interactions:
    # (a) land (plot) value rises faster in expensive areas
    loc_level = np.clip((base_ppsf - 3300) / (16000 - 3300), 0, 1.3)
    mult *= np.where(plot, 0.85 + 0.45 * loc_level, 1.0)
    # (b) luxury premium: large homes in premium areas sell at a higher rate,
    #     large homes on the outskirts at a lower one
    big = np.clip((sqft - 1800) / 1500, 0, 1.5)
    mult *= 1 + big * (0.35 * loc_level - 0.12)
    noise = rng.lognormal(0, 0.16, n_rows)           # market randomness
    price_lakh = sqft * base_ppsf * mult * noise / 1e5
    price_lakh = np.round(price_lakh, 2)

    # ---------- build text columns the same way the Kaggle file has them ----------
    size = np.where(plot, [f"{b} Bedroom" for b in bhk], [f"{b} BHK" for b in bhk])
    size = np.where((bhk == 1) & (rng.random(n_rows) < 0.15), "1 RK", size)

    society = np.array(["".join(rng.choice(list("ABCDEGHIKLMNOPRSTUVWY"), 4)).title() + "a"
                        for _ in range(n_rows)], dtype=object)
    society[rng.random(n_rows) < 0.41] = None

    total_sqft = sqft.astype(object)
    # ~1.8% ranges like "1100 - 1300"
    idx = rng.random(n_rows) < 0.018
    total_sqft[idx] = [f"{int(s*0.92)} - {int(s*1.08)}" for s in sqft[idx]]
    # ~0.4% in other units
    idx_units = (~idx) & (rng.random(n_rows) < 0.004)
    unit_text = []
    for s in sqft[idx_units]:
        if rng.random() < 0.5:
            unit_text.append(f"{s / 10.7639:.2f}Sq. Meter")
        else:
            unit_text.append(f"{s / 9.0:.2f}Sq. Yards")
    total_sqft[idx_units] = unit_text

    df = pd.DataFrame({
        "area_type": area_type,
        "availability": availability,
        "location": location,
        "size": size,
        "society": society,
        "total_sqft": total_sqft.astype(str),
        "bath": bath,
        "balcony": balcony,
        "price": price_lakh,
    })

    # ---------- inject realistic data problems ----------
    df.loc[rng.random(n_rows) < 0.0055, "bath"] = np.nan
    df.loc[rng.random(n_rows) < 0.046, "balcony"] = np.nan
    df.loc[rng.choice(n_rows, 16, replace=False), "size"] = np.nan
    df.loc[rng.choice(n_rows, 1, replace=False), "location"] = np.nan

    # unrealistic outliers (data-entry mistakes): tiny area for many rooms, crazy prices
    out_idx = rng.choice(n_rows, 160, replace=False)
    half = len(out_idx) // 2
    df.loc[out_idx[:half], "total_sqft"] = rng.integers(300, 900, half).astype(str)
    df.loc[out_idx[:half], "size"] = [f"{b} Bedroom" for b in rng.integers(5, 10, half)]
    df.loc[out_idx[half:], "price"] = np.round(df.loc[out_idx[half:], "price"]
                                               * rng.choice([0.2, 4.5], len(out_idx) - half), 2)

    # duplicate rows (~4%)
    dups = df.sample(frac=0.04, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed)
    df = df.head(n_rows).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(data):,} rows x {data.shape[1]} columns -> {OUT_FILE}")
    print(data.head(10).to_string())
