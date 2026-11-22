"""
Automatic checks for the whole project.

Run from the project folder:   python -m unittest discover tests -v
(The model must be trained first: python src/train.py)
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import market  # noqa: E402
from charts import price_curve_svg  # noqa: E402
from data_prep import clean_data, load_raw, size_to_bhk, sqft_to_number  # noqa: E402
from predict import format_price, predict_price, price_curve  # noqa: E402


class TestCleaning(unittest.TestCase):
    def test_size_to_bhk(self):
        self.assertEqual(size_to_bhk("2 BHK"), 2)
        self.assertEqual(size_to_bhk("4 Bedroom"), 4)
        self.assertEqual(size_to_bhk("1 RK"), 1)

    def test_sqft_to_number(self):
        self.assertEqual(sqft_to_number("1200"), 1200)
        self.assertEqual(sqft_to_number("1100 - 1300"), 1200)
        self.assertAlmostEqual(sqft_to_number("100Sq. Yards"), 900)
        self.assertAlmostEqual(sqft_to_number("10Sq. Meter"), 107.639)
        self.assertNotEqual(sqft_to_number("abc"), sqft_to_number("abc"))  # NaN

    def test_clean_data_has_no_missing_core_values(self):
        df = clean_data(load_raw(), verbose=False)
        self.assertGreater(len(df), 1000)
        for col in ["location", "area_type", "total_sqft", "bhk", "price"]:
            self.assertEqual(df[col].isna().sum(), 0, col)
        self.assertTrue((df["total_sqft"] / df["bhk"] >= 300).all())


class TestPrediction(unittest.TestCase):
    def test_prediction_is_positive_with_range(self):
        r = predict_price("Whitefield", 1200, 2, 2)
        self.assertGreater(r["price_lakh"], 0)
        self.assertLess(r["range_low_lakh"], r["price_lakh"])
        self.assertGreater(r["range_high_lakh"], r["price_lakh"])

    def test_bigger_house_costs_more(self):
        small = predict_price("Whitefield", 1000, 2, 2)["price_lakh"]
        big = predict_price("Whitefield", 2000, 3, 3)["price_lakh"]
        self.assertGreater(big, small)

    def test_premium_location_costs_more(self):
        outskirts = predict_price("Anekal", 1500, 3, 3)["price_lakh"]
        premium = predict_price("Koramangala", 1500, 3, 3)["price_lakh"]
        self.assertGreater(premium, outskirts)

    def test_unknown_location_uses_other(self):
        r = predict_price("Some New Place", 1200, 2, 2)
        self.assertFalse(r["known_location"])
        self.assertEqual(r["location_used"], "other")

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            predict_price("Whitefield", 400, 5, 2)  # too small for 5 bedrooms
        with self.assertRaises(ValueError):
            predict_price("Whitefield", 1200, 2, 2, area_type="Castle")

    def test_price_curve_rises_with_size(self):
        curve = price_curve("Whitefield", 1200, 2, 2)
        self.assertGreater(len(curve), 10)
        prices = [p["price_lakh"] for p in curve]
        self.assertEqual(prices, sorted(prices))
        self.assertIn(1200.0, [p["sqft"] for p in curve])

    def test_format_price(self):
        self.assertEqual(format_price(85.4), "₹85.40 Lakh")
        self.assertEqual(format_price(152.3), "₹1.52 Crore")


class TestMarketAndCharts(unittest.TestCase):
    def test_overall_stats(self):
        stats = market.overall_stats()
        self.assertGreater(stats["listings"], 1000)
        self.assertGreater(stats["priciest_ppsf"], stats["cheapest_ppsf"])

    def test_location_table_sorted_by_rate(self):
        rates = [r["median_ppsf"] for r in market.location_table()]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_similar_houses_same_location(self):
        houses = market.similar_houses("Whitefield", 1200, 2, n=5)
        self.assertEqual(len(houses), 5)
        self.assertTrue(all(h["location"] == "Whitefield" for h in houses))

    def test_chart_svg(self):
        curve = price_curve("Hebbal", 1500, 3, 3)
        svg = str(price_curve_svg(curve, 1500, predict_price("Hebbal", 1500, 3, 3)["price_lakh"]))
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>"))
        self.assertIn("This house", svg)


class TestWebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_pages_load(self):
        for url, text in [("/", b"What is this house worth?"),
                          ("/market", b"Bengaluru prices by locality"),
                          ("/model", b"How the estimate is made")]:
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200, url)
            self.assertIn(text, r.data)

    def test_form_estimate(self):
        r = self.client.post("/", data={"location": "Hebbal", "total_sqft": "1500", "bhk": "3",
                                        "bath": "3", "balcony": "2",
                                        "area_type": "Super built-up Area", "ready_to_move": "1"})
        self.assertEqual(r.status_code, 200)
        for text in [b"Estimated price", b"Price at other sizes", b"Similar houses", b"Your recent estimates"]:
            self.assertIn(text, r.data)

    def test_link_estimate_and_error(self):
        r = self.client.get("/?location=Whitefield&total_sqft=1200&bhk=2&bath=2")
        self.assertIn(b"likely range", r.data)
        r = self.client.get("/?location=Whitefield&total_sqft=100&bhk=4&bath=2")
        self.assertIn(b"Check the house details", r.data)

    def test_api(self):
        r = self.client.post("/api/predict", json={"location": "Hebbal", "total_sqft": 1500,
                                                   "bhk": 3, "bath": 3})
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.get_json()["price_lakh"], 0)
        r = self.client.post("/api/predict", json={"location": "Hebbal", "total_sqft": 10,
                                                   "bhk": 3, "bath": 3})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.get("/api/health").get_json()["status"], "ok")
        self.assertGreater(len(self.client.get("/api/locations").get_json()), 10)

    def test_404(self):
        self.assertEqual(self.client.get("/no-such-page").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
