import unittest

from fastapi.testclient import TestClient

from api.predict import app, dashboard_data, predict


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.features = {
            "age": 50,
            "anaemia": 1,
            "creatinine_phosphokinase": 2000,
            "diabetes": 0,
            "ejection_fraction": 30,
            "high_blood_pressure": 1,
            "platelets": 250000,
            "serum_creatinine": 1.2,
            "serum_sodium": 135,
            "sex": 1,
            "smoking": 0,
            "time": 100,
        }

    def test_dashboard_returns_model_metrics_and_training_ranges(self):
        result = dashboard_data()

        self.assertEqual(len(result["features"]), 12)
        self.assertEqual(result["metadata"]["selected_model"], "Random Forest")
        self.assertIn("roc_auc", result["metadata"]["test_metrics"])
        sex = next(feature for feature in result["features"] if feature["name"] == "sex")
        self.assertEqual(sex["options"][1]["label"], "1 — Male")

    def test_prediction_returns_both_probabilities_and_local_sensitivity(self):
        result = predict({"features": self.features})

        self.assertIn(result["predicted_class"], (0, 1))
        self.assertAlmostEqual(result["positive_probability"] + result["negative_probability"], 1.0)
        self.assertEqual(result["threshold"], 0.5)
        self.assertEqual(len(result["local_sensitivity"]), 12)
        self.assertEqual(result["features"], self.features)

    def test_prediction_rejects_values_outside_training_range(self):
        invalid = dict(self.features, age=39)

        with self.assertRaisesRegex(ValueError, "observed training range"):
            predict({"features": invalid})

    def test_prediction_rejects_non_binary_input(self):
        invalid = dict(self.features, smoking=0.5)

        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            predict({"features": invalid})

    def test_prediction_rejects_missing_input(self):
        invalid = dict(self.features)
        del invalid["time"]

        with self.assertRaisesRegex(ValueError, "Missing required values"):
            predict({"features": invalid})


class VercelAsgiAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.features = {
            "age": 50,
            "anaemia": 1,
            "creatinine_phosphokinase": 2000,
            "diabetes": 0,
            "ejection_fraction": 30,
            "high_blood_pressure": 1,
            "platelets": 250000,
            "serum_creatinine": 1.2,
            "serum_sodium": 135,
            "sex": 1,
            "smoking": 0,
            "time": 100,
        }

    def test_root_serves_dashboard(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("HeartGuard AI", response.text)

    def test_static_assets_are_served(self):
        self.assertEqual(self.client.get("/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/app.js").status_code, 200)

    def test_dashboard_endpoint_returns_metadata(self):
        response = self.client.get("/api/predict")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["metadata"]["selected_model"], "Random Forest")

    def test_prediction_endpoint_returns_probabilities(self):
        response = self.client.post("/api/predict", json={"features": self.features})

        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(
            response.json()["positive_probability"] + response.json()["negative_probability"],
            1.0,
        )

    def test_unknown_report_is_not_served(self):
        response = self.client.get("/reports/../../models/model_metadata.json")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
