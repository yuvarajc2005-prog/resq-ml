"""
ResQSync FastAPI Lifespan Inference Module
Demonstrates loading the serialized model and schema into RAM for sub-millisecond API scoring.
"""

import os
import json
import time
import numpy as np
import lightgbm as lgb
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "resqsync_model.txt")
FEATURES_PATH = os.path.join(BASE_DIR, "features.json")

# In FastAPI, this state lives in `app.state` initialized during lifespan startup
class LandslideHazardPredictor:
    def __init__(self, model_path: str = MODEL_PATH, features_path: str = FEATURES_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature schema not found: {features_path}")

        print(f"Loading LightGBM model into RAM from {model_path}...")
        self.model = lgb.Booster(model_file=model_path)

        with open(features_path, "r", encoding="utf-8") as f:
            self.features = json.load(f)
        print(f"Loaded feature schema: {self.features}")

        # Operational threshold configured for >95% recall safety
        self.hazard_threshold = 0.0800

    def predict(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """
        Score a single coordinate/telemetry packet.
        Strictly enforces feature ordering from features.json.
        """
        # Validate all required features are present
        for f in self.features:
            if f not in telemetry:
                raise ValueError(f"Missing required telemetry feature: '{f}' (Expected: {self.features})")

        # Construct feature vector in strict order
        input_vector = np.array([[float(telemetry[f]) for f in self.features]], dtype=np.float32)

        start_time = time.perf_counter()
        hazard_prob = float(self.model.predict(input_vector)[0])
        inference_latency_ms = (time.perf_counter() - start_time) * 1000.0

        is_hazard = hazard_prob >= self.hazard_threshold
        alert_level = "CRITICAL" if hazard_prob >= 0.50 else ("WARNING" if is_hazard else "SAFE")

        return {
            "hazard_probability": round(hazard_prob, 4),
            "is_hazard_alert": is_hazard,
            "alert_level": alert_level,
            "threshold_used": self.hazard_threshold,
            "inference_latency_ms": round(inference_latency_ms, 3),
            "downstream_action": (
                "Trigger YOLOv11 edge camera / crowdsourced verification"
                if is_hazard else "Standard telemetry logging"
            )
        }

if __name__ == "__main__":
    predictor = LandslideHazardPredictor()

    # Test Case 1: Extreme rainfall on steep slope (Disaster risk scenario in Sikkim)
    high_risk_packet = {
        "slope_angle": 38.5,
        "rainfall_72h": 220.0,
        "elevation": 1750.0,
        "soil_clay_percent": 34.0,
        "ndvi": 0.65
    }
    res1 = predictor.predict(high_risk_packet)
    print("\n--- High Risk Scenario Result ---")
    print(json.dumps(res1, indent=2))

    # Test Case 2: Stable valley floor under moderate rain
    low_risk_packet = {
        "slope_angle": 8.0,
        "rainfall_72h": 45.0,
        "elevation": 600.0,
        "soil_clay_percent": 24.0,
        "ndvi": 0.52
    }
    res2 = predictor.predict(low_risk_packet)
    print("\n--- Low Risk Scenario Result ---")
    print(json.dumps(res2, indent=2))

    # Test Case 3: Flat valley floor during monsoon downpour (No slide due to flat terrain)
    flat_monsoon_packet = {
        "slope_angle": 6.2,
        "rainfall_72h": 180.0,
        "elevation": 450.0,
        "soil_clay_percent": 26.0,
        "ndvi": 0.55
    }
    res3 = predictor.predict(flat_monsoon_packet)
    print("\n--- Flat Valley Monsoon Result ---")
    print(json.dumps(res3, indent=2))
