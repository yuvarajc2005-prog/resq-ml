# Project ResQSync: AI-Based Early Warning & Landslide Risk Monitoring (NER)
**Smart India Hackathon (Problem Statement SIH26001)**

ResQSync is a software-driven landslide early warning and risk monitoring system specifically engineered for the high-vulnerability mountain corridors of India's North Eastern Region (NER). This repository houses the complete Machine Learning and Data Engineering pipeline, delivering a lightweight, RAM-optimized LightGBM model for real-time hazard probability scoring alongside TreeSHAP explainability artifacts.

---

## 1. PoC Geographic Scope & Official Data Provenance

The Proof-of-Concept focuses on the **Sikkim & Teesta Valley Corridor** (NER Bounding Box: `[27.0°N – 28.0°N, 88.0°E – 89.0°E]`), spanning Gangtok, Pakyong, Rangpo, Namchi, Mangan, and the critical NH-10 arterial corridor.

* **Target Variable (Historical Landslide Inventory):** Official records from the **Geological Survey of India (GSI) Bhukosh Portal** and the **NASA Global Landslide Catalog (GLC)**.
* **Terrain Features (Slope & Elevation):** High-resolution 30-meter **Copernicus DEM (GLO-30)** GeoTIFF tiles sampled via `rasterio`.
* **Trigger Precipitation Telemetry:** 72-hour antecedent cumulative rainfall (`rainfall_72h`) from gridded meteorological telemetry (**Open-Meteo Historical Archive / ERA5**).
* **Soil & Vulnerability Features:** Depth-specific soil texture metrics (`soil_clay_percent`) from the **ISRIC SoilGrids REST API** (0–5 cm mean values) and vegetation cover/NDVI proxy from **ESA WorldCover 10m**.

---

## 2. Model Configuration & Training Architecture

* **Framework:** LightGBM (`lightgbm.Booster`)
* **Objective:** Binary classification (`objective: "binary"`, `metric: "auc"`)
* **Class Imbalance & Safety Weighting:** Configured with `scale_pos_weight: 10` to heavily penalize false negatives (missed landslides).
* **Tree Parameters:** `learning_rate: 0.05`, `num_leaves: 31`, `min_child_samples: 5`, boosting rounds with early stopping on the 80/20 stratified validation set.
* **Explainability:** `shap.TreeExplainer` generates validation SHAP summary attributions verifying physical consistency.

---

## 3. Key Performance & Safety Metrics

* **Validation ROC-AUC:** `0.9918` (Target Benchmark: `0.88 – 0.92`)
* **Operational Recall (Disaster Detection):** **`100.00%`** (16/16 validation landslides detected, `0` missed events)
* **Operational Precision:** `84.21%` (`3` false alarms in validation set)
* **Safety Strategy (Recall over Precision):** Missing a landslide is fatal. By operating at threshold `0.0800`, zero landslides are missed. Minor false alarms are filtered downstream through the ResQSync **YOLOv11 edge camera / crowdsourced mobile citizen reporting pipeline** before triggering public sirens.
* **Inference Latency:** `< 1.0 ms` per telemetry packet.

---

## 4. Deliverables for FastAPI + PostGIS Backend

All deliverables required for integration into the FastAPI lifespan manager are present in the repository root:

1. **`resqsync_model.txt`**: Serialized raw LightGBM text model artifact.
2. **`features.json`**: Strict ordered array of feature columns expected by the model:
   ```json
   [
     "slope_angle",
     "rainfall_72h",
     "elevation",
     "soil_clay_percent",
     "ndvi"
   ]
   ```
3. **`shap_summary.png`**: TreeSHAP summary plot showing feature importance and directional impact.
4. **`evaluation_summary.txt`**: Full Confusion Matrix, ROC-AUC score, threshold rationale, and SIH pitch documentation.
5. **`src/infer.py`**: Lifespan-ready inference wrapper showing exact memory caching and scoring logic.

---

## 5. Quickstart & Verification

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Acquire datasets and build tabular feature store
python src/data_acquisition.py

# 3. Train model, evaluate metrics, and export deliverables
python src/train.py

# 4. Run real-time inference test scenarios
python src/infer.py
```
