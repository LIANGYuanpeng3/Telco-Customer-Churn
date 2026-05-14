# Telco Customer Churn — Risk Prediction & Retention Analytics

End-to-end churn modeling on the **IBM Telco Customer Churn** dataset: identify high-risk customers, align decisions with business costs, and ship outputs suitable for **retention programs**, **sales prioritization**, and **customer support** workflows — not only offline notebooks.

---

## 1. Project Overview

This project builds a **binary churn risk** pipeline from raw CSV to **evaluation**, **threshold tuning**, **explainability artifacts**, **BI-ready exports**, a **FastAPI** inference service, and a **Streamlit** dashboard. The default serving model is **Random Forest** (with **Logistic Regression** trained in parallel for comparison and coefficient-based explanation).

Goals:

- Score customers with **P(churn)** and a **business-tuned decision threshold**
- Segment risk (High / Medium / Low) and suggest **retention-style actions**
- Package results for **API consumers** and **BI / dashboard** tools

---

## 2. Business Problem

Churn prediction matters because customer loss is **expensive** and often **detectable early** from behavior and contract patterns.

| Stakeholder need | How this project supports it |
|------------------|-------------------------------|
| **Find at-risk customers early** | Probability scores + risk bands + ranked customer lists |
| **Reduce avoidable churn** | Thresholds tuned toward **recall** so fewer true churners are missed |
| **Prioritize sales / support effort** | BI tables by contract, tenure, payment method, and recommended action |

A model that only optimizes **accuracy** can silently miss churners in imbalanced settings. This repo explicitly treats **recall and threshold choice** as first-class concerns.

---

## 3. Pipeline

High-level flow from data to productized outputs:

```mermaid
flowchart LR
  A[CSV] --> B[Cleaning]
  B --> C[Feature engineering]
  C --> D[sklearn Pipeline]
  D --> E[Model training]
  E --> F[Evaluation]
  F --> G[Threshold optimization]
  G --> H[BI export]
  H --> I[FastAPI inference]
  I --> J[Streamlit dashboard]
```

| Stage | What happens |
|-------|----------------|
| **CSV** | Telco dataset under `data/` |
| **Cleaning & FE** | Shared logic in `preprocess.py` (`basic_cleaning`, `feature_engineering`); inference uses **`prepare_inference_data`** to avoid train–serve skew |
| **sklearn Pipeline** | `ColumnTransformer` (numeric + categorical) + classifier, saved as `models/*.joblib` |
| **Training** | `train.py` — Logistic Regression + Random Forest |
| **Evaluation** | `evaluate.py` — metrics + confusion matrix + feature-importance CSVs under `outputs/` |
| **Threshold optimization** | `threshold_analysis.py` — scan thresholds, write `models/threshold_config.json` |
| **BI export** | `bi_export.py` — curated tables under `outputs/bi/` |
| **FastAPI** | `app.py` — `/health`, `/model_info`, `/predict`, `/predict_batch` |
| **Dashboard** | `dashboard.py` — reads `outputs/bi/*.csv` |

---

## 4. Models

Two models are trained and persisted:

| Model | Role |
|-------|------|
| **Logistic Regression** | Strong **baseline**, fast to train, **linearly interpretable coefficients** on transformed features — useful for “direction of risk” narratives. |
| **Random Forest** | Handles **nonlinearity** and **interactions** between contract, tenure, services, and charges; robust default for **tabular churn** benchmarks. |

Both use **`class_weight="balanced"`** to mitigate class imbalance. The API and BI export path default to **`random_forest.joblib`**.

---

## 5. Evaluation

`evaluate.py` reports more than **accuracy**:

- **Precision** — when we flag churn, how often we are right  
- **Recall** — of true churners, how many we catch (**critical in churn**: missing a churner is often more costly than an extra outreach)  
- **F1** — harmonic balance when both precision and recall matter  
- **ROC-AUC** — ranking quality of predicted probabilities  
- **Confusion matrix** — exported as CSV for audit and reporting  

**Why recall matters here:** in retention, **false negatives** (failing to act on someone who churns) typically dominate **false positives** (extra contact or offer). The project therefore complements raw metrics with **threshold optimization** (below).

---

## 6. Threshold Optimization

`sklearn`’s default `predict()` uses a **0.5** probability cutoff, which is not always appropriate when **false negatives are expensive**.  

`threshold_analysis.py` scans thresholds on the **same held-out split** as training (`prepare_data()`), writes per-threshold metrics to `outputs/threshold_analysis_*.csv`, and saves a recommended operating point to **`models/threshold_config.json`**.  

`export_results.py`, **`bi_export.py`**, and **`app.py`** apply **`P(churn) ≥ threshold`** for the binary **predicted_churn** label, keeping **deployment aligned with evaluation choices**.

If `threshold_config.json` is missing, the code **falls back to 0.5** for a safe default.

---

## 7. Explainability

`explain_model.py` generates artifacts under `outputs/`:

- **Logistic Regression** — coefficients on **post-`ColumnTransformer`** feature names (`outputs/logistic_regression_coefficients.csv`)
- **Random Forest** — `feature_importances_` (`outputs/random_forest_feature_importance.csv`)
- **Permutation importance** (Random Forest) — column-level impact on **ROC-AUC** using **raw** input names (`outputs/random_forest_permutation_importance.csv`) — easier to map to business fields  
- **Summary** — `outputs/top_features_summary.csv` with business-theme tagging (e.g. contract, tenure, charges, security/support)

One-hot encoding splits categorical fields into multiple columns; interpret **groups of levels** (e.g. “contract-related levels”) rather than a single dummy in isolation.

---

## 8. BI Outputs (`outputs/bi/`)

Run **`python bi_export.py`** after training. Tables are flat **CSV** for **Power BI**, **Tableau**, **Looker Studio**, or the included **Streamlit** app.

| File | Typical use |
|------|-------------|
| `customer_predictions.csv` | **High-risk customer list**, drill-through, filters |
| `risk_segment_summary.csv` | **Risk tier** population & **monthly revenue** exposure |
| `churn_by_contract.csv` | **Churn / risk by contract type** |
| `churn_by_tenure_group.csv` | **Tenure buckets** (0–6, 7–12, …) vs churn and model risk |
| `retention_action_summary.csv` | Workload sizing by **recommended retention action** |
| `model_kpi_summary.csv` | Footer KPI strip: model name, **selected threshold**, test metrics |

`customer_predictions.csv` includes **`actual_churn`** for validation and threshold what-if analysis in the dashboard.

---

## 9. FastAPI

**Run (dev):**

```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Inference uses **`prepare_inference_data`** + **column alignment** to the fitted pipeline (see `app.py`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Service status, **`status`**, **`model_loaded`**, model name |
| `/model_info` | `GET` | Model name, version / training time from `training_meta.json`, **selected threshold**, **feature list** |
| `/predict` | `POST` | Single customer JSON → `churn_probability`, `predicted_churn`, `risk_level`, `recommended_action` |
| `/predict_batch` | `POST` | `{ "customers": [ ... ] }` → `{ "results": [ ... ] }` with the same four fields per row |

**curl examples** (after server start):

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/model_info
```

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"gender\":\"Male\",\"SeniorCitizen\":0,\"Partner\":\"Yes\",\"Dependents\":\"No\",\"tenure\":12,\"PhoneService\":\"Yes\",\"MultipleLines\":\"No\",\"InternetService\":\"DSL\",\"OnlineSecurity\":\"No\",\"OnlineBackup\":\"Yes\",\"DeviceProtection\":\"No\",\"TechSupport\":\"No\",\"StreamingTV\":\"No\",\"StreamingMovies\":\"No\",\"Contract\":\"Month-to-month\",\"PaperlessBilling\":\"Yes\",\"PaymentMethod\":\"Electronic check\",\"MonthlyCharges\":53.85,\"TotalCharges\":468.35}"
```

```bash
curl -s -X POST http://127.0.0.1:8000/predict_batch -H "Content-Type: application/json" -d "{\"customers\":[{\"gender\":\"Male\",\"SeniorCitizen\":0,\"Partner\":\"Yes\",\"Dependents\":\"No\",\"tenure\":12,\"PhoneService\":\"Yes\",\"MultipleLines\":\"No\",\"InternetService\":\"DSL\",\"OnlineSecurity\":\"No\",\"OnlineBackup\":\"Yes\",\"DeviceProtection\":\"No\",\"TechSupport\":\"No\",\"StreamingTV\":\"No\",\"StreamingMovies\":\"No\",\"Contract\":\"Month-to-month\",\"PaperlessBilling\":\"Yes\",\"PaymentMethod\":\"Electronic check\",\"MonthlyCharges\":53.85,\"TotalCharges\":468.35}]}"
```

- **422** — validation errors (missing / wrong types), structured body  
- **503** — model failed to load (e.g. sklearn version mismatch with pickled pipeline)  
- **400** — preprocessing / column alignment failure  

---

## 10. Dashboard

Requires **`outputs/bi/*.csv`** (from `bi_export.py`):

```bash
streamlit run dashboard.py
```

The app summarizes KPIs, risk segments, churn views (contract, tenure, payment method, internet), customer filters, and a **threshold slider** (uses `actual_churn` when present).

---

## 11. Docker

From the repository root:

```bash
docker build -t telco-churn-api .
docker run --rm -p 8000:8000 telco-churn-api
```

The image starts **`uvicorn app:app --host 0.0.0.0 --port 8000`**. Mount or bake `models/*.joblib` and config JSONs as needed for your environment.

---

## 12. Key Learnings (portfolio angle)

- **ML is not only training** — the same probability must connect to **business rules** (threshold, risk tiers, recommended actions) and to **downstream systems** (API + BI).  
- **Churn is recall-sensitive** — optimizing only accuracy or a fixed 0.5 cut can **hide poor recall**; explicit threshold analysis makes trade-offs visible.  
- **Explainability has multiple lenses** — coefficients, tree importances, and permutation importance answer different “why” questions for stakeholders.  
- **Train–serve consistency** — sharing `prepare_inference_data` with the API reduced **feature drift** between offline training and online inference.  
- **Engineering hygiene** — FastAPI health/model metadata, BI exports, Streamlit demo, **`pytest tests/`**, and Docker give a **credible end-to-end story** for interviews.

---

## Quickstart (scripts)

| Step | Command |
|------|---------|
| Train | `python train.py` |
| Evaluate | `python evaluate.py` |
| Thresholds | `python threshold_analysis.py` |
| Explain | `python explain_model.py` |
| BI CSVs | `python bi_export.py` |
| API | `uvicorn app:app --reload` |
| Dashboard | `streamlit run dashboard.py` |
| Tests | `pytest tests/` |

**Note:** `joblib` models are sensitive to **scikit-learn version**. If loading fails, re-train in your current environment.

---

## Dataset

IBM Sample **Telco Customer Churn** dataset (included under `data/` for reproducibility). Verify license / usage terms for your own redistribution policy.

---

## Dependencies

See **`requirements.txt`** (includes `fastapi`, `uvicorn`, `scikit-learn`, `streamlit`, `plotly`, `pytest`, `httpx`, etc.).
