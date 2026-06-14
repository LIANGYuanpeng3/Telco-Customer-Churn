# Telco Customer Churn Prediction

## 日本語概要

通信会社の顧客データを用いて、解約・離反の可能性を予測する機械学習プロジェクトです。欠損値処理、カテゴリ変数のエンコーディング、特徴量作成、Logistic Regression / Random Forest の比較、ROC-AUC・Precision・Recall などによる評価、閾値調整、説明性出力、BI向けCSV出力、FastAPI / Docker による簡易API化までを実装しています。

## Project Overview

This project builds an end-to-end machine learning pipeline for predicting customer churn from the IBM Telco Customer Churn dataset. The latest version goes beyond notebook-style modeling and includes preprocessing, model training, evaluation, threshold tuning, explainability outputs, BI-ready CSV exports, a FastAPI inference service, Docker support, and a Streamlit dashboard.

## Main Work

- Clean missing values and normalize data types, including `TotalCharges`
- Encode categorical variables and build reusable preprocessing pipelines
- Train and compare Logistic Regression and Random Forest models
- Evaluate models with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and classification reports
- Tune decision thresholds for recall-sensitive churn detection
- Export model metrics, feature importance, customer-level predictions, and BI summary tables
- Provide FastAPI endpoints for single-customer and batch churn prediction
- Package the API with Docker
- Add tests for preprocessing, business rules, and API behavior

## Repository Structure

```text
Telco-Customer-Churn/
  train.py
  evaluate.py
  threshold_analysis.py
  explain_model.py
  export_results.py
  bi_export.py
  app.py
  dashboard.py
  preprocess.py
  business_rules.py
  config.py
  Dockerfile
  tests/
  data/
  models/
  outputs/
```

## Models

The current implementation trains:

- Logistic Regression as an interpretable baseline
- Random Forest as the default nonlinear model for API and BI output

Both models use class weighting to reduce the impact of class imbalance.

## Evaluation

The project emphasizes recall and threshold selection because missing true churn customers can be more costly than contacting extra low-risk customers. Evaluation artifacts are exported under `outputs/`, including model comparison tables, classification reports, confusion matrices, feature importances, threshold analysis results, and BI-ready summary files.

## API

Run the API locally:

```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Main endpoints:

- `GET /health`
- `GET /model_info`
- `POST /predict`
- `POST /predict_batch`

## Dashboard

```bash
streamlit run dashboard.py
```

The dashboard reads the exported BI CSV files and visualizes churn risk, customer segments, retention actions, and model KPIs.

## Technologies

- Python
- pandas
- scikit-learn
- FastAPI
- Docker
- Streamlit
- pytest
- joblib

## Portfolio Summary

Customer churn was predicted from tabular telecom customer data using a reusable machine learning pipeline. The latest project version includes data cleaning, categorical encoding, feature engineering, Logistic Regression and Random Forest comparison, ROC-AUC / precision / recall evaluation, threshold tuning, explainability exports, BI-ready outputs, FastAPI deployment, Docker packaging, and automated tests.
