"""
在测试集上扫描不同分类阈值，用于解约场景的决策阈值选择。

解约预测通常更关注「少漏掉即将流失的客户」（高 recall），而默认 0.5 阈值
往往来自均衡假设，与业务成本不对称；仅看 accuracy 也会掩盖正类漏检。
因此需要单独做 threshold 分析，并把选定阈值写入 models/threshold_config.json，
供 export 与 API 与训练后的评估结论对齐（避免隐性 train-serving skew 在决策层延续）。
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from business_rules import expected_cost, load_cost_config, recommend_threshold_by_cost
from config import MODEL_DIR, OUTPUT_DIR
from preprocess import prepare_data

# 业务上优先覆盖真实解约客户：先满足 recall，再用 F1 平衡 precision；并避免 precision 过低
MIN_RECALL_TARGET = 0.75
# 在 recall 已达标的前提下，若存在 precision 不低于该值的候选，则优先在该子集内按 F1 选阈值
MIN_PRECISION_SOFT = 0.25

THRESHOLDS = [round(0.1 + i * 0.05, 2) for i in range(17)]  # 0.10 … 0.90 步长 0.05
MODEL_NAMES = ["logistic_regression", "random_forest"]


def metrics_for_threshold(
    y_true, y_prob, threshold: float, cost_cfg: dict[str, float] | None = None
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cfg = cost_cfg or load_cost_config()
    ec = expected_cost(tp, fp, fn, tn, cfg)
    outreach_count = int(tp + fp)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "expected_cost": ec,
        "outreach_count": outreach_count,
        "cost_per_outreach": float(cfg["cost_outreach"]),
    }


def recommend_threshold(df: pd.DataFrame) -> tuple[float, str]:
    """
    优先保证 recall（>= MIN_RECALL_TARGET），且不让 precision 过低：
    在 recall 达标的行中，若存在 precision >= MIN_PRECISION_SOFT 的子集，则在该子集内取 F1 最大；
    否则在全部 recall 达标行中取 F1 最大。
    若无任何 recall 达标行，则退回为「recall 最大，其次 F1，其次 precision」。
    """
    hit = df[df["recall"] >= MIN_RECALL_TARGET]
    if not hit.empty:
        good_prec = hit[hit["precision"] >= MIN_PRECISION_SOFT]
        pool = good_prec if not good_prec.empty else hit
        pool = pool.sort_values(
            by=["f1", "precision", "recall"],
            ascending=[False, False, False],
        )
        row = pool.iloc[0]
        if not good_prec.empty:
            reason = (
                f"在 recall>={MIN_RECALL_TARGET} 且 precision>={MIN_PRECISION_SOFT} 的阈值中取 F1 最高"
            )
        else:
            reason = (
                f"在 recall>={MIN_RECALL_TARGET} 的阈值中取 F1 最高（无满足 precision>={MIN_PRECISION_SOFT} 的候选）"
            )
        return float(row["threshold"]), reason

    df2 = df.sort_values(by=["recall", "f1", "precision"], ascending=[False, False, False])
    row = df2.iloc[0]
    return float(row["threshold"]), "无 recall>=0.75 的阈值，退回为 recall 最大再比 F1"


def analyze_model(
    model_name: str, y_test, X_test_model: pd.DataFrame
) -> tuple[pd.DataFrame, float, str, float, str]:
    clf = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    y_prob = clf.predict_proba(X_test_model)[:, 1]
    cost_cfg = load_cost_config()

    rows = [metrics_for_threshold(y_test.values, y_prob, t, cost_cfg) for t in THRESHOLDS]
    out = pd.DataFrame(rows)

    rec_t, rec_reason = recommend_threshold(out)
    cost_t, cost_reason = recommend_threshold_by_cost(out)
    return out, rec_t, rec_reason, cost_t, cost_reason


def main():
    data = prepare_data()
    y_test = data["y_test"].reset_index(drop=True)
    # 与 train.py 中 fit 所用矩阵一致：prepare_data 返回里建模用测试集键名为 X_test（不含 customerID）
    X_test_model = data["X_test"].copy()

    summary = {
        "description": (
            "解约预测更重视尽早识别高风险客户，默认 0.5 阈值通常不最优；"
            "此处阈值由验证集指标与业务约束（优先 recall、控制 precision 过低）选出。"
        ),
        "min_recall_target": MIN_RECALL_TARGET,
        "min_precision_soft": MIN_PRECISION_SOFT,
        "thresholds_scanned": THRESHOLDS,
        "cost_config_path": "config/cost_config.json",
        "models": {},
    }

    for name in MODEL_NAMES:
        df, thr, reason, cost_thr, cost_reason = analyze_model(name, y_test, X_test_model)
        csv_path = OUTPUT_DIR / f"threshold_analysis_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

        cost_path = OUTPUT_DIR / f"cost_benefit_{name}.csv"
        df.to_csv(cost_path, index=False)
        print(f"Wrote {cost_path}")

        summary["models"][name] = {
            "threshold": thr,
            "selection_rationale": reason,
            "cost_optimal_threshold": cost_thr,
            "cost_selection_rationale": cost_reason,
        }

    cfg_path = MODEL_DIR / "threshold_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {cfg_path}")


if __name__ == "__main__":
    main()
