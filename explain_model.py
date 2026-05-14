"""
模型可解释性：从已训练的 Pipeline 中导出逻辑回归系数、随机森林分裂重要性，
以及排列重要性，便于面试与业务沟通「哪些因素驱动解约风险」。
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import MODEL_DIR, OUTPUT_DIR
from preprocess import prepare_data

# 与业务叙事对齐：这些主题在 ColumnTransformer 输出名（num__/cat__ 前缀）中可被匹配
BUSINESS_THEME_KEYS = [
    "contract",
    "tenure",
    "monthlycharges",
    "totalcharges",
    "onlinesecurity",
    "techsupport",
]

PERMUTATION_N_REPEATS = 15
PERMUTATION_RANDOM_STATE = 42


def business_theme_match(feature_name: str) -> str:
    """若特征名关联到业务主题则返回主题标签，否则空字符串。"""
    norm = feature_name.lower().replace(" ", "").replace("_", "")
    hits = [k for k in BUSINESS_THEME_KEYS if k in norm]
    return "|".join(hits) if hits else ""


def load_clf(name: str):
    return joblib.load(MODEL_DIR / f"{name}.joblib")


def explain_logistic_regression(clf) -> pd.DataFrame:
    preprocessor = clf.named_steps["preprocessor"]
    est = clf.named_steps["model"]
    # 与 coef_ 对齐的是 ColumnTransformer.get_feature_names_out()，而非原始列名
    names = preprocessor.get_feature_names_out()
    coef = np.ravel(est.coef_)
    if coef.shape[0] != len(names):
        raise ValueError("系数维度与变换后特征数不一致")

    df = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "business_theme": [business_theme_match(str(n)) for n in names],
        }
    )
    df = df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    df.insert(0, "rank_by_abs_coefficient", range(1, len(df) + 1))
    return df


def explain_random_forest(clf) -> pd.DataFrame:
    preprocessor = clf.named_steps["preprocessor"]
    est = clf.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    imp = est.feature_importances_
    df = pd.DataFrame(
        {
            "feature": names,
            "feature_importance": imp,
            "business_theme": [business_theme_match(str(n)) for n in names],
        }
    )
    df = df.sort_values("feature_importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank_by_importance", range(1, len(df) + 1))
    return df


def explain_random_forest_permutation(clf, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    在原始建模列上打乱，观察对模型表现的影响；列名与业务字段一致，便于向非技术人员解释。
    scoring=roc_auc 对不均衡二分类较稳健。
    """
    result = permutation_importance(
        clf,
        X_test,
        y_test,
        n_repeats=PERMUTATION_N_REPEATS,
        random_state=PERMUTATION_RANDOM_STATE,
        n_jobs=-1,
        scoring="roc_auc",
    )
    df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
            "business_theme": [business_theme_match(str(c)) for c in X_test.columns],
        }
    )
    df = df.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank_by_permutation_mean", range(1, len(df) + 1))
    return df


def build_top_features_summary(
    lr_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    汇总「最重要」与「业务易解释」特征：合并 LR / RF（变换后同一特征空间），
    并附带排列重要性在原始列上的 Top-K；额外保留命中业务主题但未进 Top 的变换后特征。
    """
    lr_slim = lr_df[
        ["feature", "coefficient", "abs_coefficient", "rank_by_abs_coefficient"]
    ].rename(columns={"rank_by_abs_coefficient": "lr_rank_by_abs_coef"})
    rf_slim = rf_df[["feature", "feature_importance", "rank_by_importance"]].rename(
        columns={"rank_by_importance": "rf_rank_by_importance"}
    )
    merged = lr_slim.merge(rf_slim, on="feature", how="outer")
    merged["business_theme"] = merged["feature"].map(lambda x: business_theme_match(str(x)))
    merged["best_rank_lr_rf"] = merged[["lr_rank_by_abs_coef", "rf_rank_by_importance"]].min(
        axis=1, skipna=True
    )
    merged_sorted = merged.sort_values("best_rank_lr_rf", na_position="last")
    top_merged = merged_sorted.head(top_k).copy()
    top_merged["row_type"] = "top_lr_rf_by_min_rank"
    top_merged["summary_block"] = "transformed_features"

    in_top = set(top_merged["feature"])
    business_extra = merged[
        (merged["business_theme"].str.len() > 0) & (~merged["feature"].isin(in_top))
    ].copy()
    business_extra["row_type"] = "business_theme_highlight"
    business_extra["summary_block"] = "transformed_features"

    perm_block = perm_df.head(top_k).copy()
    perm_block["row_type"] = "permutation_raw_feature"
    perm_block["summary_block"] = "raw_features_permutation"

    return pd.concat([top_merged, business_extra, perm_block], ignore_index=True, sort=False)


def main():
    data = prepare_data()
    X_test = data["X_test"].copy()
    y_test = data["y_test"].reset_index(drop=True)

    clf_lr = load_clf("logistic_regression")
    clf_rf = load_clf("random_forest")

    lr_out = explain_logistic_regression(clf_lr)
    lr_path = OUTPUT_DIR / "logistic_regression_coefficients.csv"
    lr_out.to_csv(lr_path, index=False)
    print(f"Wrote {lr_path}")

    rf_out = explain_random_forest(clf_rf)
    rf_path = OUTPUT_DIR / "random_forest_feature_importance.csv"
    rf_out.to_csv(rf_path, index=False)
    print(f"Wrote {rf_path}")

    perm_out = explain_random_forest_permutation(clf_rf, X_test, y_test)
    perm_path = OUTPUT_DIR / "random_forest_permutation_importance.csv"
    perm_out.to_csv(perm_path, index=False)
    print(f"Wrote {perm_path}")

    summary_df = build_top_features_summary(lr_out, rf_out, perm_out)
    summary_path = OUTPUT_DIR / "top_features_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
