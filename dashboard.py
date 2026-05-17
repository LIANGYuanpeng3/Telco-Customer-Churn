"""
Streamlit 解约风险看板：读取 outputs/bi/ 下 CSV，面向业务展示 KPI、分群、维度分析与阈值模拟。
运行：在项目根目录执行  streamlit run dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent
BI_DIR = BASE_DIR / "outputs" / "bi"
THRESHOLD_ANALYSIS_RF = BASE_DIR / "outputs" / "threshold_analysis_random_forest.csv"
COST_BENEFIT_RF = BASE_DIR / "outputs" / "cost_benefit_random_forest.csv"


def tenure_group_series(ser: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(ser, errors="coerce"),
        bins=[0, 7, 13, 25, 49, np.inf],
        right=False,
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    ).astype(str)


@st.cache_data
def load_bi_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BI_DIR / name)


@st.cache_data
def load_threshold_analysis_fallback() -> pd.DataFrame | None:
    if not THRESHOLD_ANALYSIS_RF.is_file():
        return None
    return pd.read_csv(THRESHOLD_ANALYSIS_RF)


def main():
    st.set_page_config(page_title="Telco 解约风险看板", layout="wide")
    st.title("Telco 顾客解约预测 · 业务看板")
    st.caption("数据来自 `outputs/bi/`（请先运行 `python bi_export.py` 生成 CSV）")

    if not BI_DIR.is_dir():
        st.error(f"未找到目录 `{BI_DIR}`。请在项目根目录运行：`python bi_export.py`")
        st.stop()

    required = [
        "customer_predictions.csv",
        "risk_segment_summary.csv",
        "churn_by_contract.csv",
        "churn_by_tenure_group.csv",
        "retention_action_summary.csv",
        "model_kpi_summary.csv",
        "marketing_kpi_summary.csv",
    ]
    missing = [f for f in required if not (BI_DIR / f).is_file()]
    if missing:
        st.error("缺少以下文件：\n" + "\n".join(missing))
        st.info("运行 `python bi_export.py` 可生成完整 `outputs/bi/` 数据集。")
        st.stop()

    customers = load_bi_csv("customer_predictions.csv")
    risk_seg = load_bi_csv("risk_segment_summary.csv")
    churn_contract = load_bi_csv("churn_by_contract.csv")
    churn_tenure = load_bi_csv("churn_by_tenure_group.csv")
    retention = load_bi_csv("retention_action_summary.csv")
    model_kpi = load_bi_csv("model_kpi_summary.csv")
    marketing_kpi = load_bi_csv("marketing_kpi_summary.csv")

    customers = customers.copy()
    customers["tenure_group"] = tenure_group_series(customers["tenure"])

    # ----- 一、Overall KPI -----
    st.header("一、整体 KPI")
    n = len(customers)
    mcharges = pd.to_numeric(customers["MonthlyCharges"], errors="coerce")
    has_actual = "actual_churn" in customers.columns
    churn_rate = float(customers["actual_churn"].mean()) if has_actual else float("nan")
    avg_prob = float(customers["churn_probability"].mean())
    high_mask = customers["risk_level"] == "High"
    high_n = int(high_mask.sum())
    high_rev = float(mcharges[high_mask].sum(skipna=True))

    kpi_model = model_kpi.iloc[0].to_dict() if len(model_kpi) else {}
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("测试集客户数", f"{n:,}")
    m2.metric("实际解约率", f"{churn_rate:.1%}" if has_actual else "N/A")
    m3.metric("平均预测解约概率", f"{avg_prob:.3f}")
    m4.metric("高风险客户数 (High)", f"{high_n:,}")
    m5.metric("高风险客户月费合计", f"${high_rev:,.0f}")

    if kpi_model:
        st.markdown(
            f"**当前模型**：`{kpi_model.get('model_name', '')}` · "
            f"**业务判定阈值**：`{kpi_model.get('selected_threshold', '')}` · "
            f"**ROC-AUC**：{float(kpi_model.get('roc_auc', 0)):.4f}"
        )
    if len(marketing_kpi):
        mk = marketing_kpi.iloc[0]
        st.caption(
            f"营销口径：挽留触达 {int(mk.get('retention_outreach_count', 0)):,} 人 · "
            f"高风险月费 ${float(mk.get('high_risk_monthly_revenue', 0)):,.0f} · "
            f"收入加权风险分 {float(mk.get('revenue_weighted_risk_score', 0)):,.0f}"
        )

    # ----- 二、Risk Segment -----
    st.header("二、风险分档 (High / Medium / Low)")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig_seg = px.bar(
            risk_seg,
            x="risk_level",
            y="customer_count",
            title="各风险档客户数",
            color="risk_level",
            color_discrete_map={"High": "#c0392b", "Medium": "#f39c12", "Low": "#27ae60"},
        )
        fig_seg.update_layout(showlegend=False, xaxis_title="风险档", yaxis_title="客户数")
        st.plotly_chart(fig_seg, use_container_width=True)
    with c2:
        fig_rev = px.bar(
            risk_seg,
            x="risk_level",
            y="total_monthly_revenue",
            title="各风险档月费收入规模 (合计)",
            color="risk_level",
            color_discrete_map={"High": "#c0392b", "Medium": "#f39c12", "Low": "#27ae60"},
        )
        fig_rev.update_layout(showlegend=False, xaxis_title="风险档", yaxis_title="月费合计 ($)")
        st.plotly_chart(fig_rev, use_container_width=True)

    st.dataframe(risk_seg, use_container_width=True, hide_index=True)

    # ----- 三、Churn Analysis -----
    st.header("三、解约风险 · 多维度视图")

    tab_c, tab_t, tab_p, tab_i = st.tabs(["Contract", "Tenure 分桶", "PaymentMethod", "InternetService"])

    with tab_c:
        fig = px.bar(
            churn_contract,
            x="Contract",
            y="actual_churn_rate",
            title="各合约类型 · 实际解约率",
            text=churn_contract["actual_churn_rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(
            churn_contract,
            x="Contract",
            y="average_predicted_churn_probability",
            title="各合约类型 · 平均预测解约概率",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(churn_contract, use_container_width=True, hide_index=True)

    with tab_t:
        fig = px.bar(
            churn_tenure,
            x="tenure_group",
            y="actual_churn_rate",
            title="网龄分桶 · 实际解约率",
            text=churn_tenure["actual_churn_rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(
            churn_tenure,
            x="tenure_group",
            y="average_predicted_churn_probability",
            title="网龄分桶 · 平均预测概率",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(churn_tenure, use_container_width=True, hide_index=True)

    def dim_summary(col: str) -> pd.DataFrame:
        sub = customers[[col, "churn_probability"]].copy()
        if has_actual:
            sub["actual_churn"] = customers["actual_churn"]
        sub[col] = sub[col].fillna("(缺失)").astype(str)
        g = sub.groupby(col, dropna=False)
        out = g.agg(
            customer_count=("churn_probability", "count"),
            average_predicted_churn_probability=("churn_probability", "mean"),
        ).reset_index()
        if has_actual:
            ac = sub.groupby(col, dropna=False)["actual_churn"].mean().reset_index(name="actual_churn_rate")
            out = out.merge(ac, on=col, how="left")
        return out.sort_values("customer_count", ascending=False)

    with tab_p:
        pm = dim_summary("PaymentMethod")
        if "actual_churn_rate" in pm.columns:
            fig = px.bar(pm, x="PaymentMethod", y="actual_churn_rate", title="支付方式 · 实际解约率")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(
            pm,
            x="PaymentMethod",
            y="average_predicted_churn_probability",
            title="支付方式 · 平均预测概率",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(pm, use_container_width=True, hide_index=True)

    with tab_i:
        inet = dim_summary("InternetService")
        if "actual_churn_rate" in inet.columns:
            fig = px.bar(inet, x="InternetService", y="actual_churn_rate", title="宽带类型 · 实际解约率")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(
            inet,
            x="InternetService",
            y="average_predicted_churn_probability",
            title="宽带类型 · 平均预测概率",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(inet, use_container_width=True, hide_index=True)

    # ----- 四、High risk list -----
    st.header("四、高风险客户清单与筛选")
    f1, f2, f3 = st.columns(3)
    risk_opts = ["全部"] + sorted(customers["risk_level"].dropna().unique().tolist())
    rl = f1.selectbox("风险档", risk_opts, index=risk_opts.index("High") if "High" in risk_opts else 0)
    contracts = ["全部"] + sorted(customers["Contract"].dropna().astype(str).unique().tolist())
    ct = f2.selectbox("Contract", contracts)
    tg_opts = ["全部"] + ["0-6", "7-12", "13-24", "25-48", "49+"]
    tg = f3.selectbox("Tenure 分桶", tg_opts)

    flt = customers.copy()
    if rl != "全部":
        flt = flt[flt["risk_level"] == rl]
    if ct != "全部":
        flt = flt[flt["Contract"].astype(str) == ct]
    if tg != "全部":
        flt = flt[flt["tenure_group"] == tg]

    show_cols = [
        "customerID",
        "churn_probability",
        "risk_level",
        "Contract",
        "tenure",
        "tenure_group",
        "MonthlyCharges",
        "PaymentMethod",
        "InternetService",
        "recommended_action",
    ]
    show_cols = [c for c in show_cols if c in flt.columns]
    st.dataframe(flt[show_cols].sort_values("churn_probability", ascending=False), use_container_width=True, height=400)

    # ----- 五、Cost-benefit -----
    st.header("五、成本收益模拟")
    if COST_BENEFIT_RF.is_file():
        cost_df = pd.read_csv(COST_BENEFIT_RF)
        min_row = cost_df.loc[cost_df["expected_cost"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("最低期望成本阈值", f"{float(min_row['threshold']):.2f}")
        c2.metric("该阈值期望成本", f"${float(min_row['expected_cost']):,.0f}")
        c3.metric("触达人数 (TP+FP)", f"{int(min_row['outreach_count']):,}")
        fig_cost = px.line(
            cost_df,
            x="threshold",
            y="expected_cost",
            title="各阈值下的期望成本（config/cost_config.json）",
            markers=True,
        )
        fig_cost.update_layout(yaxis_title="期望成本 ($)")
        st.plotly_chart(fig_cost, use_container_width=True)
        st.caption("假设见 `config/cost_config.json`；与第四节 F1/recall 阈值选择可不同。")
    else:
        st.info("运行 `python threshold_analysis.py` 生成 `outputs/cost_benefit_random_forest.csv`。")

    # ----- 六、Threshold simulation -----
    st.header("六、阈值模拟")
    st.caption(
        "拖动阈值：将「预测为正类（解约）」定义为 P(churn) ≥ 阈值；"
        "precision / recall / f1 依赖明细表中的 **actual_churn**（请使用最新 `bi_export` 导出）。"
    )

    thr = st.slider("概率阈值", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    y_prob = customers["churn_probability"].values
    y_hat = (y_prob >= thr).astype(int)

    pred_pos = int(y_hat.sum())
    st.metric("该阈值下预测为「解约」客户数", f"{pred_pos:,}")

    if has_actual:
        from business_rules import expected_cost, load_cost_config

        y_true = customers["actual_churn"].astype(int).values
        prec = float(precision_score(y_true, y_hat, zero_division=0))
        rec = float(recall_score(y_true, y_hat, zero_division=0))
        f1 = float(f1_score(y_true, y_hat, zero_division=0))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Precision", f"{prec:.4f}")
        k2.metric("Recall", f"{rec:.4f}")
        k3.metric("F1", f"{f1:.4f}")
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        ec = expected_cost(tp, fp, fn, tn, load_cost_config())
        k4.metric("期望成本", f"${ec:,.0f}")
    else:
        st.warning("当前 `customer_predictions.csv` 无 `actual_churn` 列，无法现场计算分类指标。")
        ta = load_threshold_analysis_fallback()
        if ta is not None:
            st.subheader("备选：预计算阈值扫描 (random_forest)")
            st.caption(f"来源：`{THRESHOLD_ANALYSIS_RF.name}`（运行 `python threshold_analysis.py` 生成）")
            close = ta.iloc[(ta["threshold"] - thr).abs().argsort()[:1]]
            st.dataframe(close, use_container_width=True, hide_index=True)
            st.dataframe(ta, use_container_width=True, height=280, hide_index=True)
        else:
            st.info("可运行 `python threshold_analysis.py` 生成 `threshold_analysis_random_forest.csv` 查看各阈值指标。")

    st.divider()
    st.subheader("模型 KPI（测试集 · 当前导出配置）")
    st.dataframe(model_kpi, use_container_width=True, hide_index=True)
    st.subheader("营销 KPI 汇总")
    st.dataframe(marketing_kpi, use_container_width=True, hide_index=True)
    st.subheader("挽留动作汇总")
    st.dataframe(retention, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
