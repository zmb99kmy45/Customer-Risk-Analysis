import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Optional: SHAP
import shap
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Wesype Risk App", layout="wide")
st.title("Customer Risk Analysis")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded is None:
    st.info("Please upload your dataset to start analysis.")
    st.stop()

df = pd.read_csv(uploaded)

# -----------------------------
# Basic checks + typing
# -----------------------------
required_cols = {
    "customer_id",
    "age",
    "gender",
    "country",
    "annual_income",
    "spending_score",
    "num_purchases",
    "avg_purchase_value",
    "membership_years",
    "website_visits_per_month",
    "cart_abandon_rate",
    "churned",
    "feedback_text",
    "last_purchase_date",
}
missing = required_cols - set(df.columns)
if missing:
    st.warning(f"Missing columns (some features may not work): {sorted(list(missing))}")

# Ensure churned numeric
if "churned" in df.columns:
    df["churned"] = pd.to_numeric(df["churned"], errors="coerce")

# Features used for ML (numeric only baseline)
features = [
    "age",
    "annual_income",
    "spending_score",
    "num_purchases",
    "avg_purchase_value",
    "membership_years",
    "website_visits_per_month",
    "cart_abandon_rate",
]
available_features = [c for c in features if c in df.columns]

if "churned" not in df.columns or df["churned"].isna().any():
    # you said missing values are 0, but keep it robust
    df = df.dropna(subset=["churned"])

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Model controls")

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.30, 0.05)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", min_value=1, value=42)

    cost_intervention = st.slider("Cost per intervention (€)", 10, 200, 50)
    cost_churn = st.slider("Cost of churn (€)", 100, 2000, 500)

    train_btn = st.button("Train / Refresh models")


# -----------------------------
# Train models once (session_state)
# -----------------------------
def train_all_models(df_: pd.DataFrame):
    # keep only numeric features and target
    X = df_[available_features].copy()
    y = df_["churned"].astype(int).copy()

    # fill numeric (should be none per your data, but robust)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=int(random_state)
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=int(random_state)),
    }

    for m in models.values():
        m.fit(X_train, y_train)

    return X, y, X_train, X_test, y_train, y_test, models


if train_btn or ("models" not in st.session_state):
    X, y, X_train, X_test, y_train, y_test, models = train_all_models(df)
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["models"] = models
    st.session_state["trained"] = True

# Pull from session
models = st.session_state.get("models", {})
X = st.session_state.get("X")
y = st.session_state.get("y")
X_train = st.session_state.get("X_train")
X_test = st.session_state.get("X_test")
y_train = st.session_state.get("y_train")
y_test = st.session_state.get("y_test")

if not st.session_state.get("trained", False):
    st.info("Click **Train / Refresh models** in the sidebar.")
    st.stop()

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(
    [
        "EDA",
        "Baseline Model",
        "Model Comparison",
        "Explainability",
        "AI Agent",
        "Executive Summary",
    ]
)

# -----------------------------
# TAB 1: EDA
# -----------------------------
with tabs[0]:
    st.header("EDA")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    churn_rate = (
        float(df["churned"].mean()) if "churned" in df.columns else float("nan")
    )
    c3.metric("Churn rate", "N/A" if np.isnan(churn_rate) else f"{churn_rate*100:.2f}%")

    st.subheader("Data types")
    st.write(df.dtypes)

    st.subheader("Missing values")
    st.write(df.isnull().sum())

    st.subheader("Target distribution")
    fig = plt.figure()
    df["churned"].value_counts().plot(kind="bar")
    plt.title("Churn Distribution (0=Active, 1=Churned)")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Numeric summary")
    st.write(df[available_features].describe())

    st.subheader("Boxplot by churn")
    feature = st.selectbox("Select feature", available_features, index=0)
    tmp = df[[feature, "churned"]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp["churned"] = pd.to_numeric(tmp["churned"], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp) < 10:
        st.warning("Not enough rows for this boxplot.")
    else:
        fig = plt.figure(figsize=(8, 4), dpi=110)
        tmp.boxplot(column=feature, by="churned")
        plt.title(f"{feature} by churn")
        plt.suptitle("")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Correlation matrix")
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        df[available_features + ["churned"]].corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
    )
    st.pyplot(fig, clear_figure=True)

    st.subheader("Correlation with churn")
    corr_with_target = df.corr(numeric_only=True)["churned"].sort_values(
        ascending=False
    )
    st.write(corr_with_target)

    if "feedback_text" in df.columns:
        st.subheader("Feedback samples")
        colA, colB = st.columns(2)
        with colA:
            st.caption("Churned (1)")
            st.write(df[df["churned"] == 1]["feedback_text"].head(5).tolist())
        with colB:
            st.caption("Active (0)")
            st.write(df[df["churned"] == 0]["feedback_text"].head(5).tolist())

# -----------------------------
# TAB 2: Baseline Model
# -----------------------------
with tabs[1]:
    st.header("Baseline Model (Logistic Regression)")

    lr = models["Logistic Regression"]
    probs = lr.predict_proba(X_test)[:, 1]
    preds = (probs >= float(threshold)).astype(int)

    auc = roc_auc_score(y_test, probs)
    st.metric("ROC-AUC", f"{auc:.3f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    st.write(cm)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, preds))

    st.caption(
        "Tip: adjust the threshold in the sidebar to trade precision vs recall for churn (class 1)."
    )

# -----------------------------
# TAB 3: Model Comparison
# -----------------------------
with tabs[2]:
    st.header("Model Comparison")

    rows = []
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= float(threshold)).astype(int)
        auc = roc_auc_score(y_test, probs)
        rep = classification_report(y_test, preds, output_dict=True)
        rows.append(
            {
                "Model": name,
                "ROC-AUC": round(float(auc), 3),
                "Recall (Churn)": round(float(rep["1"]["recall"]), 3),
                "Precision (Churn)": round(float(rep["1"]["precision"]), 3),
                "F1 (Churn)": round(float(rep["1"]["f1-score"]), 3),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False)
    st.dataframe(results_df, use_container_width=True)

    st.caption("All models are evaluated with the SAME threshold set in the sidebar.")

# -----------------------------
# TAB 4: Explainability
# -----------------------------
with tabs[3]:
    st.header("Explainability")

    gb = models["Gradient Boosting"]
    probs_gb = gb.predict_proba(X_test)[:, 1]
    preds_gb = (probs_gb >= float(threshold)).astype(int)

    st.subheader("Feature Importance (Gradient Boosting)")
    importances = gb.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": available_features, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
    st.dataframe(importance_df, use_container_width=True)

    fig = plt.figure(figsize=(8, 4), dpi=110)
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (GB)")
    st.pyplot(fig, clear_figure=True)

    st.subheader("SHAP (global) — Gradient Boosting")
    st.caption("This explains which features drive predictions across the test set.")
    explainer = shap.Explainer(gb, X_train)
    shap_values = explainer(X_test)

    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Calibration curve (Gradient Boosting)")
    prob_true, prob_pred = calibration_curve(y_test, probs_gb, n_bins=10)

    fig = plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration curve")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Business impact simulation (using current threshold)")
    cm = confusion_matrix(y_test, preds_gb)
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])

    cost_model = fp * int(cost_intervention) + fn * int(cost_churn)
    cost_no_model = int(y_test.sum()) * int(cost_churn)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cost without model (€)", f"{cost_no_model:,}".replace(",", " "))
    c2.metric("Cost with model (€)", f"{cost_model:,}".replace(",", " "))
    c3.metric(
        "Estimated savings (€)", f"{(cost_no_model - cost_model):,}".replace(",", " ")
    )

# -----------------------------
# TAB 5: AI Agent (Dust-style)
# -----------------------------
with tabs[4]:
    st.header("AI Agent (Dust-style)")

    gb = models["Gradient Boosting"]

    if "customer_id" not in df.columns:
        st.error("customer_id column is required for the Agent tab.")
        st.stop()

    customer_ids = df["customer_id"].astype(str).unique().tolist()
    selected_id = st.selectbox("Select customer_id", sorted(customer_ids))

    row = df[df["customer_id"].astype(str) == str(selected_id)].iloc[0]
    row_X = pd.DataFrame([row[available_features].to_dict()])
    row_X = row_X.apply(pd.to_numeric, errors="coerce").fillna(
        X_train.median(numeric_only=True)
    )

    risk_prob = float(gb.predict_proba(row_X)[:, 1][0])
    st.metric("Churn risk probability", f"{risk_prob:.2f}")

    med = df[available_features].median(numeric_only=True)

    def driver_notes(row_series):
        notes = []
        if row_series.get("spending_score", med.get("spending_score", 0)) < med.get(
            "spending_score", 0
        ):
            notes.append(("spending_score", "low engagement / loyalty signal", "HIGH"))
        if row_series.get("cart_abandon_rate", 0) > med.get("cart_abandon_rate", 0):
            notes.append(
                ("cart_abandon_rate", "high friction / failed purchase journey", "HIGH")
            )
        if row_series.get("website_visits_per_month", 0) < med.get(
            "website_visits_per_month", 0
        ):
            notes.append(
                ("website_visits_per_month", "low activity / weak habit", "MED")
            )
        if row_series.get("num_purchases", 0) < med.get("num_purchases", 0):
            notes.append(("num_purchases", "low value realization", "MED"))
        if row_series.get("membership_years", 0) < med.get("membership_years", 0):
            notes.append(("membership_years", "onboarding risk", "LOW"))
        return notes

    drivers = driver_notes(row)

    st.subheader("Top drivers (Agent explanation)")
    if not drivers:
        st.write("No strong negative signals vs portfolio median.")
    else:
        for f, why, sev in drivers:
            st.write(f"- **{f}** ({sev}): {why}")

    def generate_playbook(prob, drivers, feedback_text=""):
        if prob >= 0.7:
            tier = "HIGH"
        elif prob >= 0.4:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        actions = [f"**Risk tier:** {tier} (p={prob:.2f})"]
        driver_names = {d[0] for d in drivers}

        if "cart_abandon_rate" in driver_names:
            actions.append(
                "Run a *quality & friction audit*: identify drop-off points and fix top 3 causes."
            )
        if "spending_score" in driver_names or "num_purchases" in driver_names:
            actions.append(
                "Schedule a *value workshop*: define 2 quick wins with measurable ROI within 2 weeks."
            )
        if "website_visits_per_month" in driver_names:
            actions.append(
                "Launch an *enablement plan*: champions, templates, weekly office hours for 3–4 weeks."
            )
        if "membership_years" in driver_names:
            actions.append(
                "Improve onboarding: guided setup + first success in <30 minutes + clear next steps."
            )

        if isinstance(feedback_text, str) and feedback_text.strip():
            actions.append(
                f"Feedback signal: “{feedback_text[:120]}{'...' if len(feedback_text)>120 else ''}” → address this pain point."
            )

        actions.append(
            "Track weekly KPIs: active usage / success rate / time saved. Review bi-weekly with stakeholders."
        )
        return actions

    def draft_message(prob, drivers):
        tier = "high" if prob >= 0.7 else ("medium" if prob >= 0.4 else "low")
        top = [d[0] for d in drivers][:3]
        top_txt = ", ".join(top) if top else "usage signals"
        return (
            f"Hi team,\n\n"
            f"We detected a **{tier} churn risk** signal (p={prob:.2f}). The main drivers are: {top_txt}.\n\n"
            f"Proposed next steps for the coming 2 weeks:\n"
            f"- 60–90 min workshop to confirm 2–3 high-ROI use cases\n"
            f"- Enablement plan (champions + templates + weekly office hours)\n"
            f"- Quality/friction review to reduce drop-offs\n\n"
            f"I can share a short dashboard view and align on KPIs (usage, success rate, time saved).\n\n"
            f"Best,\nFadhila"
        )

    st.subheader("Agent output: recommended plan")
    playbook = generate_playbook(risk_prob, drivers, row.get("feedback_text", ""))
    for i, a in enumerate(playbook, 1):
        st.write(f"{i}. {a}")

    st.subheader("Agent output: message draft (client-facing)")
    st.text_area("Copy/paste message", draft_message(risk_prob, drivers), height=220)

    st.subheader("Agent output: JSON (ready for an agent platform)")

    def build_agent_json(customer_id, prob, drivers, row, threshold_value):
        driver_objs = [
            {"feature": f, "severity": sev, "explanation": why}
            for (f, why, sev) in drivers
        ]

        if prob >= 0.7:
            tier = "HIGH"
        elif prob >= 0.4:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        payload = {
            "agent_name": "wesype-risk-analyst",
            "version": "1.0",
            "input": {
                "customer_id": str(customer_id),
                "features": {
                    k: (None if pd.isna(row.get(k)) else float(row.get(k)))
                    for k in available_features
                },
                "feedback_text": (
                    ""
                    if pd.isna(row.get("feedback_text"))
                    else str(row.get("feedback_text"))
                ),
                "last_purchase_date": (
                    ""
                    if pd.isna(row.get("last_purchase_date"))
                    else str(row.get("last_purchase_date"))
                ),
            },
            "model": {
                "type": "GradientBoostingClassifier",
                "decision_threshold": float(threshold_value),
                "risk_probability": float(prob),
                "risk_tier": tier,
            },
            "explanations": {
                "drivers": driver_objs,
                "notes": "Drivers are derived from portfolio-median comparisons for fast, stable local explanations (can be replaced with SHAP local explanations).",
            },
            "actions": {
                "playbook": playbook,
                "message_draft": draft_message(prob, drivers),
            },
            "kpis_to_track": [
                "weekly_active_usage",
                "success_rate",
                "time_saved_estimate",
                "user_feedback_trend",
            ],
        }
        return payload

    agent_payload = build_agent_json(selected_id, risk_prob, drivers, row, threshold)

    st.code(json.dumps(agent_payload, indent=2), language="json")
    st.download_button(
        label="Download agent JSON",
        data=json.dumps(agent_payload, indent=2).encode("utf-8"),
        file_name=f"agent_payload_{selected_id}.json",
        mime="application/json",
    )

# -----------------------------
# TAB 6: Executive Summary
# -----------------------------
with tabs[5]:
    st.header("Executive Summary")

    churn_rate = float(df["churned"].mean())
    st.write(f"- Dataset size: **{len(df)}** rows")
    st.write(f"- Churn rate: **{churn_rate*100:.2f}%** (imbalanced)")

    # show best model by AUC
    aucs = []
    for name, m in models.items():
        p = m.predict_proba(X_test)[:, 1]
        aucs.append((name, float(roc_auc_score(y_test, p))))
    best_name, best_auc = sorted(aucs, key=lambda x: x[1], reverse=True)[0]

    st.write(f"- Best ROC-AUC model: **{best_name}** with **{best_auc:.3f}**")

    st.subheader("What this project demonstrates")
    st.write(
        """
- Solid EDA and correct handling of class imbalance (thresholding + relevant metrics).
- Model benchmarking (LogReg / RF / Gradient Boosting) with consistent evaluation.
- Explainability (feature importance + SHAP) and probability calibration to ensure trust.
- Operationalization mindset: AI Agent outputs playbooks, client-ready messages, and agent-ready JSON.
"""
    )

    st.subheader("Conclusion")
    st.write(
        "I built an AI Risk Agent that predicts churn risk, explains the main drivers, validates model reliability (calibration), "
        "and turns predictions into action plans and client-ready communication—packaged as a Streamlit app and exportable agent JSON."
    )
