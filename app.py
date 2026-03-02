import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# -----------------------------
# Inspired by your original project
# -----------------------------
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, InputLayer, SimpleRNN
from tensorflow.keras.models import Sequential


def get_train_test_from_series(series_1d: np.ndarray, split_percent=0.8):
    """
    Same spirit as your original get_train_test():
    - clean NaNs/Infs
    - scale to [0,1]
    - flatten to 1D
    - split 80/20
    """
    # 1) clean
    s = pd.Series(series_1d).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30:
        raise ValueError(
            f"Time series too short after cleaning ({len(s)} points). "
            "Try monthly aggregation or reduce time_steps."
        )

    # 2) scale
    series_2d = s.values.astype("float32").reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(series_2d).flatten()

    # 3) split
    n = len(data)
    split = int(n * split_percent)
    train_data = data[:split]
    test_data = data[split:]

    return train_data, test_data, data, scaler


def get_XY(dat, time_steps):
    """
    Same logic as your get_XY():
    Take blocks of time_steps and predict the next value.
    """
    # target indices (every time_steps)
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]

    rows_x = len(Y)
    X = dat[range(time_steps * rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))
    return X, Y


def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(SimpleRNN(hidden_units, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def create_LSTM(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(LSTM(hidden_units, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def plot_result_streamlit(trainY, testY, train_pred, test_pred, title):
    actual = np.append(trainY, testY)
    predictions = np.append(train_pred, test_pred)
    rows = len(actual)

    fig = plt.figure(figsize=(14, 5), dpi=110)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color="r")
    plt.legend(["Actual", "Predictions"])
    plt.xlabel("Observation number (per block of time_steps)")
    plt.ylabel("Scaled value")
    plt.title(title)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Time series building from Kaggle-like CSV
# -----------------------------
def build_time_series(df: pd.DataFrame, freq: str, target_metric: str):
    d = df.copy()

    # Parse date
    d["last_purchase_date"] = pd.to_datetime(d["last_purchase_date"], errors="coerce")
    valid_dates = int(d["last_purchase_date"].notna().sum())
    if valid_dates < 10:
        raise ValueError(
            "Too few valid last_purchase_date values after parsing. Check date format in CSV."
        )

    d = d.dropna(subset=["last_purchase_date"])

    # Ensure numeric columns used for metrics
    d["churned"] = pd.to_numeric(d["churned"], errors="coerce")
    d["cart_abandon_rate"] = pd.to_numeric(d["cart_abandon_rate"], errors="coerce")
    d["website_visits_per_month"] = pd.to_numeric(
        d["website_visits_per_month"], errors="coerce"
    )
    d["num_purchases"] = pd.to_numeric(d["num_purchases"], errors="coerce")

    # Period grouping
    d["period"] = d["last_purchase_date"].dt.to_period(freq).dt.to_timestamp()

    agg = (
        d.groupby("period")
        .agg(
            churn_rate=("churned", "mean"),
            avg_cart_abandon_rate=("cart_abandon_rate", "mean"),
            avg_website_visits_per_month=("website_visits_per_month", "mean"),
            avg_num_purchases=("num_purchases", "mean"),
            volume=("customer_id", "count"),
        )
        .reset_index()
        .sort_values("period")
    )

    if agg.empty:
        raise ValueError(
            "Aggregation produced an empty time series. Check last_purchase_date coverage."
        )

    # Build full date index to fill gaps
    if freq == "W":
        full_index = pd.date_range(agg["period"].min(), agg["period"].max(), freq="W")
    else:
        full_index = pd.date_range(agg["period"].min(), agg["period"].max(), freq="MS")

    agg = (
        agg.set_index("period")
        .reindex(full_index)
        .reset_index()
        .rename(columns={"index": "period"})
    )

    # Interpolate numeric metrics, but if an entire column is NaN, fallback to 0
    metric_cols = [
        "churn_rate",
        "avg_cart_abandon_rate",
        "avg_website_visits_per_month",
        "avg_num_purchases",
    ]
    for c in metric_cols:
        if agg[c].isna().all():
            agg[c] = 0.0
        else:
            agg[c] = agg[c].interpolate(limit_direction="both")

    agg["volume"] = agg["volume"].fillna(0)

    # Extract series and hard-clean NaNs/Infs
    series = agg[target_metric].astype("float32").values
    series = np.where(np.isfinite(series), series, np.nan)

    # If still NaNs, fill with forward/backward then 0
    if np.isnan(series).any():
        s = pd.Series(series).fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        series = s.values.astype("float32")

    return agg, series

    # Ensure churned numeric
    d["churned"] = pd.to_numeric(d["churned"], errors="coerce")

    agg = (
        d.groupby("period")
        .agg(
            churn_rate=("churned", "mean"),
            avg_cart_abandon_rate=("cart_abandon_rate", "mean"),
            avg_website_visits_per_month=("website_visits_per_month", "mean"),
            avg_num_purchases=("num_purchases", "mean"),
            volume=("customer_id", "count"),
        )
        .reset_index()
        .sort_values("period")
    )

    # Fill missing periods (to avoid gaps)
    full_index = pd.date_range(
        agg["period"].min(), agg["period"].max(), freq=("W" if freq == "W" else "MS")
    )
    agg = (
        agg.set_index("period")
        .reindex(full_index)
        .reset_index()
        .rename(columns={"index": "period"})
    )
    # interpolate smoothly for modeling (or 0 for volume)
    for c in [
        "churn_rate",
        "avg_cart_abandon_rate",
        "avg_website_visits_per_month",
        "avg_num_purchases",
    ]:
        agg[c] = agg[c].interpolate(limit_direction="both")
    agg["volume"] = agg["volume"].fillna(0)

    # Return chosen series plus meta
    return agg, agg[target_metric].values.astype("float32")


def playbook_from_risk(is_risky: bool):
    if is_risky:
        return [
            "Run a use-case workshop (60–90 min) to identify 2–3 high-ROI workflows for the next sprint.",
            "Reinforce enablement: appoint champions, run weekly office hours, and publish templates.",
            "Improve quality: audit failures, refine prompts, and strengthen data/knowledge connectivity.",
            "Set adoption KPIs (weekly usage / success rate / time saved) and review bi-weekly with stakeholders.",
        ]
    return [
        "Scale adoption: replicate best workflows across adjacent teams and standardize templates.",
        "Industrialize: monitoring, governance, and change-management cadence.",
        "Expand ROI: prioritize automations with measurable time/cost savings.",
    ]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Wesype-style Time Series Risk Forecast", layout="wide")
st.title("AI Adoption / Churn Time-Series Forecast (RNN / LSTM) — Streamlit Demo")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (Kaggle schema)", type=["csv"])

    if uploaded is None:
        st.info("Upload your CSV to continue.")
        st.stop()

df = pd.read_csv(uploaded)

required = {
    "customer_id",
    "cart_abandon_rate",
    "website_visits_per_month",
    "num_purchases",
    "churned",
    "last_purchase_date",
}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns for this demo: {sorted(list(missing))}")
    st.stop()

tabs = st.tabs(["EDA", "Modeling", "Risk Dashboard", "Executive Summary"])

# -----------------------------
# EDA TAB
# -----------------------------
with tabs[0]:
    st.header("EDA")

    # Basic churn distribution
    churn = pd.to_numeric(df["churned"], errors="coerce")
    churn_rate = float(churn.dropna().mean()) if churn.notna().any() else float("nan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df)}")
    c2.metric("Churn rate", "N/A" if np.isnan(churn_rate) else f"{churn_rate*100:.1f}%")
    c3.metric(
        "Countries", f"{df['country'].nunique() if 'country' in df.columns else 'N/A'}"
    )

    st.subheader("Quick sanity checks")
    st.write(df.head(10))

    # Boxplot (active only if enough non-null)
    st.subheader("Drivers (boxplot by churn)")
    feat = st.selectbox(
        "Select feature",
        [
            "cart_abandon_rate",
            "website_visits_per_month",
            "num_purchases",
            "annual_income",
            "spending_score",
        ],
        index=0,
    )
    tmp = df[[feat, "churned"]].copy()
    tmp[feat] = pd.to_numeric(tmp[feat], errors="coerce")
    tmp["churned"] = pd.to_numeric(tmp["churned"], errors="coerce")
    tmp = tmp.dropna()

    if len(tmp) < 20:
        st.warning(
            "Not enough valid rows for this boxplot (too many NaNs or parsing issues). Try another feature."
        )
    else:
        fig = plt.figure(figsize=(8, 4), dpi=110)
        tmp.boxplot(column=feat, by="churned")
        plt.title(f"{feat} by churn")
        plt.suptitle("")
        plt.xlabel("churned (0=active, 1=churned)")
        plt.ylabel(feat)
        st.pyplot(fig, clear_figure=True)

    st.subheader("Build the time series (aggregation view)")
    freq = st.radio("Aggregation", ["W", "M"], index=0, help="W=weekly, M=monthly")
    metric = st.selectbox(
        "Target metric to forecast",
        [
            "churn_rate",
            "avg_cart_abandon_rate",
            "avg_website_visits_per_month",
            "avg_num_purchases",
        ],
        index=0,
    )

    agg, series = build_time_series(df, freq=freq, target_metric=metric)
    st.caption(
        f"Time series points: {len(series)} | NaNs: {int(np.isnan(series).sum())}"
    )
    st.write(agg.tail(10))

    fig = plt.figure(figsize=(14, 4), dpi=110)
    plt.plot(agg["period"], agg[metric])
    plt.title(f"Time series: {metric} ({'weekly' if freq=='W' else 'monthly'})")
    plt.xlabel("period")
    plt.ylabel(metric)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# MODELING TAB
# -----------------------------
with tabs[1]:
    st.header("Modeling (RNN / LSTM) — same structure as your notebook")

    left, right = st.columns([1, 1])
    with left:
        freq = st.radio("Aggregation", ["W", "M"], index=0)
        metric = st.selectbox(
            "Target metric",
            [
                "churn_rate",
                "avg_cart_abandon_rate",
                "avg_website_visits_per_month",
                "avg_num_purchases",
            ],
            index=0,
        )
        time_steps = st.slider("time_steps", 4, 24, 12)
        split_percent = st.slider("Train split %", 0.6, 0.9, 0.8, step=0.05)

    with right:
        model_type = st.selectbox("Model type", ["SimpleRNN", "LSTM"], index=1)
        hidden_units = st.slider("hidden_units", 2, 64, 8)
        epochs = st.slider("epochs", 5, 100, 20)
        batch_size = st.selectbox("batch_size", [1, 4, 8, 16, 32], index=0)

    agg, series = build_time_series(df, freq=freq, target_metric=metric)

    if len(series) < time_steps * 3:
        st.error(
            "Not enough periods to train. Try monthly aggregation or reduce time_steps."
        )
        st.stop()

    train_data, test_data, data_scaled, scaler = get_train_test_from_series(
        series, split_percent=split_percent
    )

    trainX, trainY = get_XY(train_data, time_steps)
    testX, testY = get_XY(test_data, time_steps)

    st.caption(
        f"Train blocks: {len(trainY)} | Test blocks: {len(testY)} (each block = {time_steps} periods)"
    )

    if st.button("Train model"):
        tf.random.set_seed(42)

        input_shape = (time_steps, 1)
        if model_type == "SimpleRNN":
            model = create_RNN(
                hidden_units, 1, input_shape, activation=["tanh", "linear"]
            )
        else:
            model = create_LSTM(
                hidden_units, 1, input_shape, activation=["tanh", "linear"]
            )

        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

        model.fit(
            trainX,
            trainY,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[es],
        )

        train_pred = model.predict(trainX, verbose=0).flatten()
        test_pred = model.predict(testX, verbose=0).flatten()

        train_rmse = rmse(trainY, train_pred)
        test_rmse = rmse(testY, test_pred)

        c1, c2 = st.columns(2)
        c1.metric("Train RMSE (scaled)", f"{train_rmse:.4f}")
        c2.metric("Test RMSE (scaled)", f"{test_rmse:.4f}")

        plot_result_streamlit(
            trainY,
            testY,
            train_pred,
            test_pred,
            f"{model_type} — Actual vs Predicted ({metric})",
        )

        # Save artifacts for Risk Dashboard
        st.session_state["ts_artifacts"] = {
            "model": model,
            "scaler": scaler,
            "metric": metric,
            "freq": freq,
            "time_steps": time_steps,
            "split_percent": split_percent,
            "agg": agg,
            "series_scaled": data_scaled,
        }

    else:
        st.info(
            "Click **Train model** to run forecasting (same logic as your notebook)."
        )


# -----------------------------
# RISK DASHBOARD TAB
# -----------------------------
with tabs[2]:
    st.header("Risk Dashboard")

    art = st.session_state.get("ts_artifacts", None)
    if art is None:
        st.info("Train a model first (Modeling tab).")
        st.stop()

    metric = art["metric"]
    agg = art["agg"]
    model = art["model"]
    time_steps = art["time_steps"]

    st.subheader("Latest forecast (next block)")
    risk_threshold = st.slider("Risk threshold (scaled)", 0.0, 1.0, 0.6, step=0.05)

    # Build last input window from full scaled series
    data_scaled = art["series_scaled"]
    if len(data_scaled) < time_steps:
        st.error("Not enough data for last window.")
        st.stop()

    last_window = data_scaled[-time_steps:].reshape(1, time_steps, 1)
    next_pred_scaled = float(model.predict(last_window, verbose=0).flatten()[0])

    c1, c2 = st.columns(2)
    c1.metric("Predicted next value (scaled)", f"{next_pred_scaled:.3f}")
    c2.metric("Risk?", "HIGH" if next_pred_scaled >= risk_threshold else "LOW")

    is_risky = next_pred_scaled >= risk_threshold

    st.subheader("Recommended playbook")
    for i, a in enumerate(playbook_from_risk(is_risky), 1):
        st.write(f"{i}. {a}")

    st.subheader("Context chart")
    fig = plt.figure(figsize=(14, 4), dpi=110)
    plt.plot(agg["period"], agg[metric])
    plt.title(f"History of {metric} (built from last_purchase_date aggregation)")
    plt.xlabel("period")
    plt.ylabel(metric)
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# EXECUTIVE SUMMARY TAB
# -----------------------------
with tabs[3]:
    st.header("Executive Summary")

    art = st.session_state.get("ts_artifacts", None)
    if art is None:
        st.info("Train a model first (Modeling tab).")
        st.stop()

    metric = art["metric"]
    freq = art["freq"]
    time_steps = art["time_steps"]

    st.markdown("### What this demonstrates (Wesype-style)")
    st.write(
        f"""
- Built a **time-series forecasting pipeline** (scaling → train/test split → sequence blocks → RNN/LSTM) inspired by my original RNN/LSTM project.
- Transformed customer-level data into an **adoption/churn signal over time** by aggregating on `last_purchase_date` ({'weekly' if freq=='W' else 'monthly'}).
- Produced a **risk-triggered playbook** turning model outputs into concrete transformation actions (enablement, use-case workshops, quality & data connectivity).
- Parameters: `time_steps={time_steps}`, model family: RNN/LSTM, evaluated with **RMSE** (consistent with the original notebook).
"""
    )

    st.markdown("### How to pitch it to Wesype (1-liner)")
    st.write(
        "“I adapted my RNN/LSTM time-series forecasting project to model adoption/churn signals over time and packaged it into a Streamlit dashboard that triggers concrete AI transformation playbooks.”"
    )
