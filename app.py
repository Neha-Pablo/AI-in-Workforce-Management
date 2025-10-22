import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Forecasting & Staffing Planner | Neha Korati", layout="wide")

st.title("AI Forecasting & Staffing Planner")
st.caption("Built by Neha Korati | MBA, Operations & Supply Chain | AI in Workforce Management")
st.markdown("Upload your call volume dataset, generate forecasts, and compute staffing plans interactively.")

# ------------- Robust upload: main uploader + fallback button -------------
uploaded = st.file_uploader("Upload CSV (any schema)", type=["csv"], accept_multiple_files=False)
use_sample = st.button("Or click to load a sample dataset")

# 1) LOAD DATA (no assumptions yet)
raw_df = None
source_note = ""
if use_sample:
    rng = pd.date_range("2024-01-01 06:00", "2024-04-30 22:00", freq="30min")
    sdf = pd.DataFrame({"timestamp": rng})
    dow = sdf.timestamp.dt.dayofweek
    hour = sdf.timestamp.dt.hour + sdf.timestamp.dt.minute/60
    dow_factor = np.select([dow<=3, dow==4, dow==5, dow==6], [1.15, 1.00, 0.85, 0.80])
    intraday = (0.6*np.exp(-0.5*((hour-9)/1.8)**2)
                + 1.0*np.exp(-0.5*((hour-13)/2)**2)
                + 0.5*np.exp(-0.5*((hour-18)/2.5)**2))
    trend = 1 + 0.0004*np.arange(len(sdf))
    np.random.seed(42)
    noise = np.random.normal(0, 3, len(sdf))
    spikes = (np.random.rand(len(sdf)) < 0.002) * np.random.randint(20, 60, len(sdf))
    base = 35
    sdf["volume"] = np.clip(base*dow_factor*intraday*trend + noise + spikes, 0, None).round().astype(int)
    raw_df = sdf.copy()
    source_note = "Loaded sample data."
elif uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)  # don't parse yet; we’ll map columns first
        source_note = "CSV uploaded."
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        raw_df = None

if raw_df is None:
    st.info("Upload a CSV (any column names) or click the Sample button.")
    st.stop()

st.success(source_note)

# 2) COLUMN MAPPING UI
st.markdown("### Column mapping")
cols = list(raw_df.columns)

# simple guesses
lower = [c.lower() for c in cols]
guess_ts = None
for key in ["timestamp", "time", "date", "datetime"]:
    guess_ts = next((c for c in cols if key in c.lower()), guess_ts)
guess_vol = next((c for c in cols if any(k in c.lower() for k in ["volume","calls","contacts","count"])), None)

timestamp_col = st.selectbox("Select the Timestamp column", options=cols, index=cols.index(guess_ts) if guess_ts in cols else 0)
volume_col    = st.selectbox("Select the Volume column", options=cols, index=cols.index(guess_vol) if guess_vol in cols else 0)

# optional date format help
with st.expander("⏱️ (Optional) Specify date format if parsing fails"):
    st.markdown("Common examples: `%%Y-%%m-%%d %%H:%%M:%%S`, `%%m/%%d/%%Y %%I:%%M %%p`")
    date_fmt = st.text_input("Date/time format (leave blank to auto-detect)", value="")

# 3) BUILD STANDARDIZED DF (timestamp, volume) + FRIENDLY ERRORS
df = raw_df[[timestamp_col, volume_col]].rename(columns={timestamp_col: "timestamp", volume_col: "volume"}).copy()

# parse dates
try:
    if date_fmt.strip():
        df["timestamp"] = pd.to_datetime(df["timestamp"], format=date_fmt, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True)
except Exception as e:
    st.error(f"Could not parse dates: {e}")
    st.stop()

# coerce volume to numeric
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

# basic validations
issues = []
if df["timestamp"].isna().any():
    n = int(df["timestamp"].isna().sum())
    issues.append(f"{n} rows have invalid timestamps.")
if df["volume"].isna().any():
    n = int(df["volume"].isna().sum())
    issues.append(f"{n} rows have non-numeric or missing volume.")
if (df["volume"] < 0).any():
    n = int((df["volume"] < 0).sum())
    issues.append(f"{n} rows have negative volume (not allowed).")
# drop bad rows for now
df = df.dropna(subset=["timestamp","volume"])
df = df[df["volume"] >= 0]

if len(issues) > 0:
    st.warning("Data checks:\n- " + "\n- ".join(issues))

if df.empty:
    st.error("After cleaning, no usable rows remain. Please adjust column mapping or date format.")
    st.stop()

# sort & preview
df = df.sort_values("timestamp").reset_index(drop=True)
st.dataframe(df.head(10), use_container_width=True)
st.caption("Mapped preview (first 10 rows). Columns standardized to `timestamp` and `volume`.")


if df is None:
    st.info("Upload a CSV with columns 'timestamp' and 'volume' or click the sample button.")
    st.stop()

# ------------- Prep and features -------------
df = df.sort_values("timestamp").reset_index(drop=True)
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["lag_1day"] = df["volume"].shift(48)
df["lag_1week"] = df["volume"].shift(48*7)
work = df.dropna().reset_index(drop=True)

# Sidebar controls
st.sidebar.header("Model Parameters")
aht = st.sidebar.number_input("Average Handle Time (seconds)", 60, 900, 300, 10)
occupancy = st.sidebar.slider("Occupancy", 0.5, 0.95, 0.85, 0.01)
shrinkage = st.sidebar.slider("Shrinkage", 0.0, 0.5, 0.30, 0.01)

# ------------- Train test split -------------
split_idx = int(len(work)*0.8)
train, test = work.iloc[:split_idx], work.iloc[split_idx:]

# ------------- Baseline and AI forecasts -------------
test["forecast_baseline"] = test["lag_1week"]

features = ["hour", "dayofweek", "is_weekend", "lag_1day", "lag_1week"]
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(train[features], train["volume"])
test["forecast_rf"] = rf.predict(test[features])

mae_base = mean_absolute_error(test["volume"], test["forecast_baseline"])
mae_rf = mean_absolute_error(test["volume"], test["forecast_rf"])
improvement = (1 - mae_rf/mae_base) * 100

st.sidebar.subheader("Model Performance")
st.sidebar.write(f"Baseline MAE: {mae_base:.2f}")
st.sidebar.write(f"AI Model MAE: {mae_rf:.2f}")
st.sidebar.write(f"Improvement: {improvement:.2f}%")

choice = st.radio("Select Forecast Type", ["Baseline", "AI Model"], horizontal=True)
forecast_col = "forecast_baseline" if choice == "Baseline" else "forecast_rf"

# ------------- Staffing function -------------
def staffing_from_volume(vol, aht_seconds, interval_minutes=30, occupancy=0.85, shrinkage=0.3):
    calls = np.array(vol)
    interval_seconds = interval_minutes * 60
    workload_hours = calls * aht_seconds / 3600.0
    available_hours = interval_seconds / 3600.0
    fte_required = workload_hours / (occupancy * (1 - shrinkage) * available_hours)
    return fte_required

test["fte_required"] = staffing_from_volume(test[forecast_col], aht, 30, occupancy, shrinkage)

# ------------- EDA section from Colab, with clear colors -------------
st.markdown("### Exploratory Charts")

# 1) Daily trend
daily = df.set_index("timestamp")["volume"].resample("D").sum()
fig1, ax1 = plt.subplots(figsize=(10, 3))
ax1.plot(daily.index, daily.values, color="tab:blue")
ax1.set_title("Daily Call Volume Trend")
ax1.set_xlabel("Date"); ax1.set_ylabel("Calls")
st.pyplot(fig1)

# 2) Intraday hourly pattern
hourly = df.groupby("hour")["volume"].mean()
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(hourly.index, hourly.values, marker="o", color="tab:orange")
ax2.set_title("Average Intraday Pattern"); ax2.set_xlabel("Hour"); ax2.set_ylabel("Avg Calls")
st.pyplot(fig2)

# 3) Weekday pattern
wd = df.groupby("dayofweek")["volume"].mean()
fig3, ax3 = plt.subplots(figsize=(6, 3))
ax3.bar(wd.index, wd.values, color="tab:green")
ax3.set_title("Average Volume by Day of Week (0=Mon)")
ax3.set_xlabel("Day"); ax3.set_ylabel("Avg Calls")
st.pyplot(fig3)

# ------------- Forecast vs actual with contrasting colors -------------
st.markdown("### Forecast vs Actual")
fig4, ax4 = plt.subplots(figsize=(12, 3.5))
ax4.plot(test["timestamp"], test["volume"], label="Actual", color="tab:blue")
ax4.plot(test["timestamp"], test[forecast_col], label="Forecast", color="tab:orange")
ax4.set_xlabel("Time"); ax4.set_ylabel("Calls")
ax4.legend(loc="upper right")
st.pyplot(fig4)

# ------------- FTE requirement with a distinct color -------------
st.markdown("### FTE Requirement")
fig5, ax5 = plt.subplots(figsize=(12, 3.5))
ax5.plot(test["timestamp"], test["fte_required"], label="FTE Required", color="tab:purple")
ax5.set_xlabel("Time"); ax5.set_ylabel("FTE")
ax5.legend(loc="upper right")
st.pyplot(fig5)

# ------------- Download -------------
st.markdown("### Download Staffing Plan")
out = test[["timestamp", forecast_col, "fte_required"]].rename(columns={forecast_col: "forecast"})
csv = out.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "staffing_plan.csv", "text/csv")

# ------------- Help -------------
with st.expander("About this App and quick tips"):
    st.markdown(
        "If drag and drop does not respond, click Browse files. "
        "You can also use the Sample button to load demo data."
    )
