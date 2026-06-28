import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from sqlalchemy import create_engine
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
import os
import json
st.set_page_config(page_title="Flash Flood Prediction", layout="wide")

STATIONS = ['Deraniyagala', 'Glencourse', 'Hanwella', 'Holombuwa', 'Kithulgala', 'Nagalagam Street', 'Norwood']

ALERT_LEVELS = {
    'Nagalagam Street': {'alert': 1.20, 'minor': 1.50, 'major': 2.00},
    'Hanwella': {'alert': 7.00, 'minor': 8.00, 'major': 10.00},
    'Glencourse': {'alert': 15.00, 'minor': 16.50, 'major': 19.00},
    'Kithulgala': {'alert': 3.00, 'minor': 4.00, 'major': 6.00},
    'Holombuwa': {'alert': 3.00, 'minor': 3.40, 'major': 5.00},
    'Deraniyagala': {'alert': 4.80, 'minor': 5.80, 'major': 6.40},
    'Norwood': {'alert': 1.50, 'minor': 3.00, 'major': 4.50},
}

def get_alert_numeric(station, water_level):
    if pd.isna(water_level): return 0.0
    # Handle the difference in station name format if any (Nagalagam vs Nagalagam Street)
    key = station if station in ALERT_LEVELS else f"{station} Street"
    if key not in ALERT_LEVELS: key = station.replace("Nagalagam", "Nagalagam Street")
    
    levels = ALERT_LEVELS[key]
    if water_level >= levels['major']: return 2.0
    if water_level >= levels['minor']: return 1.0
    if water_level >= levels['alert']: return 0.5
    return 0.0

@st.cache_data(ttl=60)
def load_data():
    try:
        db_url = st.secrets["DB_URL"]
        engine = create_engine(db_url)
        df = pd.read_sql("SELECT * FROM records", engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        return df
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame()

def prepare_features(df):
    if df.empty: return None, None
    
    # Pivot to wide format
    df_wide = df.pivot_table(index='timestamp', columns='station', values=['water_level', 'rain_fall'])
    
    # Flatten multi-index columns
    df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]
    
    # Resample to 3H intervals
    df_3h = df_wide.resample('3h').mean()
    
    # Interpolate missing values slightly
    df_3h = df_3h.interpolate(method='linear', limit=2).ffill().bfill()
    
    # Rename columns to match training data
    rename_map = {}
    for s in STATIONS:
        # In training data, Nagalagam Street is sometimes 'Nagalagam Street' or 'Nagalagam'
        # Looking at feature_cols, we need to match them.
        s_name = s
        rename_map[f"water_level_{s}"] = f"current_water_level_{s_name}"
        rename_map[f"rain_fall_{s}"] = f"rainfall_mm_{s_name}"
    
    df_3h = df_3h.rename(columns=rename_map)
    
    # Calculate alert_numeric
    for s in STATIONS:
        df_3h[f"alert_numeric_{s}"] = df_3h[f"current_water_level_{s}"].apply(lambda x: get_alert_numeric(s, x))
    
    # Calculate lags
    lags = [3, 6, 12, 24] # in hours
    for lag in lags:
        periods = lag // 3
        for col in df_3h.columns:
            if 'lag' not in col: # avoid lagging already lagged cols
                df_3h[f"{col}_lag_{lag}h"] = df_3h[col].shift(periods)
                
    # Drop rows with NaNs caused by lagging
    df_features = df_3h.dropna()
    return df_3h, df_features

st.title("🌊 Real-Time Flash Flood Prediction System")

df_raw = load_data()

if df_raw.empty:
    st.warning("No data found. Please run `python ingest.py` to populate the database.")
    st.stop()

df_3h, df_features = prepare_features(df_raw)

if df_features is None or df_features.empty:
    st.warning("Not enough data to generate 24h lag features. Please ensure the database has at least 24 hours of data.")
    st.stop()

try:
    feature_cols = joblib.load('models/feature_cols.pkl')
    model = joblib.load('models/unified_model.pkl')
except Exception as e:
    st.error(f"Could not load models. Did you run `python train_model.py`? Error: {e}")
    st.stop()

# Get the latest row for prediction
latest_row = df_features.iloc[[-1]]
latest_time = latest_row.index[0]

# Ensure columns match EXACTLY
for col in feature_cols:
    if col not in latest_row.columns:
        latest_row[col] = 0
X_pred = latest_row[feature_cols]
preds = model.predict(X_pred)

latest_raw_time = df_raw['timestamp'].max()

st.write(f"### Latest Data Timestamp : **{latest_raw_time.strftime('%Y-%m-%d %H:%M')}**")
st.write(f"*(Predictions are based on the latest 3-hour resampled window starting at: {latest_time.strftime('%Y-%m-%d %H:%M')})*")

# Get target names
stations_short = ['Deraniyagala', 'Glencourse', 'Hanwella', 'Holombuwa', 'Kithulgala', 'Nagalagam Street', 'Norwood']
targets_3h = [f'Target_{s}_WL_3h' for s in stations_short]
targets_6h = [f'Target_{s}_WL_6h' for s in stations_short]
targets_12h = [f'Target_{s}_WL_12h' for s in stations_short]
all_targets = targets_3h + targets_6h + targets_12h

pred_df = pd.DataFrame(preds, columns=all_targets, index=[0])

st.markdown("---")
st.subheader("🔮 Predictions & Alerts")

selected_horizon = st.radio("Select Prediction Horizon:", ["+3 Hours", "+6 Hours", "+12 Hours"], horizontal=True)

cols = st.columns(len(STATIONS))
latest_raw_per_station = df_raw.sort_values('timestamp').groupby('station').tail(1).set_index('station')

for idx, station in enumerate(STATIONS):
    s_short = station
    
    # Get exact newest raw reading for the dashboard display
    if station in latest_raw_per_station.index:
        current_wl = latest_raw_per_station.loc[station, 'water_level']
    else:
        current_wl = 0.0
        
    pred_3h = pred_df[f'Target_{s_short}_WL_3h'].values[0]
    pred_6h = pred_df[f'Target_{s_short}_WL_6h'].values[0]
    pred_12h = pred_df[f'Target_{s_short}_WL_12h'].values[0]
    
    if selected_horizon == "+3 Hours":
        pred_val = pred_3h
    elif selected_horizon == "+6 Hours":
        pred_val = pred_6h
    else:
        pred_val = pred_12h
        
    levels = ALERT_LEVELS[station]
    
    def get_alert_status(wl, lvls):
        if wl >= lvls['major']: return "🚨 MAJOR FLOOD"
        elif wl >= lvls['minor']: return "⚠️ MINOR FLOOD"
        elif wl >= lvls['alert']: return "🔔 ALERT"
        else: return "✅ NORMAL"
        
    curr_alert = get_alert_status(current_wl, levels)
    pred_alert = get_alert_status(pred_val, levels)
    
    with cols[idx]:
        st.markdown(f"#### {station}")
        st.metric("Current Level", f"{current_wl:.2f} m")
        st.markdown(f"**{curr_alert}**")
        st.write("---")
        st.metric(f"Predicted ({selected_horizon})", f"{pred_val:.2f} m")
        st.markdown(f"**{pred_alert}**")

st.markdown("---")
st.subheader("📈 Historical Trends (Resampled 3H)")

selected_station = st.selectbox("Select Station", STATIONS)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_3h.index, y=df_3h[f"current_water_level_{selected_station}"], name="Water Level", line=dict(color='blue')))
# Add threshold lines
levels = ALERT_LEVELS[selected_station]
fig.add_hline(y=levels['alert'], line_dash="dash", line_color="orange", annotation_text="Alert Level")
fig.add_hline(y=levels['minor'], line_dash="dash", line_color="red", annotation_text="Minor Flood")
fig.add_hline(y=levels['major'], line_dash="dash", line_color="darkred", annotation_text="Major Flood")

fig.update_layout(title=f"{selected_station} Water Level History", yaxis_title="Water Level (m)", height=400)
st.plotly_chart(fig, width='stretch')

st.markdown("---")
with st.expander("⚙️ Advanced Information"):
    st.write("### Model Performance Metrics")
    try:
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        st.write(f"- **Last Trained:** {metadata.get('last_trained', 'N/A')}")
        st.write(f"- **Overall RMSE (Root Mean Squared Error):** {metadata.get('rmse', 'N/A')}")
        st.write(f"- **Overall MAE (Mean Absolute Error):** {metadata.get('mae', 'N/A')}")
    except Exception as e:
        st.write("Model metadata not available.")

