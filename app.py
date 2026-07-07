import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import json
st.set_page_config(page_title="Flash Flood Prediction", layout="wide")

STATIONS = ['Norwood', 'Kithulgala', 'Deraniyagala', 'Holombuwa', 'Glencourse', 'Hanwella', 'Nagalagam Street']

ALERT_LEVELS = {
    'Nagalagam Street': {'alert': 1.20, 'minor': 1.50, 'major': 2.00},
    'Hanwella': {'alert': 7.00, 'minor': 8.00, 'major': 10.00},
    'Glencourse': {'alert': 15.00, 'minor': 16.50, 'major': 19.00},
    'Kithulgala': {'alert': 3.00, 'minor': 4.00, 'major': 6.00},
    'Holombuwa': {'alert': 3.00, 'minor': 3.40, 'major': 5.00},
    'Deraniyagala': {'alert': 4.80, 'minor': 5.80, 'major': 6.40},
    'Norwood': {'alert': 1.50, 'minor': 3.00, 'major': 4.50},
}

STATION_COORDS = {
    'Norwood': {'lat': 6.83944, 'lon': 80.61167},
    'Kithulgala': {'lat': 6.99056, 'lon': 80.41222},
    'Deraniyagala': {'lat': 6.92444, 'lon': 80.33778},
    'Holombuwa': {'lat': 7.18528, 'lon': 80.26472},
    'Glencourse': {'lat': 6.97444, 'lon': 80.18278},
    'Hanwella': {'lat': 6.90944, 'lon': 80.07944},
    'Nagalagam Street': {'lat': 6.95972, 'lon': 79.87694}
}

@st.cache_data
def load_impact_zones():
    if os.path.exists('impact_zones.json'):
        with open('impact_zones.json', 'r') as f:
            return json.load(f)
    return {}

impact_zones_data = load_impact_zones()

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

from sqlalchemy import create_engine
import os

@st.cache_data(ttl=60)
def load_data():
    # 1. Check if running on Streamlit Cloud with Supabase secrets
    if "DB_URL" in st.secrets:
        try:
            db_url = st.secrets["DB_URL"]
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            
            engine = create_engine(db_url)
            df = pd.read_sql("SELECT * FROM records", engine)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            return df
        except Exception as e:
            st.error(f"Supabase connection error: {e}")
            return pd.DataFrame()
            
    # 2. Fallback to Local SQLite
    if not os.path.exists('flood_data.db'):
        try:
            import ingest
            ingest.ingest()
        except Exception as e:
            print(f"Failed to auto-ingest locally: {e}")
            return pd.DataFrame()
            
    if not os.path.exists('flood_data.db'):
        return pd.DataFrame()
        
    conn = sqlite3.connect('flood_data.db')
    df = pd.read_sql("SELECT * FROM records", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    return df

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

# Sidebar Navigation
with st.sidebar:
    page = option_menu(
        menu_title="Navigation",
        options=["Predictions", "Historical Trends", "Advanced Information"],
        icons=["graph-up-arrow", "clock-history", "gear"],
        menu_icon="compass",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
        }
    )
    
    st.markdown("---")
    st.markdown("### Data Sources")
    st.caption(
        """
        **[Sri Lanka Irrigation Dept](http://www.irrigation.gov.lk/):** Real-time gauge water levels and rainfall.\n
        **[Open-Meteo API](https://open-meteo.com/):** Supplementary meteorological precipitation data.\n
        **[OpenStreetMap](https://www.openstreetmap.org/):** Basemaps and topographical boundaries.
        """
    )

if page == "Predictions":
    # Get target names
    stations_short = ['Deraniyagala', 'Glencourse', 'Hanwella', 'Holombuwa', 'Kithulgala', 'Nagalagam Street', 'Norwood']
    targets_3h = [f'Target_{s}_WL_3h' for s in stations_short]
    targets_6h = [f'Target_{s}_WL_6h' for s in stations_short]
    targets_12h = [f'Target_{s}_WL_12h' for s in stations_short]
    all_targets = targets_3h + targets_6h + targets_12h
    
    pred_df = pd.DataFrame(preds, columns=all_targets, index=[0])
    
    st.markdown("---")
    st.subheader("Predictions & Alerts")
    
    selected_horizon = st.radio("Select Prediction Horizon:", ["+3 Hours", "+6 Hours", "+12 Hours"], horizontal=True)
    
    recent_2_raw = df_raw.sort_values('timestamp').groupby('station').tail(2)
    
    menu_col, card_col = st.columns([1, 3])
    
    with menu_col:
        st.markdown("### Stations")
        selected_station_pred = option_menu(
            menu_title=None,
            options=STATIONS,
            icons=['geo-alt']*len(STATIONS),
            default_index=0,
            styles={"container": {"padding": "0!important"}}
        )
        
    with card_col:
        station = selected_station_pred
        s_short = station
        
        # Get trend, current reading, and rainfall
        station_data = recent_2_raw[recent_2_raw['station'] == station]
        water_levels = station_data['water_level'].values
        rain_falls = station_data['rain_fall'].values
        
        if len(water_levels) > 0:
            current_wl = water_levels[-1]
            current_rf = rain_falls[-1]
        else:
            current_wl = 0.0
            current_rf = 0.0
            
        if len(water_levels) >= 2:
            trend_diff = current_wl - water_levels[0]
        else:
            trend_diff = 0.0
            
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
        
        with st.container(border=True):
            title_col, thresh_col = st.columns([1, 1])
            with title_col:
                st.markdown(f"## {station}")
            with thresh_col:
                st.markdown(f"""
                <div style='text-align: right; padding-top: 25px;'>
                    <span style='background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold;'>🔔 Alert: {levels['alert']}m</span>
                    <span style='background-color: #fd7e14; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; margin-left: 8px;'>⚠️ Minor: {levels['minor']}m</span>
                    <span style='background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; margin-left: 8px;'>🚨 Major: {levels['major']}m</span>
                </div>
                """, unsafe_allow_html=True)
            st.write("---")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Current Level", f"{current_wl:.2f} m", delta=f"{trend_diff:.2f} m", delta_color="inverse")
                st.markdown(f"**{curr_alert}**")
            
            with m_col2:
                pred_diff = pred_val - current_wl
                st.metric(f"Predicted ({selected_horizon})", f"{pred_val:.2f} m", delta=f"{pred_diff:.2f} m", delta_color="inverse")
                st.markdown(f"**{pred_alert}**")
                
            with m_col3:
                st.metric("Hourly Rainfall", f"{current_rf:.1f} mm")
                if current_rf == 0:
                    st.markdown("**✅ DRY**")
                elif current_rf < 5.0:
                    st.markdown("**🌧️ LIGHT RAIN**")
                else:
                    st.markdown("**☔ HEAVY RAIN**")
            
            st.write("---")
            
            station_impacts = impact_zones_data.get(station, {}).get("impact_zones", {})
            
            st.markdown("### ⚠️ Dynamic Impact Assessment")
            if pred_alert == "✅ NORMAL":
                st.success("No areas are expected to be impacted based on current predictions.")
            else:
                st.warning("The following areas are expected to be impacted based on the predicted water level:")
                
                cum_zones = []
                if pred_alert in ["🔔 ALERT", "⚠️ MINOR FLOOD", "🚨 MAJOR FLOOD"]:
                    cum_zones.extend(station_impacts.get("near_alert", []))
                if pred_alert in ["⚠️ MINOR FLOOD", "🚨 MAJOR FLOOD"]:
                    cum_zones.extend(station_impacts.get("near_minor", []))
                if pred_alert == "🚨 MAJOR FLOOD":
                    cum_zones.extend(station_impacts.get("near_major", []))
                    
                for zone in cum_zones:
                    st.write(f"- {zone}")
                    
            if station_impacts:
                st.write("")
                with st.expander("📍 Reference: All Possible Inundation Areas"):
                    if station_impacts.get("near_alert"):
                        st.markdown("**Alert Level**")
                        for zone in station_impacts["near_alert"]: st.write(f"- {zone}")
                    if station_impacts.get("near_minor"):
                        st.markdown("**Minor Flood**")
                        for zone in station_impacts["near_minor"]: st.write(f"- {zone}")
                    if station_impacts.get("near_major"):
                        st.markdown("**Major Flood**")
                        for zone in station_impacts["near_major"]: st.write(f"- {zone}")
                        
            st.write("---")
            st.markdown("### 🗺️ River Basin Context")
            
            all_coords = []
            for s in STATIONS:
                c = STATION_COORDS[s].copy()
                c['station'] = s
                c['size'] = 18 if s == station else 8
                c['color'] = "red" if s == station else "blue"
                all_coords.append(c)
                
            map_data_all = pd.DataFrame(all_coords)
            
            # Plot all stations
            fig_map = px.scatter_mapbox(
                map_data_all, 
                lat="lat", 
                lon="lon",
                zoom=9, 
                height=350
            )
            
            fig_map.data[0].update(
                mode='markers+text',
                marker=dict(
                    size=map_data_all['size'],
                    color=map_data_all['color'],
                    opacity=1.0
                ),
                text=[s if s == station else "" for s in map_data_all['station']],
                textposition="bottom right",
                hoverinfo='text',
                hovertext=map_data_all['station']
            )
            
            # Add basin GeoJSON layer
            try:
                with open('kelani_basin.geojson', 'r') as f:
                    basin_geojson = json.load(f)
                
                fig_map.update_layout(
                    mapbox_layers=[
                        {
                            "sourcetype": "geojson",
                            "source": basin_geojson,
                            "type": "fill",
                            "color": "rgba(0, 100, 255, 0.1)"
                        },
                        {
                            "sourcetype": "geojson",
                            "source": basin_geojson,
                            "type": "line",
                            "color": "rgba(0, 100, 255, 0.8)",
                            "line": {"width": 2}
                        }
                    ]
                )
            except Exception as e:
                pass
            
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0,"t":0,"l":0,"b":0},
                showlegend=False,
                mapbox_center={"lat": 6.95, "lon": 80.20} # Center on basin
            )
            st.plotly_chart(fig_map, use_container_width=True)

elif page == "Historical Trends":
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

elif page == "Advanced Information":
    st.markdown("---")
    st.subheader("⚙️ Advanced Information")
    st.write("### Model Performance Metrics")
    try:
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        st.write(f"- **Last Trained:** {metadata.get('last_trained', 'N/A')}")
        st.write(f"- **Overall RMSE (Root Mean Squared Error):** {metadata.get('rmse', 'N/A')}")
        st.write(f"- **Overall MAE (Mean Absolute Error):** {metadata.get('mae', 'N/A')}")
    except Exception as e:
        st.write("Model metadata not available.")

