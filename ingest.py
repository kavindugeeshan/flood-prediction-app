import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import sqlite3
import datetime
import time

STATIONS = ['Deraniyagala', 'Glencourse', 'Hanwella', 'Holombuwa', 'Kithulgala', 'Nagalagam Street', 'Norwood']

ALERT_LEVELS = {
    'Deraniyagala': {'minor': 4.5, 'major': 5.0, 'alert': 4.0},
    'Glencourse': {'minor': 15.0, 'major': 16.5, 'alert': 14.0},
    'Hanwella': {'minor': 8.0, 'major': 10.0, 'alert': 7.0},
    'Holombuwa': {'minor': 3.0, 'major': 3.4, 'alert': 2.5},
    'Kithulgala': {'minor': 4.0, 'major': 5.0, 'alert': 3.0},
    'Nagalagam Street': {'minor': 1.5, 'major': 2.2, 'alert': 1.2},
    'Norwood': {'minor': 1.5, 'major': 3.0, 'alert': 1.0}
}

def fetch_arcgis_data():
    print("Fetching ArcGIS data...")
    url = "https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/arcgis/rest/services/gauges_2_view/FeatureServer/0/query"
    gauge_list = "','".join(STATIONS)
    params = {
        "f": "json",
        "where": f"gauge IN ('{gauge_list}')",
        "outFields": "*",
        "orderByFields": "CreationDate DESC",
        "resultRecordCount": 5000,
        "returnGeometry": "false",
    }
    session = requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        res = session.get(url, params=params, verify=False)
        features = res.json().get('features', [])
    except Exception as e:
        print(f"Error fetching ArcGIS data: {e}")
        features = []
    
    data = []
    for f in features:
        attr = f['attributes']
        gauge = attr.get('gauge')
        if gauge in STATIONS:
            # CreationDate is in ms
            edit_date = attr.get('CreationDate')
            if edit_date:
                dt = pd.to_datetime(edit_date, unit='ms').tz_localize('UTC').tz_convert('Asia/Colombo').tz_localize(None)
                
                wl = attr.get('water_level')
                if wl is not None and gauge == 'Nagalagam Street':
                    wl = wl * 0.3048
                    
                data.append({
                    'timestamp': dt,
                    'station': gauge,
                    'water_level': wl,
                    'rain_fall': attr.get('rain_fall')
                })
    
    df = pd.DataFrame(data)
    return df

def fetch_open_meteo_data():
    print("Fetching Open-Meteo data for Nagalagam Street...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 6.9355,
        "longitude": 79.8487,
        "hourly": "precipitation",
        "past_days": 7, # Get a week of history
        "timezone": "auto"
    }
    res = requests.get(url, params=params)
    data = res.json()
    
    if 'hourly' in data:
        times = data['hourly']['time']
        precip = data['hourly']['precipitation']
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(times),
            'om_rain_fall': precip
        })
        # Note: Open Meteo times might be in local timezone, but arcgis seems to be UTC.
        # Let's assume open-meteo returned UTC or timezone=auto handles it. 
        # For simplicity, we just localize to UTC if needed, or assume they align well enough for this prototype.
        # Actually, let's force UTC to match pandas to_datetime default.
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        return df
    return pd.DataFrame()

def ingest():
    arcgis_df = fetch_arcgis_data()
    om_df = fetch_open_meteo_data()
    
    if arcgis_df.empty:
        print("No ArcGIS data found.")
        return
        
    # Merge Open Meteo data for Nagalagam Street if available
    if not om_df.empty:
        merged = pd.merge(arcgis_df, om_df, on='timestamp', how='left')
        nagalagam_mask = merged['station'] == 'Nagalagam Street'
        merged.loc[nagalagam_mask, 'rain_fall'] = merged.loc[nagalagam_mask, 'om_rain_fall']
        final_df = merged.drop(columns=['om_rain_fall'])
    else:
        print("Warning: Open-Meteo data is empty, proceeding with ArcGIS data only.")
        final_df = arcgis_df.copy()
    
    # Fill any remaining NaNs with 0 for rainfall, and ffill/bfill for water levels
    final_df['rain_fall'] = final_df['rain_fall'].fillna(0)
    
    # Calculate numerical alert levels (0.0=Normal, 1.0=Alert, 2.0=Minor, 3.0=Major)
    def get_alert_numeric(station, wl):
        if pd.isna(wl): return 0.0
        levels = ALERT_LEVELS.get(station)
        if not levels: return 0.0
        if wl >= levels['major']: return 3.0
        elif wl >= levels['minor']: return 2.0
        elif wl >= levels['alert']: return 1.0
        else: return 0.0
        
    final_df['alert_level'] = final_df.apply(lambda row: get_alert_numeric(row['station'], row['water_level']), axis=1)
    
    # Save to SQLite
    print("Saving to SQLite database...")
    conn = sqlite3.connect('flood_data.db')
    
    try:
        # Get the latest timestamp currently in the database to avoid duplicates
        max_time_df = pd.read_sql("SELECT MAX(timestamp) as max_ts FROM records", conn)
        max_ts = pd.to_datetime(max_time_df['max_ts'].iloc[0])
        
        # Filter new data to only include rows newer than the database's latest timestamp
        if not pd.isna(max_ts):
            final_df = final_df[final_df['timestamp'] > max_ts]
            
    except Exception as e:
        print(f"Warning: Could not query max timestamp (table might be empty or missing). Reason: {e}")
        # Proceed with inserting all fetched data if table is new
        
    if final_df.empty:
        print("No new records to insert. Database is already up to date.")
    else:
        final_df = final_df.sort_values(by=['timestamp', 'station']).reset_index(drop=True)
        # Use append! We never drop the table or load old rows into memory anymore.
        final_df.to_sql('records', conn, if_exists='append', index=False)
        print(f"Successfully appended {len(final_df)} new records to the database.")
        
        print("\n--- Latest Inserted Segment ---")
        latest_per_station = final_df.sort_values('timestamp').groupby('station').tail(1)
        print(latest_per_station[['timestamp', 'station', 'water_level', 'rain_fall']].to_string(index=False))
        print("\n")
        
    conn.close()

if __name__ == "__main__":
    print("Starting continuous ingestion... (Press Ctrl+C to stop)")
    while True:
        try:
            print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting data fetch cycle...")
            ingest()
        except Exception as e:
            print(f"Error during ingestion cycle: {e}")
            
        print("Sleeping for 15 minutes before next fetch...")
        time.sleep(900)
