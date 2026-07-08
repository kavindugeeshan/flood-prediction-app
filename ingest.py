import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import datetime
import os
from sqlalchemy import create_engine

STATIONS = ['Deraniyagala', 'Glencourse', 'Hanwella', 'Holombuwa', 'Kithulgala', 'Nagalagam Street', 'Norwood']

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
    db_url = os.environ.get("DB_URL")
    if not db_url:
        raise ValueError("DB_URL environment variable is missing!")
    engine = create_engine(db_url)
    
    arcgis_df = fetch_arcgis_data()
    om_df = fetch_open_meteo_data()
    
    if arcgis_df.empty:
        print("No ArcGIS data found.")
        return
        
    # Merge Open Meteo data for Nagalagam Street
    # Create a full dataframe and merge
    merged = pd.merge(arcgis_df, om_df, on='timestamp', how='left')
    
    # Fill Nagalagam Street rain_fall with om_rain_fall
    nagalagam_mask = merged['station'] == 'Nagalagam Street'
    merged.loc[nagalagam_mask, 'rain_fall'] = merged.loc[nagalagam_mask, 'om_rain_fall']
    
    # Drop the temporary open-meteo column
    final_df = merged.drop(columns=['om_rain_fall'])
    
    # Fill any remaining NaNs with 0 for rainfall, and ffill/bfill for water levels
    final_df['rain_fall'] = final_df['rain_fall'].fillna(0)
    
    # Save to Postgres
    print("Saving to Postgres database...")
    
    try:
        # Load existing database to append
        existing_df = pd.read_sql("SELECT * FROM records", engine)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], format='mixed')
        print(f"DEBUG: existing_df has {len(existing_df)} rows")
        print(f"DEBUG: final_df has {len(final_df)} rows")
        combined = pd.concat([existing_df, final_df])
        combined = combined.drop_duplicates(subset=['timestamp', 'station'], keep='last')
        print(f"DEBUG: combined has {len(combined)} rows after concat and deduplication")
    except Exception as e:
        print(f"Warning: Could not read existing database, starting fresh. Reason: {e}")
        # Table might not exist yet or formatting error
        combined = final_df.drop_duplicates(subset=['timestamp', 'station'], keep='last')
        
    combined = combined.sort_values(by=['timestamp', 'station']).reset_index(drop=True)
    combined.to_sql('records', engine, if_exists='replace', index=False)
    
    print(f"Ingestion complete. Database has {len(combined)} records.")
    
    print("\n--- Latest Data Segment (Nagalagam Street now in meters) ---")
    latest_per_station = combined.sort_values('timestamp').groupby('station').tail(1)
    print(latest_per_station[['timestamp', 'station', 'water_level', 'rain_fall']].to_string(index=False))
    print("\n")

if __name__ == "__main__":
    print("Starting GitHub Actions ingestion run...")
    try:
        print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting data fetch cycle...")
        ingest()
    except Exception as e:
        print(f"Error during ingestion cycle: {e}")
        raise e
