import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime

VALHALLA_URL = "http://localhost:8002"

def get_route_data(p1, p2, mode, cell_id):
    """Helper to ping Valhalla for a specific mode."""
    try:
        payload = {
            "locations": [{"lat": p1[0], "lon": p1[1]}, {"lat": p2[0], "lon": p2[1]}],
            "costing": mode,
            "directions_options": {"units": "kilometers"}
        }
        resp = requests.post(f"{VALHALLA_URL}/route", json=payload, timeout=5)
        if resp.status_code == 200:
            res = resp.json()['trip']['summary']
            return {
                "cell_id": cell_id,
                "mode": mode,
                "distance_km": res['length'],
                "time_sec": res['time'],
                "speed_kmh": (res['length'] / (res['time'] / 3600)) if res['time'] > 0 else 0
            }
    except:
        pass
    return None

def process_cell_basic(row):
    """Extracts points from geometry and tests routes."""
    try:
        # Extract coordinates from the polygon string
        raw_coords = row['geometry_4326'].replace('POLYGON ((', '').replace('))', '')
        pairs = [p.split() for p in raw_coords.split(', ')]
        lons = [float(p[0]) for p in pairs]
        lats = [float(p[1]) for p in pairs]
        
        # Define 2 points: Center and a slight offset
        p1 = (np.mean(lats), np.mean(lons))
        p2 = (np.max(lats), np.max(lons))
        
        results = []
        for mode in ['auto', 'bicycle', 'truck']:
            data = get_route_data(p1, p2, mode, row['de_grid_id'])
            if data:
                results.append(data)
        return results
    except Exception as e:
        print(f"Error processing cell {row.get('de_grid_id')}: {e}")
        return []

def main():
    # 1. Setup folders
    os.makedirs("results/speed", exist_ok=True)
    
    # 2. Load Data
    print("Reading input data...")
    df = pd.read_csv("data/1000m.csv")
    
    # Filter for cells that actually have roads (to save time)
    active_cells = df[df["road_length_m"] > 0]
    print(f"Total cells to process: {len(active_cells)}")

    all_results = []
    
    # 3. Execution Loop
    try:
        for i, (index, row) in enumerate(active_cells.iterrows()):
            cell_results = process_cell_basic(row)
            all_results.extend(cell_results)
            
            # Print progress every 50 cells so the console stays clean
            if (i + 1) % 50 == 0 or (i + 1) == len(active_cells):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {i+1}/{len(active_cells)} cells...")

    except KeyboardInterrupt:
        print("\nStopped by user. Saving partial results...")

    # 4. Save Final Output
    if all_results:
        output_df = pd.DataFrame(all_results)
        filename = f"results/speed/bremen_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        output_df.to_csv(filename, index=False)
        print(f"\n✓ SUCCESS! Results saved to: {filename}")
    else:
        print("\nNo routes were successfully calculated.")

if __name__ == "__main__":
    main()