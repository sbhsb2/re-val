import os, requests, pandas as pd, numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

VALHALLA_URL = "http://localhost:8002"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * 2 * atan2(sqrt(d), sqrt(1-d))

def process_cell_full_stats(row, num_points=50):
    # Parse Polygon
    coords_str = row['geometry_4326'].replace('POLYGON ((', '').replace('))', '')
    pairs = [p.split() for p in coords_str.split(', ')]
    lons, lats = [float(p[0]) for p in pairs], [float(p[1]) for p in pairs]
    
    # 1. Force find 50 points, 50m apart
    points = []
    attempts = 0
    while len(points) < num_points and attempts < 2000:
        lat, lon = np.random.uniform(min(lats), max(lats)), np.random.uniform(min(lons), max(lons))
        if not points or all(haversine(lat, lon, p[0], p[1]) > 0.05 for p in points):
            points.append((lat, lon))
        attempts += 1
    
    print(f"   -> Found {len(points)} valid points in {attempts} attempts.")
    if len(points) < 2: return []

    results = []
    # 2. Sequential trips (1->2, 2->3, etc.)
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        straight_dist = haversine(p1[0], p1[1], p2[0], p2[1])
        
        for mode in ['auto', 'bicycle', 'truck']:
            try:
                payload = {
                    "locations": [{"lat": p1[0], "lon": p1[1]}, {"lat": p2[0], "lon": p2[1]}],
                    "costing": mode, "directions_options": {"units": "kilometers"}
                }
                resp = requests.post(f"{VALHALLA_URL}/route", json=payload, timeout=2)
                if resp.status_code == 200:
                    res = resp.json()['trip']['summary']
                    results.append({
                        "cell_id": row['de_grid_id'],
                        "mode": mode,
                        "speed": res['length'] / (res['time']/3600) if res['time'] > 0 else 0,
                        "cf": res['length'] / straight_dist if straight_dist > 0 else 1
                    })
            except: continue
    return results

def main():
    df = pd.read_csv("data/1000m.csv")
    # Take 10 cells with roads
    test_cells = df[df["road_length_m"] > 0].head(10)
    
    all_trips = []
    for i, (_, row) in enumerate(test_cells.iterrows()):
        print(f"[{i+1}/10] Processing Cell {row['de_grid_id']}...")
        all_trips.extend(process_cell_full_stats(row))

    if all_trips:
        tdf = pd.DataFrame(all_trips)
        
        # 3. Calculate Aggregates per Mode per Cell
        final_stats = tdf.groupby(['cell_id', 'mode']).agg(
            avg_speed_kmh=('speed', 'mean'),
            median_speed_kmh=('speed', 'median'),
            std_speed_kmh=('speed', 'std'),
            avg_cf_mode=('cf', 'mean'),
            median_cf_mode=('cf', 'median'),
            std_cf_mode=('cf', 'std')
        ).reset_index()

        # Merge with original data
        final = final_stats.merge(df[['de_grid_id', 'population', 'road_length_m']], 
                                left_on='cell_id', right_on='de_grid_id')
        
        out_file = "results/speed/advanced_stats_10_cells.csv"
        final.to_csv(out_file, index=False)
        print(f"\n✓ FINISHED! Total rows: {len(final)}")
        print(f"Output File: {out_file}")

if __name__ == "__main__":
    main()