import os, requests, pandas as pd, numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from scipy import stats

VALHALLA_URL = "http://localhost:8002"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def process_complex_cell(row, num_samples=10): # Set to 10 for the quick test
    try:
        # 1. Parse Geometry
        coords_str = row['geometry_4326'].replace('POLYGON ((', '').replace('))', '')
        pairs = [p.split() for p in coords_str.split(', ')]
        lons, lats = [float(p[0]) for p in pairs], [float(p[1]) for p in pairs]
        
        cell_results = []
        
        # 2. Generate Points (50m apart logic)
        points = []
        attempts = 0
        while len(points) < 2 and attempts < 100:
            lat, lon = np.random.uniform(min(lats), max(lats)), np.random.uniform(min(lons), max(lons))
            if all(haversine(lat, lon, p[0], p[1]) > 0.05 for p in points):
                points.append((lat, lon))
            attempts += 1

        if len(points) < 2: return []

        # 3. Perform Trips
        p1, p2 = points[0], points[1]
        straight_dist = haversine(p1[0], p1[1], p2[0], p2[1])
        
        for mode in ['auto', 'bicycle', 'truck']:
            resp = requests.post(f"{VALHALLA_URL}/route", json={
                "locations": [{"lat": p1[0], "lon": p1[1]}, {"lat": p2[0], "lon": p2[1]}],
                "costing": mode, "directions_options": {"units": "kilometers"}
            }, timeout=5)
            
            if resp.status_code == 200:
                res = resp.json()['trip']['summary']
                d, t = res['length'], res['time']
                cell_results.append({
                    "cell_id": row['de_grid_id'],
                    "cell_population": row.get('population', 0),
                    "cell_road_length": row['road_length_m'],
                    "point1_lat": p1[0], "point1_lon": p1[1],
                    "point2_lat": p2[0], "point2_lon": p2[1],
                    "distance_km": d, "time_seconds": t,
                    "speed_kmh": d / (t/3600) if t > 0 else 0,
                    "mode": mode, "straight_dist_km": straight_dist,
                    "circuity": d / straight_dist if straight_dist > 0 else 1
                })
        return cell_results
    except:
        return []

def main():
    df = pd.read_csv("data/1000m.csv")
    test_cells = df[df["road_length_m"] > 0].head(10) # Testing first 10
    
    final_data = []
    print(f"Starting analysis on {len(test_cells)} cells...")
    
    for _, row in test_cells.iterrows():
        res = process_complex_cell(row)
        final_data.extend(res)
        print(f"Cell {row['de_grid_id']} done.")

    if final_data:
        pdf = pd.DataFrame(final_data)
        # Adding simple stats for the mode
        pdf['time_minutes'] = pdf['time_seconds'] / 60
        output_path = "results/speed/advanced_test_10.csv"
        pdf.to_csv(output_path, index=False)
        print(f"✓ Created {output_path} with {len(pdf.columns)} columns.")

if __name__ == "__main__":
    main()