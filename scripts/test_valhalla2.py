"""
Simple Speed & Circuity Analysis (Sequential Version)
======================================================

Calculates speeds and circuity for Bremen grid cells.
Sequential processing - runs one cell at a time (slower but simpler).

Output CSV columns:
- cell_id, cell_lat, cell_lon, cell_population, cell_road_length
- point1_lat, point1_lon, point2_lat, point2_lon
- distance_km, time_seconds, time_minutes, speed_kmh, mode
- straight_dist_km, circuity
- avg_cf_<mode>, median_cf_<mode>, std_cf_<mode>
- avg_speed_<mode>, median_speed_<mode>, std_speed_<mode>

Usage: python simple_speed_analysis.py
"""

import pandas as pd
import numpy as np
import requests
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import os

# Configuration
VALHALLA_URL = "http://localhost:8002"
DATA_DIR = "data"
RESULTS_DIR = "results"

# Sample points per cell (start small for testing!)
SAMPLE_POINTS = 5  # Only 5 points = 20 routes per cell (fast for testing)

# Transport modes to analyze
MODES = ['auto', 'bicycle', 'truck']


def check_valhalla():
    """
    BLOCK 1: Check if Valhalla is running
    
    What it does:
    - Tests connection to Valhalla server
    - Exits if not available
    
    Why: Prevents running analysis if routing engine not ready
    """
    print("Checking Valhalla connection...")
    try:
        response = requests.get(f"{VALHALLA_URL}/status", timeout=5)
        if response.status_code == 200:
            print("✓ Valhalla is running\n")
            return True
    except:
        pass
    
    print("✗ Valhalla not accessible!")
    print("  Start with: docker start valhalla-service")
    print("  Or run: docker-compose up -d")
    return False


def load_grid_data():
    """
    BLOCK 2: Load grid CSV file
    
    What it does:
    - Reads 500m.csv from data/ folder
    - Filters to Bremen city only
    - Filters to cells with roads (road_length_m > 0)
    
    Returns: DataFrame with active grid cells
    
    Why: Need grid cells as starting point for analysis
    
    Note: You'll need to download 500m.csv separately
          (too large for GitHub, see data/README.md)
    """
    print("="*70)
    print("LOADING GRID DATA")
    print("="*70)
    
    filepath = os.path.join(DATA_DIR, "500m.csv")
    
    if not os.path.exists(filepath):
        print(f"\n✗ ERROR: {filepath} not found!")
        print("\nPlease download grid data:")
        print("1. Get 500m.csv from your database")
        print("2. Place it in the data/ folder")
        print("3. Run this script again")
        return None
    
    print(f"\nLoading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✓ Loaded {len(df)} total cells")
    
    # Filter to Bremen
    if 'city_name' in df.columns:
        df = df[df['city_name'] == 'Bremen, Stadt'].copy()
        print(f"✓ Bremen cells: {len(df)}")
    
    # Filter to cells with roads
    df = df[df['road_length_m'] > 0].copy()
    print(f"✓ Active cells (with roads): {len(df)}\n")
    
    return df


def extract_cell_bounds(geometry_str):
    """
    BLOCK 3: Extract cell boundaries from geometry
    
    What it does:
    - Parses "POLYGON ((lon lat, lon lat, ...))" string
    - Extracts min/max coordinates
    - Calculates center point
    
    Input: "POLYGON ((8.8 53.0, 8.9 53.0, ...))"
    Output: Dict with lat_min, lat_max, lon_min, lon_max, center_lat, center_lon
    
    Why: Need cell boundaries to generate random sample points inside
    """
    try:
        # Remove "POLYGON ((" and "))"
        coords_str = geometry_str.replace('POLYGON ((', '').replace('))', '')
        
        # Split into coordinate pairs
        coord_pairs = coords_str.split(', ')
        
        lats = []
        lons = []
        
        for pair in coord_pairs:
            parts = pair.strip().split()
            if len(parts) == 2:
                lon = float(parts[0])
                lat = float(parts[1])
                lons.append(lon)
                lats.append(lat)
        
        if len(lats) == 0:
            return None
        
        return {
            'lat_min': min(lats),
            'lat_max': max(lats),
            'lon_min': min(lons),
            'lon_max': max(lons),
            'center_lat': sum(lats) / len(lats),
            'center_lon': sum(lons) / len(lons)
        }
    except:
        return None


def generate_sample_points(bounds, num_points):
    """
    BLOCK 4: Generate random points within cell
    
    What it does:
    - Creates N random points inside cell boundaries
    - Points are uniformly distributed across cell area
    
    Input: Cell bounds, number of points
    Output: List of points with lat/lon coordinates
    
    Why: Need multiple points to test different routes within cell
    """
    points = []
    
    for i in range(num_points):
        lat = np.random.uniform(bounds['lat_min'], bounds['lat_max'])
        lon = np.random.uniform(bounds['lon_min'], bounds['lon_max'])
        points.append({
            'lat': lat,
            'lon': lon,
            'point_id': i
        })
    
    return points


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    BLOCK 5: Calculate straight-line distance
    
    What it does:
    - Uses haversine formula for Earth's curvature
    - Calculates "as the crow flies" distance
    
    Input: Two points (lat/lon)
    Output: Distance in kilometers
    
    Why: Need straight distance to calculate circuity = route_dist / straight_dist
    """
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def get_route_from_valhalla(lat1, lon1, lat2, lon2, mode):
    """
    BLOCK 6: Get route from Valhalla
    
    What it does:
    - Sends routing request to Valhalla server
    - Gets actual route following roads (not straight line)
    
    Input: Start point, end point, transport mode
    Output: Distance (km) and time (seconds) from Valhalla
    
    Why: Get real-world routing data based on road network
    """
    try:
        response = requests.post(
            f"{VALHALLA_URL}/route",
            json={
                'locations': [
                    {'lat': lat1, 'lon': lon1},
                    {'lat': lat2, 'lon': lon2}
                ],
                'costing': mode,
                'directions_options': {'units': 'kilometers'}
            },
            timeout=10
        )
        
        if response.status_code == 200:
            summary = response.json()['trip']['summary']
            return {
                'distance_km': summary['length'],
                'time_seconds': summary['time']
            }
        else:
            return None
    except:
        return None


def analyze_single_cell(cell, cell_number, total_cells):
    """
    BLOCK 7: Analyze one grid cell
    
    What it does:
    1. Extract cell boundaries
    2. Generate sample points inside cell
    3. Calculate routes between all point pairs (N × (N-1) combinations)
    4. For each route:
       - Get straight distance (haversine)
       - Skip if < 50m
       - For each transport mode (car/bike/truck):
         * Get route from Valhalla
         * Calculate speed = distance / time
         * Calculate circuity = route_distance / straight_distance
    5. Collect all route details
    
    Input: Cell data from DataFrame, cell index
    Output: List of route dictionaries
    
    Why: This is the core analysis - one cell at a time
    """
    print(f"\nCell {cell_number}/{total_cells}: {cell['de_grid_id']}")
    
    # Extract cell geometry
    bounds = extract_cell_bounds(cell['geometry_4326'])
    if bounds is None:
        print("  ✗ Could not parse geometry")
        return []
    
    # Generate sample points
    points = generate_sample_points(bounds, SAMPLE_POINTS)
    print(f"  Generated {len(points)} sample points")
    
    # Storage for this cell's routes
    routes = []
    route_count = 0
    
    # Calculate routes between all point pairs
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:  # Skip same point
                continue
            
            p1 = points[i]
            p2 = points[j]
            
            # Straight-line distance
            straight_dist = haversine_distance(
                p1['lat'], p1['lon'],
                p2['lat'], p2['lon']
            )
            
            # Skip if too short (≤ 50m)
            if straight_dist <= 0.05:  # 0.05 km = 50m
                continue
            
            # Calculate route for each mode
            for mode in MODES:
                route_info = get_route_from_valhalla(
                    p1['lat'], p1['lon'],
                    p2['lat'], p2['lon'],
                    mode
                )
                
                if route_info:
                    distance_km = route_info['distance_km']
                    time_seconds = route_info['time_seconds']
                    
                    # Calculate speed
                    time_hours = time_seconds / 3600
                    speed_kmh = distance_km / time_hours if time_hours > 0 else 0
                    
                    # Only accept realistic speeds (1-100 km/h)
                    if 1 <= speed_kmh <= 100:
                        # Calculate circuity
                        circuity = distance_km / straight_dist if straight_dist > 0 else None
                        
                        # Only accept realistic circuity (1.0-5.0)
                        if circuity and 1.0 <= circuity <= 5.0:
                            # Store route details
                            routes.append({
                                'cell_id': cell['de_grid_id'],
                                'cell_lat': bounds['center_lat'],
                                'cell_lon': bounds['center_lon'],
                                'cell_population': cell.get('population', 0),
                                'cell_road_length': cell['road_length_m'],
                                'point1_lat': p1['lat'],
                                'point1_lon': p1['lon'],
                                'point2_lat': p2['lat'],
                                'point2_lon': p2['lon'],
                                'distance_km': distance_km,
                                'time_seconds': time_seconds,
                                'time_minutes': time_seconds / 60,
                                'speed_kmh': speed_kmh,
                                'mode': mode,
                                'straight_dist_km': straight_dist,
                                'circuity': circuity
                            })
                            
                            route_count += 1
    
    print(f"  ✓ Calculated {route_count} valid routes")
    
    return routes


def calculate_cell_statistics(routes_df):
    """
    BLOCK 8: Calculate statistics per cell
    
    What it does:
    - Groups all routes by cell_id
    - For each cell, calculates:
      * Average, median, std for speed (per mode)
      * Average, median, std for circuity (per mode)
    
    Input: DataFrame with all detailed routes
    Output: DataFrame with cell-level statistics
    
    Why: Need aggregated metrics per cell for analysis
    """
    print("\nCalculating cell-level statistics...")
    
    summaries = []
    
    for cell_id in routes_df['cell_id'].unique():
        cell_routes = routes_df[routes_df['cell_id'] == cell_id]
        
        summary = {
            'cell_id': cell_id,
            'cell_lat': cell_routes['cell_lat'].iloc[0],
            'cell_lon': cell_routes['cell_lon'].iloc[0],
            'cell_population': cell_routes['cell_population'].iloc[0],
            'cell_road_length': cell_routes['cell_road_length'].iloc[0]
        }
        
        for mode in MODES:
            mode_routes = cell_routes[cell_routes['mode'] == mode]
            
            if len(mode_routes) > 0:
                # Speed statistics
                summary[f'avg_speed_{mode}'] = mode_routes['speed_kmh'].mean()
                summary[f'median_speed_{mode}'] = mode_routes['speed_kmh'].median()
                summary[f'std_speed_{mode}'] = mode_routes['speed_kmh'].std()
                
                # Circuity statistics
                summary[f'avg_cf_{mode}'] = mode_routes['circuity'].mean()
                summary[f'median_cf_{mode}'] = mode_routes['circuity'].median()
                summary[f'std_cf_{mode}'] = mode_routes['circuity'].std()
                
                summary[f'num_routes_{mode}'] = len(mode_routes)
            else:
                # No routes for this mode
                summary[f'avg_speed_{mode}'] = np.nan
                summary[f'median_speed_{mode}'] = np.nan
                summary[f'std_speed_{mode}'] = np.nan
                summary[f'avg_cf_{mode}'] = np.nan
                summary[f'median_cf_{mode}'] = np.nan
                summary[f'std_cf_{mode}'] = np.nan
                summary[f'num_routes_{mode}'] = 0
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


def save_results(routes_df, summary_df):
    """
    BLOCK 9: Save results to CSV
    
    What it does:
    - Creates results/ directory if needed
    - Saves detailed routes (every route calculated)
    - Saves cell summaries (statistics per cell)
    - Adds timestamp to filenames
    
    Why: Preserve analysis results for further use
    """
    print("\nSaving results...")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed routes
    routes_file = os.path.join(RESULTS_DIR, f"detailed_routes_{timestamp}.csv")
    routes_df.to_csv(routes_file, index=False)
    print(f"✓ Detailed routes: {routes_file}")
    print(f"  ({len(routes_df)} routes)")
    
    # Cell summaries
    summary_file = os.path.join(RESULTS_DIR, f"cell_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Cell summary: {summary_file}")
    print(f"  ({len(summary_df)} cells)")
    
    return routes_file, summary_file


def main():
    """
    BLOCK 10: Main execution
    
    What it does:
    1. Check Valhalla is running
    2. Load grid data
    3. For each cell (sequential, one at a time):
       - Analyze the cell
       - Collect route details
    4. Calculate cell statistics
    5. Save results to CSV
    
    Why: Orchestrates the entire analysis workflow
    """
    print("="*70)
    print("SIMPLE SPEED & CIRCUITY ANALYSIS (Sequential)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Sample points per cell: {SAMPLE_POINTS}")
    print(f"  Transport modes: {', '.join(MODES)}")
    print(f"  Processing: Sequential (one cell at a time)")
    
    # Step 1: Check Valhalla
    if not check_valhalla():
        return
    
    # Step 2: Load data
    df = load_grid_data()
    if df is None or len(df) == 0:
        return
    
    # For testing, analyze only first 10 cells
    print(f"\n{'='*70}")
    print(f"ANALYZING CELLS (Testing with first 10 cells)")
    print('='*70)
    
    df_test = df.head(10)  # Only first 10 cells for quick test!
    print(f"\nProcessing {len(df_test)} cells...")
    print(f"(Full analysis would process {len(df)} cells)")
    
    # Step 3: Analyze each cell sequentially
    all_routes = []
    
    for idx, (_, cell) in enumerate(df_test.iterrows(), 1):
        cell_routes = analyze_single_cell(cell, idx, len(df_test))
        all_routes.extend(cell_routes)
    
    if len(all_routes) == 0:
        print("\n✗ No routes calculated!")
        return
    
    # Step 4: Convert to DataFrame
    routes_df = pd.DataFrame(all_routes)
    
    print(f"\n{'='*70}")
    print(f"ROUTE CALCULATION COMPLETE")
    print('='*70)
    print(f"Total routes: {len(routes_df)}")
    
    # Step 5: Calculate statistics
    summary_df = calculate_cell_statistics(routes_df)
    
    # Step 6: Save results
    save_results(routes_df, summary_df)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print('='*70)
    print("\nNext steps:")
    print("1. Check results/ folder for CSV files")
    print("2. To analyze ALL cells, remove the .head(10) limit in the code")
    print("3. To use more sample points, increase SAMPLE_POINTS variable")


if __name__ == '__main__':
    main()