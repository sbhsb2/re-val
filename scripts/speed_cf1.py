"""
Interactive Speed & Circuity Analysis
======================================

User-friendly version with interactive prompts.
Sequential processing - no parallel complexity.

Features:
- Choose resolution (1000m/500m/250m)
- Choose number of sample points
- Choose number of cells to analyze
- Sample points guaranteed 50m apart
- All N×(N-1) route combinations

Usage: python interactive_analysis.py
"""

import pandas as pd
import numpy as np
import requests
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION SECTION - User inputs go here
# ============================================================================

VALHALLA_URL = "http://localhost:8002"
DATA_DIR = "data"
RESULTS_DIR = "results"
MODES = ['auto', 'bicycle', 'truck']

# These will be set by user input
RESOLUTION = None      # User chooses: '1000m', '500m', or '250m'
SAMPLE_POINTS = None   # User chooses: number of points per cell
NUM_CELLS = None       # User chooses: how many cells to analyze


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def check_valhalla():
    """Check if Valhalla is running."""
    try:
        response = requests.get(f"{VALHALLA_URL}/status", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False


# ============================================================================
# USER INPUT SECTION - Get choices from user
# ============================================================================

def get_user_inputs():
    """
    BLOCK 1: Get user configuration
    
    What it does:
    - Asks user to choose resolution
    - Asks user to choose number of sample points
    - Asks user to choose number of cells
    
    Why: Makes script flexible for different analysis needs
    
    ⭐ USER INPUT SECTION - This is where configuration happens
    """
    global RESOLUTION, SAMPLE_POINTS, NUM_CELLS
    
    print("="*70)
    print("INTERACTIVE SPEED & CIRCUITY ANALYSIS")
    print("="*70)
    
    # ========================================
    # VARIABLE 1: Resolution
    # ========================================
    print("\n📏 Choose grid resolution:")
    print("  1. 1000m (1km × 1km cells)")
    print("  2. 500m (500m × 500m cells)")
    print("  3. 250m (250m × 250m cells)")
    
    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()
        if choice == '1':
            RESOLUTION = '1000m'
            break
        elif choice == '2':
            RESOLUTION = '500m'
            break
        elif choice == '3':
            RESOLUTION = '250m'
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"✓ Selected: {RESOLUTION}")
    
    # ========================================
    # VARIABLE 2: Sample Points
    # ========================================
    print("\n📍 How many sample points per cell?")
    print("  Recommendations:")
    print("    - 5 points = 20 routes (fast, for testing)")
    print("    - 10 points = 90 routes (medium)")
    print("    - 20 points = 380 routes (detailed)")
    print("    - 30 points = 870 routes (comprehensive)")
    
    while True:
        try:
            choice = input("\nEnter number of points (5-50): ").strip()
            num = int(choice)
            if 5 <= num <= 50:
                SAMPLE_POINTS = num
                break
            else:
                print("❌ Please enter a number between 5 and 50.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    routes_per_cell = SAMPLE_POINTS * (SAMPLE_POINTS - 1)
    print(f"✓ Selected: {SAMPLE_POINTS} points")
    print(f"  → {routes_per_cell} route pairs per cell")
    
    # ========================================
    # VARIABLE 3: Number of Cells
    # ========================================
    print("\n🏘️  How many cells to analyze?")
    print("  Recommendations:")
    print("    - 10 cells = quick test (~5-10 minutes)")
    print("    - 50 cells = medium analysis (~30-60 minutes)")
    print("    - 100 cells = large analysis (~2-3 hours)")
    print("    - ALL cells = full analysis (several hours)")
    
    while True:
        choice = input("\nEnter number of cells (or 'all' for all cells): ").strip().lower()
        if choice == 'all':
            NUM_CELLS = None  # Will use all cells
            print("✓ Selected: ALL cells")
            break
        else:
            try:
                num = int(choice)
                if num > 0:
                    NUM_CELLS = num
                    print(f"✓ Selected: {NUM_CELLS} cells")
                    break
                else:
                    print("❌ Please enter a positive number.")
            except ValueError:
                print("❌ Please enter a number or 'all'.")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Resolution: {RESOLUTION}")
    print(f"Sample points per cell: {SAMPLE_POINTS}")
    print(f"Route pairs per cell: {routes_per_cell}")
    print(f"Cells to analyze: {'ALL' if NUM_CELLS is None else NUM_CELLS}")
    print(f"Transport modes: {', '.join(MODES)}")
    
    if NUM_CELLS:
        total_routes = NUM_CELLS * routes_per_cell * len(MODES)
        print(f"\nEstimated total routes: ~{total_routes:,}")
        print(f"Estimated time: {total_routes // 100} minutes")
    
    print("="*70)
    
    confirm = input("\nProceed with these settings? (yes/no): ").strip().lower()
    if confirm != 'yes' and confirm != 'y':
        print("❌ Cancelled by user")
        return False
    
    return True


# ============================================================================
# DATA LOADING
# ============================================================================

def load_grid_data():
    """
    BLOCK 2: Load grid data
    
    What it does:
    - Loads CSV file based on RESOLUTION choice
    - Filters to Bremen city
    - Filters to cells with roads
    
    ⭐ Uses RESOLUTION variable set by user
    """
    print("\n" + "="*70)
    print(f"LOADING GRID DATA ({RESOLUTION})")
    print("="*70)
    
    filepath = os.path.join(DATA_DIR, f"{RESOLUTION}.csv")
    
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: {filepath} not found!")
        print(f"\nPlease ensure {RESOLUTION}.csv is in the data/ folder")
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
    print(f"✓ Active cells (with roads): {len(df)}")
    
    return df


# ============================================================================
# SAMPLE POINT GENERATION WITH 50M MINIMUM SPACING
# ============================================================================

def generate_sample_points_with_spacing(bounds, num_points):
    """
    BLOCK 3: Generate sample points with minimum 50m spacing
    
    What it does:
    - Generates random points inside cell boundaries
    - Ensures each point is at least 50m away from all other points
    - Uses iterative placement with distance checking
    
    Input: Cell bounds, number of points to generate
    Output: List of points guaranteed to be 50m+ apart
    
    ⭐ LOGIC CHANGE: 50m minimum spacing enforced here
    ⭐ Uses SAMPLE_POINTS variable set by user
    
    Why: Prevents routes that are too short/meaningless
    """
    MIN_DISTANCE_KM = 0.05  # 50 meters in km
    MAX_ATTEMPTS = 1000     # Prevent infinite loops
    
    points = []
    attempts = 0
    
    while len(points) < num_points and attempts < MAX_ATTEMPTS:
        # Generate random point
        lat = np.random.uniform(bounds['lat_min'], bounds['lat_max'])
        lon = np.random.uniform(bounds['lon_min'], bounds['lon_max'])
        
        # Check distance to all existing points
        too_close = False
        for existing_point in points:
            distance = haversine_distance(
                lat, lon,
                existing_point['lat'], existing_point['lon']
            )
            if distance < MIN_DISTANCE_KM:
                too_close = True
                break
        
        # If far enough from all points, add it
        if not too_close:
            points.append({
                'lat': lat,
                'lon': lon,
                'point_id': len(points)
            })
        
        attempts += 1
    
    # If we couldn't generate enough points, warn user
    if len(points) < num_points:
        print(f"    ⚠️  Warning: Could only generate {len(points)} points (requested {num_points})")
        print(f"       Cell too small for {num_points} points with 50m spacing")
    
    return points


def extract_cell_bounds(geometry_str):
    """Extract cell boundaries from POLYGON string."""
    try:
        coords_str = geometry_str.replace('POLYGON ((', '').replace('))', '')
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


# ============================================================================
# ROUTING CALCULATIONS
# ============================================================================

def get_route_from_valhalla(lat1, lon1, lat2, lon2, mode):
    """Get route details from Valhalla."""
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
    except:
        pass
    return None


def analyze_single_cell(cell, cell_number, total_cells):
    """
    BLOCK 4: Analyze one grid cell
    
    What it does:
    1. Extract cell boundaries
    2. Generate sample points with 50m spacing (uses SAMPLE_POINTS)
    3. Calculate N×(N-1) routes between all point pairs
    4. Skip routes where straight distance ≤ 50m (redundant with spacing, but safe)
    5. For each route:
       - Get straight distance
       - Get route from Valhalla for each mode
       - Calculate speed and circuity
    
    ⭐ LOGIC CHANGE: N×(N-1) directed routes calculated here
    ⭐ Uses SAMPLE_POINTS variable
    
    Why: This is the main analysis loop - one cell at a time
    """
    print(f"\nCell {cell_number}/{total_cells}: {cell['de_grid_id']}")
    
    # Get cell geometry
    bounds = extract_cell_bounds(cell['geometry_4326'])
    if bounds is None:
        print("  ❌ Could not parse geometry")
        return []
    
    # Generate sample points with 50m spacing
    points = generate_sample_points_with_spacing(bounds, SAMPLE_POINTS)
    
    if len(points) < 2:
        print("  ❌ Not enough points generated")
        return []
    
    print(f"  ✓ Generated {len(points)} sample points (50m+ apart)")
    
    # Storage for routes
    routes = []
    route_count = 0
    
    # ========================================
    # MAIN ROUTING LOOP: N×(N-1) combinations
    # ========================================
    # This calculates ALL directed route pairs
    # Point 0 → Point 1, Point 0 → Point 2, etc.
    # Point 1 → Point 0, Point 1 → Point 2, etc.
    # Total: N × (N-1) routes
    # ========================================
    
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:  # Skip same point
                continue
            
            p1 = points[i]
            p2 = points[j]
            
            # Calculate straight-line distance
            straight_dist = haversine_distance(
                p1['lat'], p1['lon'],
                p2['lat'], p2['lon']
            )
            
            # Extra safety check (should already be > 50m due to spacing)
            if straight_dist <= 0.05:  # 50m
                continue
            
            # Get routes for each transport mode
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
                    
                    # Realistic speed check
                    if 1 <= speed_kmh <= 100:
                        # Calculate circuity
                        circuity = distance_km / straight_dist if straight_dist > 0 else None
                        
                        # Realistic circuity check
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


# ============================================================================
# STATISTICS CALCULATION
# ============================================================================

def calculate_cell_statistics(routes_df):
    """
    BLOCK 5: Calculate statistics per cell
    
    What it does:
    - Groups routes by cell_id
    - Calculates avg, median, std for speed and circuity
    - One row per cell with all statistics
    
    Output columns (as requested):
    - avg_speed_<mode>, median_speed_<mode>, std_speed_<mode>
    - avg_cf_<mode>, median_cf_<mode>, std_cf_<mode>
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
        
        # Calculate statistics for each mode
        for mode in MODES:
            mode_routes = cell_routes[cell_routes['mode'] == mode]
            
            if len(mode_routes) > 0:
                # Speed statistics
                summary[f'avg_speed_{mode}'] = mode_routes['speed_kmh'].mean()
                summary[f'median_speed_{mode}'] = mode_routes['speed_kmh'].median()
                summary[f'std_speed_{mode}'] = mode_routes['speed_kmh'].std()
                
                # Circuity statistics (cf = circuity factor)
                summary[f'avg_cf_{mode}'] = mode_routes['circuity'].mean()
                summary[f'median_cf_{mode}'] = mode_routes['circuity'].median()
                summary[f'std_cf_{mode}'] = mode_routes['circuity'].std()
                
                summary[f'num_routes_{mode}'] = len(mode_routes)
            else:
                summary[f'avg_speed_{mode}'] = np.nan
                summary[f'median_speed_{mode}'] = np.nan
                summary[f'std_speed_{mode}'] = np.nan
                summary[f'avg_cf_{mode}'] = np.nan
                summary[f'median_cf_{mode}'] = np.nan
                summary[f'std_cf_{mode}'] = np.nan
                summary[f'num_routes_{mode}'] = 0
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(routes_df, summary_df):
    """
    BLOCK 6: Save results to CSV
    
    What it does:
    - Creates results/ directory
    - Saves detailed routes (every route calculated)
    - Saves cell summaries (statistics per cell)
    
    Output files contain all requested columns
    """
    print("\nSaving results...")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed routes CSV
    routes_file = os.path.join(
        RESULTS_DIR, 
        f"detailed_routes_{RESOLUTION}_{SAMPLE_POINTS}pts_{timestamp}.csv"
    )
    routes_df.to_csv(routes_file, index=False)
    print(f"✓ Detailed routes: {routes_file}")
    print(f"  Columns: {', '.join(routes_df.columns[:8])}...")
    print(f"  ({len(routes_df)} routes)")
    
    # Cell summary CSV
    summary_file = os.path.join(
        RESULTS_DIR,
        f"cell_summary_{RESOLUTION}_{SAMPLE_POINTS}pts_{timestamp}.csv"
    )
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Cell summary: {summary_file}")
    print(f"  Columns: cell_id, avg_speed_*, median_speed_*, std_speed_*, avg_cf_*, etc.")
    print(f"  ({len(summary_df)} cells)")
    
    return routes_file, summary_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    BLOCK 7: Main execution
    
    What it does:
    1. Get user inputs (resolution, points, cells)
    2. Check Valhalla
    3. Load grid data
    4. Analyze cells sequentially
    5. Calculate statistics
    6. Save results
    
    ⭐ Uses all user-configured variables
    """
    print("\n" + "="*70)
    print("INTERACTIVE SPEED & CIRCUITY ANALYSIS")
    print("="*70)
    
    # Step 1: Get user configuration
    if not get_user_inputs():
        return
    
    # Step 2: Check Valhalla
    print("\nChecking Valhalla connection...")
    if not check_valhalla():
        print("❌ Valhalla not accessible!")
        print("   Start with: docker-compose up -d")
        return
    print("✓ Valhalla is running")
    
    # Step 3: Load grid data
    df = load_grid_data()
    if df is None or len(df) == 0:
        return
    
    # Step 4: Select cells to analyze
    if NUM_CELLS is not None and NUM_CELLS < len(df):
        df_analyze = df.head(NUM_CELLS)
        print(f"\n✓ Will analyze first {NUM_CELLS} of {len(df)} cells")
    else:
        df_analyze = df
        print(f"\n✓ Will analyze ALL {len(df)} cells")
    
    # Step 5: Analyze cells sequentially
    print("\n" + "="*70)
    print("ANALYZING CELLS")
    print("="*70)
    
    all_routes = []
    
    for idx, (_, cell) in enumerate(df_analyze.iterrows(), 1):
        cell_routes = analyze_single_cell(cell, idx, len(df_analyze))
        all_routes.extend(cell_routes)
    
    if len(all_routes) == 0:
        print("\n❌ No routes calculated!")
        return
    
    # Step 6: Convert to DataFrame
    routes_df = pd.DataFrame(all_routes)
    
    print(f"\n{'='*70}")
    print("ROUTE CALCULATION COMPLETE")
    print('='*70)
    print(f"Total routes: {len(routes_df):,}")
    
    # Step 7: Calculate statistics
    summary_df = calculate_cell_statistics(routes_df)
    
    # Step 8: Save results
    save_results(routes_df, summary_df)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print('='*70)
    print(f"\nConfiguration used:")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  Sample points: {SAMPLE_POINTS}")
    print(f"  Cells analyzed: {len(df_analyze)}")
    print(f"  Total routes: {len(routes_df):,}")
    print(f"\nCheck results/ folder for CSV files")


if __name__ == '__main__':
    main()