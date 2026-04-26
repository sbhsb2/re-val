"""
Interactive Speed & Circuity Analysis - Enhanced Version
=========================================================

New Features:
- Grid subdivision for systematic sample point placement
- Visual grid boundaries and sub-squares on map
- Detailed route failure logging
- Population-based grid selection

Usage: python interactive_analysis_enhanced.py
"""

import pandas as pd
import numpy as np
import requests
from math import radians, sin, cos, sqrt, atan2, floor
from datetime import datetime
import os
import folium
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

VALHALLA_URL = "http://localhost:8002"
DATA_DIR = "data"
RESULTS_DIR = "results"
MODES = ['auto', 'bicycle', 'truck']

RESOLUTION = None
SAMPLE_POINTS = None
NUM_CELLS = None
GRID_SELECTION = None

BREMEN_LAT_MIN = 53.00
BREMEN_LAT_MAX = 53.15
BREMEN_LON_MIN = 8.70
BREMEN_LON_MAX = 8.95


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


def decode_polyline(encoded, precision=6):
    """Decode Valhalla's encoded polyline to lat/lon coordinates."""
    inv = 1.0 / (10 ** precision)
    decoded = []
    previous = [0, 0]
    i = 0
    
    while i < len(encoded):
        ll = [0, 0]
        for j in [0, 1]:
            shift = 0
            byte = 0x20
            while byte >= 0x20:
                byte = ord(encoded[i]) - 63
                i += 1
                ll[j] |= (byte & 0x1f) << shift
                shift += 5
            ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
            previous[j] = ll[j]
        decoded.append([ll[0] * inv, ll[1] * inv])
    
    return decoded


# ============================================================================
# USER INPUT SECTION
# ============================================================================

def get_user_inputs():
    """Get user configuration."""
    global RESOLUTION, SAMPLE_POINTS, NUM_CELLS, GRID_SELECTION
    
    print("="*70)
    print("INTERACTIVE SPEED & CIRCUITY ANALYSIS - ENHANCED")
    print("Grid Subdivision + Visual Boundaries + Error Logging")
    print("="*70)
    
    # Resolution
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
    
    # Sample Points
    print("\n📍 How many sample points per cell?")
    print("  Note: Will use grid subdivision (nearest perfect square)")
    print("  Examples:")
    print("    - 9 points → 3×3 grid")
    print("    - 10 points → 3×3 grid (9 points used)")
    print("    - 16 points → 4×4 grid")
    print("    - 25 points → 5×5 grid")
    
    while True:
        try:
            choice = input("\nEnter number of points (4-25): ").strip()
            num = int(choice)
            if 4 <= num <= 25:
                SAMPLE_POINTS = num
                break
            else:
                print("❌ Please enter a number between 4 and 25.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Calculate actual grid size
    grid_size = int(floor(sqrt(SAMPLE_POINTS)))
    actual_points = grid_size * grid_size
    routes_per_cell = actual_points * (actual_points - 1)
    
    print(f"✓ Selected: {SAMPLE_POINTS} points")
    print(f"  → Using {grid_size}×{grid_size} grid = {actual_points} points")
    print(f"  → {routes_per_cell} route pairs per cell")
    
    # Grid Selection
    print("\n🏘️  Select grid cells to analyze:")
    print("  1. High population areas (dense neighborhoods)")
    print("  2. Medium population areas (moderate density)")
    print("  3. Low/No population areas (industrial/sparse)")
    print("  4. Mixed (some from each category)")
    
    while True:
        choice = input("\nEnter choice (1/2/3/4): ").strip()
        if choice == '1':
            GRID_SELECTION = 'high'
            print("✓ Selected: High population areas")
            break
        elif choice == '2':
            GRID_SELECTION = 'medium'
            print("✓ Selected: Medium population areas")
            break
        elif choice == '3':
            GRID_SELECTION = 'low'
            print("✓ Selected: Low/No population areas")
            break
        elif choice == '4':
            GRID_SELECTION = 'mixed'
            print("✓ Selected: Mixed population areas")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    # Number of Cells
    print("\n📊 How many cells to analyze?")
    print("  Recommendations:")
    print("    - 5 cells = very quick test (~5 minutes)")
    print("    - 10 cells = quick test with good visualization")
    print("    - 20 cells = medium analysis (~30-60 minutes)")
    
    while True:
        try:
            choice = input("\nEnter number of cells (5-50): ").strip()
            num = int(choice)
            if 5 <= num <= 50:
                NUM_CELLS = num
                print(f"✓ Selected: {NUM_CELLS} cells")
                break
            else:
                print("❌ Please enter a number between 5 and 50.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Resolution: {RESOLUTION}")
    print(f"Grid subdivision: {grid_size}×{grid_size}")
    print(f"Actual sample points: {actual_points}")
    print(f"Route pairs per cell: {routes_per_cell}")
    print(f"Grid selection: {GRID_SELECTION.upper()} population areas")
    print(f"Cells to analyze: {NUM_CELLS}")
    print(f"Transport modes: {', '.join(MODES)}")
    
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
# GRID SELECTION BY POPULATION
# ============================================================================

def select_grids_by_population(df):
    """Select grids based on population strategy."""
    print("\n" + "="*70)
    print(f"SELECTING GRIDS ({GRID_SELECTION.upper()} POPULATION)")
    print("="*70)
    
    df['population'] = df['population'].fillna(0)
    df_sorted = df.sort_values('population', ascending=False).reset_index(drop=True)
    
    total_cells = len(df_sorted)
    third = total_cells // 3
    
    if GRID_SELECTION == 'high':
        df_selected = df_sorted.head(third)
        print(f"✓ High population cells: top {third} cells")
        print(f"  Population range: {df_selected['population'].min():.0f} - {df_selected['population'].max():.0f}")
    
    elif GRID_SELECTION == 'medium':
        df_selected = df_sorted.iloc[third:2*third]
        print(f"✓ Medium population cells: {third} cells")
        print(f"  Population range: {df_selected['population'].min():.0f} - {df_selected['population'].max():.0f}")
    
    elif GRID_SELECTION == 'low':
        df_selected = df_sorted.tail(third)
        print(f"✓ Low population cells: bottom {third} cells")
        print(f"  Population range: {df_selected['population'].min():.0f} - {df_selected['population'].max():.0f}")
    
    else:  # mixed
        cells_per_category = NUM_CELLS // 3
        high = df_sorted.head(cells_per_category)
        medium = df_sorted.iloc[third:third+cells_per_category]
        low = df_sorted.tail(cells_per_category)
        df_selected = pd.concat([high, medium, low])
        print(f"✓ Mixed cells: {cells_per_category} from each category")
    
    df_final = df_selected.head(NUM_CELLS)
    print(f"\n✓ Selected {len(df_final)} cells for analysis")
    
    return df_final


def load_grid_data():
    """Load and filter grid data."""
    print("\n" + "="*70)
    print(f"LOADING GRID DATA ({RESOLUTION})")
    print("="*70)
    
    filepath = os.path.join(DATA_DIR, f"{RESOLUTION}.csv")
    
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: {filepath} not found!")
        return None
    
    print(f"\nLoading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✓ Loaded {len(df)} total cells")
    
    if 'city_name' in df.columns:
        df = df[df['city_name'] == 'Bremen, Stadt'].copy()
        print(f"✓ Bremen cells: {len(df)}")
    
    df = df[df['road_length_m'] > 0].copy()
    print(f"✓ Active cells (with roads): {len(df)}")
    
    return df


# ============================================================================
# GRID SUBDIVISION SAMPLE POINT GENERATION
# ============================================================================

def generate_grid_subdivision_points(bounds, num_points):
    """
    BLOCK 3: Generate sample points using grid subdivision
    
    ⭐ NEW LOGIC: Grid subdivision method
    
    What it does:
    1. Calculate grid size (e.g., 9 points → 3×3 grid)
    2. Divide cell into equal sub-squares
    3. Place point at center of each sub-square (diagonal intercept)
    4. Verify 50m spacing (backup check)
    
    Why: Ensures systematic, well-distributed sample points
    """
    
    # Calculate grid dimensions
    grid_size = int(floor(sqrt(num_points)))
    actual_points = grid_size * grid_size
    
    if actual_points < num_points:
        print(f"    ℹ️  Using {grid_size}×{grid_size} grid ({actual_points} points instead of {num_points})")
    
    # Cell dimensions
    lat_range = bounds['lat_max'] - bounds['lat_min']
    lon_range = bounds['lon_max'] - bounds['lon_min']
    
    # Sub-square dimensions
    sub_lat_size = lat_range / grid_size
    sub_lon_size = lon_range / grid_size
    
    points = []
    sub_squares = []  # Store for visualization
    
    # Generate points at center of each sub-square
    for row in range(grid_size):
        for col in range(grid_size):
            # Sub-square bounds
            sub_lat_min = bounds['lat_min'] + row * sub_lat_size
            sub_lat_max = sub_lat_min + sub_lat_size
            sub_lon_min = bounds['lon_min'] + col * sub_lon_size
            sub_lon_max = sub_lon_min + sub_lon_size
            
            # Center point (diagonal intercept)
            center_lat = (sub_lat_min + sub_lat_max) / 2
            center_lon = (sub_lon_min + sub_lon_max) / 2
            
            points.append({
                'lat': center_lat,
                'lon': center_lon,
                'point_id': len(points),
                'row': row,
                'col': col
            })
            
            sub_squares.append({
                'bounds': [[sub_lat_min, sub_lon_min], [sub_lat_max, sub_lon_max]],
                'row': row,
                'col': col
            })
    
    # Verify 50m minimum spacing (safety check)
    MIN_DISTANCE_KM = 0.05
    violations = 0
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = haversine_distance(
                points[i]['lat'], points[i]['lon'],
                points[j]['lat'], points[j]['lon']
            )
            if dist < MIN_DISTANCE_KM:
                violations += 1
    
    if violations > 0:
        print(f"    ⚠️  Warning: {violations} point pairs < 50m apart")
    
    return points, sub_squares


def extract_cell_bounds(geometry_str):
    """Extract cell boundaries from POLYGON."""
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
# ROUTING WITH ENHANCED ERROR LOGGING
# ============================================================================

# Global route failure tracking
route_failures = {
    'auto': {'total': 0, 'reasons': {}},
    'bicycle': {'total': 0, 'reasons': {}},
    'truck': {'total': 0, 'reasons': {}}
}


def get_route_from_valhalla(lat1, lon1, lat2, lon2, mode):
    """
    Get route from Valhalla with detailed error logging.
    
    ⭐ ENHANCED: Now logs why routes fail
    
    Returns:
    - dict with distance, time, geometry if successful
    - None if failed (with logged reason)
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
            data = response.json()
            summary = data['trip']['summary']
            
            geometry = None
            if 'legs' in data['trip'] and len(data['trip']['legs']) > 0:
                geometry = data['trip']['legs'][0].get('shape', None)
            
            return {
                'distance_km': summary['length'],
                'time_seconds': summary['time'],
                'geometry': geometry
            }
        else:
            # Log failure reason
            reason = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    reason = error_data['error']
            except:
                pass
            
            route_failures[mode]['total'] += 1
            route_failures[mode]['reasons'][reason] = route_failures[mode]['reasons'].get(reason, 0) + 1
            
            return None
            
    except requests.exceptions.Timeout:
        route_failures[mode]['total'] += 1
        route_failures[mode]['reasons']['Timeout (>10s)'] = route_failures[mode]['reasons'].get('Timeout (>10s)', 0) + 1
        return None
    except Exception as e:
        route_failures[mode]['total'] += 1
        reason = f"Exception: {type(e).__name__}"
        route_failures[mode]['reasons'][reason] = route_failures[mode]['reasons'].get(reason, 0) + 1
        return None


def analyze_single_cell(cell, cell_number, total_cells):
    """
    Analyze one grid cell with grid subdivision.
    
    ⭐ UPDATED: Uses grid subdivision for sample points
    ⭐ STORES: Sub-square info for visualization
    """
    print(f"\nCell {cell_number}/{total_cells}: {cell['de_grid_id']}")
    
    bounds = extract_cell_bounds(cell['geometry_4326'])
    if bounds is None:
        print("  ❌ Could not parse geometry")
        return [], None, None
    
    # Generate points using grid subdivision
    points, sub_squares = generate_grid_subdivision_points(bounds, SAMPLE_POINTS)
    
    print(f"  ✓ Generated {len(points)} points in grid pattern")
    
    routes = []
    route_count = 0
    
    # N×(N-1) directed pairs
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            
            p1 = points[i]
            p2 = points[j]
            
            straight_dist = haversine_distance(
                p1['lat'], p1['lon'],
                p2['lat'], p2['lon']
            )
            
            if straight_dist <= 0.05:
                continue
            
            for mode in MODES:
                route_info = get_route_from_valhalla(
                    p1['lat'], p1['lon'],
                    p2['lat'], p2['lon'],
                    mode
                )
                
                if route_info:
                    distance_km = route_info['distance_km']
                    time_seconds = route_info['time_seconds']
                    geometry = route_info['geometry']
                    
                    time_hours = time_seconds / 3600
                    speed_kmh = distance_km / time_hours if time_hours > 0 else 0
                    
                    if 1 <= speed_kmh <= 100:
                        circuity = distance_km / straight_dist if straight_dist > 0 else None
                        
                        if circuity and 1.0 <= circuity <= 5.0:
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
                                'circuity': circuity,
                                'geometry': geometry
                            })
                            
                            route_count += 1
    
    print(f"  ✓ Calculated {route_count} valid routes")
    
    return routes, points, sub_squares


# ============================================================================
# STATISTICS CALCULATION
# ============================================================================

def calculate_cell_statistics(routes_df):
    """Calculate avg, median, std for each cell and mode."""
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
                summary[f'avg_speed_{mode}'] = mode_routes['speed_kmh'].mean()
                summary[f'median_speed_{mode}'] = mode_routes['speed_kmh'].median()
                summary[f'std_speed_{mode}'] = mode_routes['speed_kmh'].std()
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
# MAP VISUALIZATION WITH GRID BOUNDARIES
# ============================================================================

def create_interactive_route_map(routes_df, cell_data_dict):
    """
    Create interactive map with route visualization AND grid boundaries.
    
    ⭐ NEW FEATURES:
    - Black outline for main grid cell boundary
    - Colored sub-squares showing grid subdivision
    - Click routes to see grid highlights
    
    cell_data_dict: {cell_id: (bounds, points, sub_squares)}
    
    TO REMOVE SUB-SQUARE VISUALIZATION:
    - Comment out lines marked with "# REMOVE THIS BLOCK"
    """
    print("\nCreating interactive route visualization map...")
    
    center_lat = (BREMEN_LAT_MIN + BREMEN_LAT_MAX) / 2
    center_lon = (BREMEN_LON_MIN + BREMEN_LON_MAX) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    mode_colors = {
        'auto': '#1f77b4',
        'bicycle': '#2ca02c',
        'truck': '#ff7f0e'
    }
    
    # Process each cell
    for cell_id in routes_df['cell_id'].unique():
        cell_routes = routes_df[routes_df['cell_id'] == cell_id]
        
        if cell_id not in cell_data_dict:
            continue
        
        bounds, points, sub_squares = cell_data_dict[cell_id]
        
        # ====================================================================
        # GRID BOUNDARY VISUALIZATION
        # ====================================================================
        
        # Main grid cell boundary (BLACK OUTLINE)
        # ⭐ This shows the main grid cell when clicking routes
        grid_boundary = folium.Rectangle(
            bounds=[
                [bounds['lat_min'], bounds['lon_min']],
                [bounds['lat_max'], bounds['lon_max']]
            ],
            color='black',
            weight=3,
            fill=False,
            popup=f"<b>Grid Cell:</b> {cell_id}"
        )
        grid_boundary.add_to(m)
        
        # ====================================================================
        # SUB-SQUARE VISUALIZATION (OPTIONAL - CAN BE REMOVED)
        # ====================================================================
        # REMOVE THIS BLOCK START (if you don't want sub-squares)
        for sub in sub_squares:
            folium.Rectangle(
                bounds=sub['bounds'],
                color='#FFD700',  # Gold color for sub-squares
                weight=1,
                fill=True,
                fillColor='#FFD700',
                fillOpacity=0.1,
                popup=f"Sub-square ({sub['row']},{sub['col']})"
            ).add_to(m)
        # REMOVE THIS BLOCK END
        # ====================================================================
        
        # Routes by mode
        for mode in MODES:
            mode_routes = cell_routes[cell_routes['mode'] == mode]
            
            if len(mode_routes) == 0:
                continue
            
            fg = folium.FeatureGroup(
                name=f"{cell_id} - {mode.upper()}",
                show=False
            )
            
            # Draw routes
            for _, route in mode_routes.iterrows():
                if route['geometry']:
                    try:
                        coords = decode_polyline(route['geometry'])
                        
                        folium.PolyLine(
                            locations=coords,
                            color=mode_colors[mode],
                            weight=2,
                            opacity=0.6,
                            popup=f"""
                                <b>Cell:</b> {cell_id}<br>
                                <b>Mode:</b> {mode}<br>
                                <b>Distance:</b> {route['distance_km']:.2f} km<br>
                                <b>Time:</b> {route['time_minutes']:.1f} min<br>
                                <b>Speed:</b> {route['speed_kmh']:.1f} km/h<br>
                                <b>Circuity:</b> {route['circuity']:.2f}
                            """
                        ).add_to(fg)
                    except:
                        pass
            
            # Add sample points (only once)
            if mode == MODES[0]:
                for point in points:
                    folium.CircleMarker(
                        location=[point['lat'], point['lon']],
                        radius=3,
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.7,
                        popup=f"Sample point {point['point_id']}"
                    ).add_to(fg)
            
            fg.add_to(m)
    
    folium.LayerControl(
        position='topleft',
        collapsed=False
    ).add_to(m)
    
    # Legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px">
    <p style="margin: 0; font-weight: bold;">Route Visualization</p>
    <p style="margin: 5px 0;">Resolution: {RESOLUTION}</p>
    <p style="margin: 5px 0;">Grid: {int(floor(sqrt(SAMPLE_POINTS)))}×{int(floor(sqrt(SAMPLE_POINTS)))}</p>
    <hr>
    <p style="margin: 5px 0; font-weight: bold;">Transport Modes:</p>
    <p style="margin: 2px;"><span style="color:#1f77b4;">━━</span> Car</p>
    <p style="margin: 2px;"><span style="color:#2ca02c;">━━</span> Bicycle</p>
    <p style="margin: 2px;"><span style="color:#ff7f0e;">━━</span> Truck</p>
    <hr>
    <p style="margin: 5px 0; font-weight: bold;">Grid Boundaries:</p>
    <p style="margin: 2px;"><span style="color:black;">▬</span> Main cell</p>
    <p style="margin: 2px;"><span style="color:#FFD700;">□</span> Sub-squares</p>
    <p style="margin: 2px;"><span style="color:red;">●</span> Sample points</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_file = os.path.join(
        RESULTS_DIR,
        f"route_map_{RESOLUTION}_{GRID_SELECTION}_{timestamp}.html"
    )
    m.save(map_file)
    print(f"✓ Interactive map: {map_file}")
    
    return map_file


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(routes_df, summary_df):
    """Save CSV files."""
    print("\nSaving CSV results...")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    routes_csv = routes_df.drop('geometry', axis=1, errors='ignore')
    
    routes_file = os.path.join(
        RESULTS_DIR,
        f"detailed_routes_{RESOLUTION}_{GRID_SELECTION}_{timestamp}.csv"
    )
    routes_csv.to_csv(routes_file, index=False)
    print(f"✓ Detailed routes: {routes_file}")
    print(f"  ({len(routes_csv)} routes)")
    
    summary_file = os.path.join(
        RESULTS_DIR,
        f"cell_summary_{RESOLUTION}_{GRID_SELECTION}_{timestamp}.csv"
    )
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Cell summary: {summary_file}")
    print(f"  ({len(summary_df)} cells)")
    
    return routes_file, summary_file


def print_route_failure_summary():
    """
    Print summary of why routes failed.
    
    ⭐ NEW: Shows routing failure statistics
    """
    print("\n" + "="*70)
    print("ROUTE FAILURE ANALYSIS")
    print("="*70)
    
    for mode in MODES:
        failures = route_failures[mode]
        
        if failures['total'] == 0:
            print(f"\n{mode.upper()}: ✅ All routes successful")
        else:
            print(f"\n{mode.upper()}: ❌ {failures['total']} routes failed")
            print("  Reasons:")
            
            sorted_reasons = sorted(
                failures['reasons'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for reason, count in sorted_reasons:
                print(f"    • {reason}: {count} times")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with enhanced features."""
    print("\n" + "="*70)
    print("INTERACTIVE SPEED & CIRCUITY ANALYSIS - ENHANCED")
    print("="*70)
    
    if not get_user_inputs():
        return
    
    print("\nChecking Valhalla connection...")
    if not check_valhalla():
        print("❌ Valhalla not accessible!")
        return
    print("✓ Valhalla is running")
    
    df = load_grid_data()
    if df is None or len(df) == 0:
        return
    
    df_selected = select_grids_by_population(df)
    
    print("\n" + "="*70)
    print("ANALYZING CELLS")
    print("="*70)
    
    all_routes = []
    cell_data_dict = {}  # Store bounds, points, sub_squares for map
    
    for idx, (_, cell) in enumerate(df_selected.iterrows(), 1):
        cell_routes, points, sub_squares = analyze_single_cell(cell, idx, len(df_selected))
        
        if cell_routes:
            all_routes.extend(cell_routes)
            
            # Extract bounds from first route
            if len(cell_routes) > 0:
                first_route = cell_routes[0]
                bounds = extract_cell_bounds(cell['geometry_4326'])
                cell_data_dict[cell['de_grid_id']] = (bounds, points, sub_squares)
    
    if len(all_routes) == 0:
        print("\n❌ No routes calculated!")
        print_route_failure_summary()
        return
    
    routes_df = pd.DataFrame(all_routes)
    
    print(f"\n{'='*70}")
    print("ROUTE CALCULATION COMPLETE")
    print('='*70)
    print(f"Total routes: {len(routes_df):,}")
    
    # Print failure summary
    print_route_failure_summary()
    
    summary_df = calculate_cell_statistics(routes_df)
    
    create_interactive_route_map(routes_df, cell_data_dict)
    
    save_results(routes_df, summary_df)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print('='*70)
    print(f"\nConfiguration:")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  Grid selection: {GRID_SELECTION}")
    print(f"  Grid subdivision: {int(floor(sqrt(SAMPLE_POINTS)))}×{int(floor(sqrt(SAMPLE_POINTS)))}")
    print(f"  Sample points: {int(floor(sqrt(SAMPLE_POINTS)))**2}")
    print(f"  Cells analyzed: {len(df_selected)}")
    print(f"  Total routes: {len(routes_df):,}")
    print(f"\nOutputs:")
    print(f"  ✓ Interactive route map (with grid boundaries)")
    print(f"  ✓ Detailed routes CSV")
    print(f"  ✓ Cell summary CSV")
    print(f"\nCheck results/ folder!")


if __name__ == '__main__':
    main()