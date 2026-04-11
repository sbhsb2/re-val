"""
Interactive Speed & Circuity Analysis with Route Visualization
===============================================================

Features:
- Choose resolution (1000m/500m/250m)
- Choose sample points
- Smart grid selection by population levels
- Interactive maps showing routes by transport mode
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
import folium
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

VALHALLA_URL = "http://localhost:8002"
DATA_DIR = "data"
RESULTS_DIR = "results"
MODES = ['auto', 'bicycle', 'truck']

# User-configured variables
RESOLUTION = None
SAMPLE_POINTS = None
NUM_CELLS = None
GRID_SELECTION = None  # 'high', 'medium', 'low', or 'mixed'

# Bremen bounding box for maps
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
    """
    Decode Valhalla's encoded polyline to lat/lon coordinates.
    
    What it does:
    - Converts encoded string to list of [lat, lon] points
    - Used to draw actual route paths on map
    
    Why: Valhalla returns routes as encoded strings (compact format)
         We need actual coordinates to draw on map
    """
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
    """
    BLOCK 1: Get user configuration
    
    ⭐ USER INPUT SECTION
    - Resolution choice
    - Sample points
    - Grid selection strategy
    """
    global RESOLUTION, SAMPLE_POINTS, NUM_CELLS, GRID_SELECTION
    
    print("="*70)
    print("INTERACTIVE SPEED & CIRCUITY ANALYSIS")
    print("With Route Visualization Maps")
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
    print("    - 10 points = 90 routes (good for visualization)")
    print("    - 15 points = 210 routes (detailed)")
    
    while True:
        try:
            choice = input("\nEnter number of points (5-30): ").strip()
            num = int(choice)
            if 5 <= num <= 30:
                SAMPLE_POINTS = num
                break
            else:
                print("❌ Please enter a number between 5 and 30.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    routes_per_cell = SAMPLE_POINTS * (SAMPLE_POINTS - 1)
    print(f"✓ Selected: {SAMPLE_POINTS} points")
    print(f"  → {routes_per_cell} route pairs per cell")
    
    # ========================================
    # VARIABLE 3: Grid Selection Strategy (NEW!)
    # ========================================
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
    
    # ========================================
    # VARIABLE 4: Number of Cells
    # ========================================
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
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Resolution: {RESOLUTION}")
    print(f"Sample points per cell: {SAMPLE_POINTS}")
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
    """
    BLOCK 2: Select grids based on population strategy
    
    What it does:
    - Categorizes cells into high/medium/low population
    - Selects cells based on user's choice
    
    ⭐ GRID SELECTION LOGIC - Population-based filtering
    
    Categories:
    - High: Top 33% by population
    - Medium: Middle 33%
    - Low: Bottom 33% (including 0)
    """
    print("\n" + "="*70)
    print(f"SELECTING GRIDS ({GRID_SELECTION.upper()} POPULATION)")
    print("="*70)
    
    # Fill NaN populations with 0
    df['population'] = df['population'].fillna(0)
    
    # Sort by population
    df_sorted = df.sort_values('population', ascending=False).reset_index(drop=True)
    
    total_cells = len(df_sorted)
    third = total_cells // 3
    
    if GRID_SELECTION == 'high':
        # Top 33% by population
        df_selected = df_sorted.head(third)
        print(f"✓ High population cells: top {third} cells")
        print(f"  Population range: {df_selected['population'].min():.0f} - {df_selected['population'].max():.0f}")
    
    elif GRID_SELECTION == 'medium':
        # Middle 33%
        df_selected = df_sorted.iloc[third:2*third]
        print(f"✓ Medium population cells: {third} cells")
        print(f"  Population range: {df_selected['population'].min():.0f} - {df_selected['population'].max():.0f}")
    
    elif GRID_SELECTION == 'low':
        # Bottom 33%
        df_selected = df_sorted.tail(third)
        print(f"✓ Low population cells: bottom {third} cells")
        print(f"  Population range: {df_selected['population'].min():.0f} - {df_selected['population'].max():.0f}")
    
    else:  # mixed
        # Take some from each category
        cells_per_category = NUM_CELLS // 3
        high = df_sorted.head(cells_per_category)
        medium = df_sorted.iloc[third:third+cells_per_category]
        low = df_sorted.tail(cells_per_category)
        df_selected = pd.concat([high, medium, low])
        print(f"✓ Mixed cells: {cells_per_category} from each category")
        print(f"  High pop: {high['population'].min():.0f} - {high['population'].max():.0f}")
        print(f"  Medium pop: {medium['population'].min():.0f} - {medium['population'].max():.0f}")
        print(f"  Low pop: {low['population'].min():.0f} - {low['population'].max():.0f}")
    
    # Take only requested number of cells
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
# SAMPLE POINT GENERATION WITH 50M SPACING
# ============================================================================

def generate_sample_points_with_spacing(bounds, num_points):
    """
    BLOCK 3: Generate sample points with 50m minimum spacing
    
    ⭐ 50M SPACING LOGIC enforced here
    """
    MIN_DISTANCE_KM = 0.05
    MAX_ATTEMPTS = 1000
    
    points = []
    attempts = 0
    
    while len(points) < num_points and attempts < MAX_ATTEMPTS:
        lat = np.random.uniform(bounds['lat_min'], bounds['lat_max'])
        lon = np.random.uniform(bounds['lon_min'], bounds['lon_max'])
        
        too_close = False
        for existing_point in points:
            distance = haversine_distance(
                lat, lon,
                existing_point['lat'], existing_point['lon']
            )
            if distance < MIN_DISTANCE_KM:
                too_close = True
                break
        
        if not too_close:
            points.append({
                'lat': lat,
                'lon': lon,
                'point_id': len(points)
            })
        
        attempts += 1
    
    if len(points) < num_points:
        print(f"    ⚠️  Warning: Only generated {len(points)}/{num_points} points")
    
    return points


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
# ROUTING WITH GEOMETRY
# ============================================================================

def get_route_from_valhalla(lat1, lon1, lat2, lon2, mode):
    """
    Get route details from Valhalla including geometry.
    
    Returns:
    - distance_km
    - time_seconds
    - geometry (encoded polyline)
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
            
            # Get encoded geometry
            geometry = None
            if 'legs' in data['trip'] and len(data['trip']['legs']) > 0:
                geometry = data['trip']['legs'][0].get('shape', None)
            
            return {
                'distance_km': summary['length'],
                'time_seconds': summary['time'],
                'geometry': geometry
            }
    except:
        pass
    return None


def analyze_single_cell(cell, cell_number, total_cells):
    """
    BLOCK 4: Analyze one grid cell
    
    ⭐ N×(N-1) ROUTING LOGIC - All directed pairs calculated here
    
    Also stores route geometry for map visualization
    """
    print(f"\nCell {cell_number}/{total_cells}: {cell['de_grid_id']}")
    
    bounds = extract_cell_bounds(cell['geometry_4326'])
    if bounds is None:
        print("  ❌ Could not parse geometry")
        return []
    
    points = generate_sample_points_with_spacing(bounds, SAMPLE_POINTS)
    
    if len(points) < 2:
        print("  ❌ Not enough points")
        return []
    
    print(f"  ✓ Generated {len(points)} sample points (50m+ apart)")
    
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
                                'geometry': geometry  # Store for map
                            })
                            
                            route_count += 1
    
    print(f"  ✓ Calculated {route_count} valid routes")
    
    return routes


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
# MAP VISUALIZATION (NEW!)
# ============================================================================

def create_interactive_route_map(routes_df):
    """
    BLOCK 5: Create interactive map with route visualization
    
    What it does:
    - Creates map centered on Bremen
    - For each cell, draws grid boundary
    - Routes grouped by cell and mode (separate layers)
    - Click cell to see routes
    - Toggle modes on/off
    
    ⭐ MAP CREATION - Shows actual route geometries
    
    Features:
    - Layer control for each cell × mode combination
    - Route colors: auto=blue, bicycle=green, truck=orange
    - Sample points shown as circles
    - Cell boundaries as rectangles
    """
    print("\nCreating interactive route visualization map...")
    
    center_lat = (BREMEN_LAT_MIN + BREMEN_LAT_MAX) / 2
    center_lon = (BREMEN_LON_MIN + BREMEN_LON_MAX) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Colors for modes
    mode_colors = {
        'auto': '#1f77b4',      # Blue
        'bicycle': '#2ca02c',   # Green
        'truck': '#ff7f0e'      # Orange
    }
    
    # Process each cell
    for cell_id in routes_df['cell_id'].unique():
        cell_routes = routes_df[routes_df['cell_id'] == cell_id]
        
        # Get cell info
        cell_lat = cell_routes['cell_lat'].iloc[0]
        cell_lon = cell_routes['cell_lon'].iloc[0]
        cell_pop = cell_routes['cell_population'].iloc[0]
        
        # Create layer group for each mode
        for mode in MODES:
            mode_routes = cell_routes[cell_routes['mode'] == mode]
            
            if len(mode_routes) == 0:
                continue
            
            # Feature group for this cell × mode combination
            fg = folium.FeatureGroup(
                name=f"{cell_id} - {mode.upper()}",
                show=False  # Hidden by default to avoid clutter
            )
            
            # Draw each route
            for _, route in mode_routes.iterrows():
                if route['geometry']:
                    # Decode geometry
                    try:
                        coords = decode_polyline(route['geometry'])
                        
                        # Draw route line
                        folium.PolyLine(
                            locations=coords,
                            color=mode_colors[mode],
                            weight=2,
                            opacity=0.6,
                            popup=f"""
                                <b>Mode:</b> {mode}<br>
                                <b>Distance:</b> {route['distance_km']:.2f} km<br>
                                <b>Time:</b> {route['time_minutes']:.1f} min<br>
                                <b>Speed:</b> {route['speed_kmh']:.1f} km/h<br>
                                <b>Circuity:</b> {route['circuity']:.2f}
                            """
                        ).add_to(fg)
                    except:
                        pass
            
            # Add sample points (only once per cell, not per mode)
            if mode == MODES[0]:  # Only for first mode
                # Get unique origin points
                origins = mode_routes[['point1_lat', 'point1_lon']].drop_duplicates()
                
                for _, point in origins.iterrows():
                    folium.CircleMarker(
                        location=[point['point1_lat'], point['point1_lon']],
                        radius=3,
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.7,
                        popup=f"Sample point"
                    ).add_to(fg)
            
            fg.add_to(m)
    
    # Add layer control
    folium.LayerControl(
        position='topleft',
        collapsed=False
    ).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px">
    <p style="margin: 0; font-weight: bold;">Route Visualization</p>
    <p style="margin: 5px 0;">Resolution: {RESOLUTION}</p>
    <p style="margin: 5px 0;">Sample points: {SAMPLE_POINTS}</p>
    <hr>
    <p style="margin: 5px 0; font-weight: bold;">Transport Modes:</p>
    <p style="margin: 2px;"><span style="color:#1f77b4;">━━</span> Car</p>
    <p style="margin: 2px;"><span style="color:#2ca02c;">━━</span> Bicycle</p>
    <p style="margin: 2px;"><span style="color:#ff7f0e;">━━</span> Truck</p>
    <hr>
    <p style="margin: 5px 0; font-size:10px;">Use layer control to toggle routes</p>
    <p style="margin: 5px 0; font-size:10px;">Red dots = sample points</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_file = os.path.join(
        RESULTS_DIR,
        f"route_map_{RESOLUTION}_{GRID_SELECTION}_{timestamp}.html"
    )
    m.save(map_file)
    print(f"✓ Interactive map: {map_file}")
    print(f"  Features: {len(routes_df['cell_id'].unique())} cells × {len(MODES)} modes")
    print(f"  Use layer control to show/hide routes")
    
    return map_file


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(routes_df, summary_df):
    """Save CSV files."""
    print("\nSaving CSV results...")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed routes (without geometry for CSV)
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with map visualization."""
    print("\n" + "="*70)
    print("INTERACTIVE SPEED & CIRCUITY ANALYSIS")
    print("With Route Visualization")
    print("="*70)
    
    # Get user inputs
    if not get_user_inputs():
        return
    
    # Check Valhalla
    print("\nChecking Valhalla connection...")
    if not check_valhalla():
        print("❌ Valhalla not accessible!")
        return
    print("✓ Valhalla is running")
    
    # Load and filter data
    df = load_grid_data()
    if df is None or len(df) == 0:
        return
    
    # Select grids by population
    df_selected = select_grids_by_population(df)
    
    # Analyze cells
    print("\n" + "="*70)
    print("ANALYZING CELLS")
    print("="*70)
    
    all_routes = []
    
    for idx, (_, cell) in enumerate(df_selected.iterrows(), 1):
        cell_routes = analyze_single_cell(cell, idx, len(df_selected))
        all_routes.extend(cell_routes)
    
    if len(all_routes) == 0:
        print("\n❌ No routes calculated!")
        return
    
    routes_df = pd.DataFrame(all_routes)
    
    print(f"\n{'='*70}")
    print("ROUTE CALCULATION COMPLETE")
    print('='*70)
    print(f"Total routes: {len(routes_df):,}")
    
    # Calculate statistics
    summary_df = calculate_cell_statistics(routes_df)
    
    # Create interactive map
    create_interactive_route_map(routes_df)
    
    # Save CSV results
    save_results(routes_df, summary_df)
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print('='*70)
    print(f"\nConfiguration:")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  Grid selection: {GRID_SELECTION}")
    print(f"  Sample points: {SAMPLE_POINTS}")
    print(f"  Cells analyzed: {len(df_selected)}")
    print(f"  Total routes: {len(routes_df):,}")
    print(f"\nOutputs:")
    print(f"  ✓ Interactive route map (HTML)")
    print(f"  ✓ Detailed routes CSV")
    print(f"  ✓ Cell summary CSV")
    print(f"\nCheck results/ folder!")


if __name__ == '__main__':
    main()