"""
Machine Learning Models for Last-Mile Logistics
================================================

Seven Models:
1. Clustering - Group similar neighborhoods
2. Speed Predictor - Predict delivery speed
3. Hub Optimizer - Find best hub locations
4. Delivery Time Estimator - Estimate trip duration
5. Demand-Aware Hub Optimizer - Using parcel demand
6. Demand Predictor - Forecast parcel volume
7. Demand + Infrastructure Clustering - Identify priority areas

Usage: python ml_models_with_demand.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import folium
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import os

RESULTS_DIR = "results/ml_models"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


# ============================================================================
# MODEL 1: CLUSTERING - Group Similar Cells
# ============================================================================

def extract_cell_bounds_from_geometry(geometry_str):
    """Extract cell boundaries from POLYGON geometry string."""
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
        
        if len(lats) < 2:
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


def cluster_neighborhoods(cell_summary_csv):
    """
    CLUSTERING MODEL
    
    What it does:
    - Groups cells with similar characteristics
    - Uses K-means clustering
    - Features: speed, circuity, population, road density
    
    Why useful:
    - Identify neighborhood types
    - Target delivery strategies per cluster
    - Understand city structure
    
    ⭐ LOGIC: Lines 80-200
    """
    print("="*70)
    print("MODEL 1: NEIGHBORHOOD CLUSTERING")
    print("="*70)
    
    df = pd.read_csv(cell_summary_csv)
    print(f"\nLoaded {len(df)} cells")
    
    # Feature engineering
    features = []
    for mode in ['auto', 'bicycle', 'truck']:
        if f'avg_speed_{mode}' in df.columns:
            features.append(f'avg_speed_{mode}')
        if f'avg_cf_{mode}' in df.columns:
            features.append(f'avg_cf_{mode}')
    
    if 'cell_population' in df.columns:
        features.append('cell_population')
    if 'cell_road_length' in df.columns:
        features.append('cell_road_length')
    
    print(f"\nFeatures: {len(features)}")
    
    df_features = df[features].dropna()
    df_clean = df.loc[df_features.index].copy()
    
    print(f"Complete records: {len(df_features)}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # K-means (k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_clean['cluster'] = labels
    
    # Characterize clusters
    cluster_names = {}
    cluster_colors = {}
    color_map = {
        0: '#e41a1c',  # Red
        1: '#377eb8',  # Blue
        2: '#4daf4a',  # Green
        3: '#984ea3'   # Purple
    }
    
    print(f"\n{'='*70}")
    print("CLUSTER CHARACTERISTICS")
    print('='*70)
    
    for cluster_id in range(4):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        
        avg_speed = cluster_data['avg_speed_auto'].mean() if 'avg_speed_auto' in df_clean.columns else 0
        avg_pop = cluster_data['cell_population'].mean() if 'cell_population' in df_clean.columns else 0
        avg_cf = cluster_data['avg_cf_auto'].mean() if 'avg_cf_auto' in df_clean.columns else 0
        
        print(f"\n📊 Cluster {cluster_id} ({len(cluster_data)} cells)")
        print(f"  Avg speed: {avg_speed:.1f} km/h")
        print(f"  Avg population: {avg_pop:.0f}")
        print(f"  Avg circuity: {avg_cf:.2f}")
        
        # Define cluster type based on characteristics
        if avg_pop > 1000 and avg_speed < 30:
            cluster_type = "Dense Urban Core"
            cluster_names[cluster_id] = cluster_type
        elif avg_pop > 500 and avg_speed > 35:
            cluster_type = "Residential Suburban"
            cluster_names[cluster_id] = cluster_type
        elif avg_pop < 200:
            cluster_type = "Industrial/Sparse"
            cluster_names[cluster_id] = cluster_type
        else:
            cluster_type = "Mixed Use"
            cluster_names[cluster_id] = cluster_type
        
        cluster_colors[cluster_id] = color_map[cluster_id]
        print(f"  Type: {cluster_type}")
    
    # Create map with grid squares
    create_cluster_map(df_clean, cluster_names, cluster_colors)
    
    # Save
    output_file = os.path.join(RESULTS_DIR, "clustered_cells.csv")
    df_clean.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    return df_clean


def create_cluster_map(df_clustered, cluster_names, cluster_colors):
    """
    ⭐ CREATE CLUSTER MAP WITH GRID SQUARES
    
    What it does:
    - Draws black-outlined grid squares for each cell
    - Colors by cluster
    - Adds cluster labels
    
    Why squares: Shows actual grid cell boundaries (not circles)
    """
    print("\nCreating cluster visualization map with grid boundaries...")
    
    center_lat = df_clustered['cell_lat'].mean()
    center_lon = df_clustered['cell_lon'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # ========================================
    # DRAW GRID SQUARES FOR EACH CELL
    # ========================================
    
    for _, row in df_clustered.iterrows():
        cluster_id = int(row['cluster'])
        color = cluster_colors.get(cluster_id, '#cccccc')
        cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        
        # Extract cell geometry if available
        if 'geometry_4326' in row and pd.notna(row['geometry_4326']):
            bounds = extract_cell_bounds_from_geometry(row['geometry_4326'])
            
            if bounds:
                # Draw rectangle (grid square)
                folium.Rectangle(
                    bounds=[
                        [bounds['lat_min'], bounds['lon_min']],
                        [bounds['lat_max'], bounds['lon_max']]
                    ],
                    color='black',           # Black outline
                    weight=2,                # Line width
                    fill=True,
                    fillColor=color,         # Cluster color fill
                    fillOpacity=0.6,
                    popup=f"""
                        <b>Cell:</b> {row['cell_id']}<br>
                        <b>Cluster:</b> {cluster_id}<br>
                        <b>Type:</b> {cluster_name}<br>
                        <b>Population:</b> {row.get('cell_population', 'N/A'):.0f}
                    """
                ).add_to(m)
        else:
            # Fallback: draw rectangle using cell_lat/lon
            # Assume ~500m or 1000m cells (adjust as needed)
            cell_size_deg = 0.005  # ~500m at Bremen latitude
            
            folium.Rectangle(
                bounds=[
                    [row['cell_lat'] - cell_size_deg, row['cell_lon'] - cell_size_deg],
                    [row['cell_lat'] + cell_size_deg, row['cell_lon'] + cell_size_deg]
                ],
                color='black',
                weight=2,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                popup=f"""
                    <b>Cell:</b> {row['cell_id']}<br>
                    <b>Cluster:</b> {cluster_id}<br>
                    <b>Type:</b> {cluster_name}
                """
            ).add_to(m)
    
    # ========================================
    # ADD LEGEND WITH CLUSTER DEFINITIONS
    # ========================================
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                width: 280px; 
                background-color: white; 
                border: 3px solid grey; 
                z-index: 9999; 
                font-size: 12px; 
                padding: 15px;
                border-radius: 5px">
    
    <p style="margin: 0 0 10px 0; font-weight: bold; font-size: 14px;">
        🎯 NEIGHBORHOOD CLUSTERS
    </p>
    
    <hr style="margin: 8px 0;">
    '''
    
    # Add color legend for each cluster
    cluster_descriptions = {
        0: "Dense Urban Core<br/><small>High pop, slow speeds</small>",
        1: "Residential Suburban<br/><small>Medium pop, good speeds</small>",
        2: "Industrial/Sparse<br/><small>Low pop, high speeds</small>",
        3: "Mixed Use<br/><small>Variable characteristics</small>"
    }
    
    for cluster_id in sorted(cluster_names.keys()):
        color = cluster_colors.get(cluster_id, '#cccccc')
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        desc = cluster_descriptions.get(cluster_id, "")
        
        legend_html += f'''
    <div style="margin: 8px 0; padding: 8px; background-color: {color}; 
                border: 1px solid black; border-radius: 3px; opacity: 0.7;">
        <p style="margin: 0; font-weight: bold; color: white; text-shadow: 1px 1px 1px black;">
            Cluster {cluster_id}: {name}
        </p>
        <p style="margin: 3px 0; font-size: 11px; color: white; text-shadow: 1px 1px 1px black;">
            {desc}
        </p>
    </div>
        '''
    
    legend_html += '''
    <hr style="margin: 8px 0;">
    <p style="margin: 5px 0; font-size: 10px; color: #666;">
        Black outline = grid cell boundary<br>
        Color = neighborhood cluster type
    </p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_file = os.path.join(RESULTS_DIR, f"cluster_map_grid_{timestamp}.html")
    m.save(map_file)
    print(f"✓ Cluster map: {map_file}")


# ============================================================================
# MODEL 2: SPEED PREDICTOR
# ============================================================================

def train_speed_predictor(detailed_routes_csv):
    """
    SPEED PREDICTION MODEL
    
    Formula: speed = f(mode, distance, circuity, population)
    
    ⭐ LOGIC: Lines 250-380
    """
    print("\n" + "="*70)
    print("MODEL 2: SPEED PREDICTOR")
    print("="*70)
    
    df = pd.read_csv(detailed_routes_csv)
    print(f"\nLoaded {len(df)} routes")
    
    # Feature engineering
    df['mode_auto'] = (df['mode'] == 'auto').astype(int)
    df['mode_bicycle'] = (df['mode'] == 'bicycle').astype(int)
    df['mode_truck'] = (df['mode'] == 'truck').astype(int)
    df['population_k'] = df['cell_population'] / 1000
    df['road_density'] = df['cell_road_length'] / (df['cell_population'] + 1)
    
    feature_cols = [
        'mode_auto', 'mode_bicycle', 'mode_truck',
        'distance_km', 'circuity', 'population_k', 'road_density'
    ]
    target = 'speed_kmh'
    
    df_model = df[feature_cols + [target]].dropna()
    print(f"Complete records: {len(df_model)}")
    
    X = df_model[feature_cols]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    print("\nTraining model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n{'='*70}")
    print(f"MAE: {mae:.2f} km/h  |  R²: {r2:.3f}")
    print('='*70)
    
    return model, feature_cols


# ============================================================================
# MODEL 3: HUB OPTIMIZER (BASIC)
# ============================================================================

def optimize_hub_locations(cell_summary_csv, num_hubs=5, coverage_km=3.0):
    """
    HUB LOCATION OPTIMIZER (Infrastructure-based)
    
    ⭐ LOGIC: Lines 450-580
    """
    print("\n" + "="*70)
    print("MODEL 3: HUB OPTIMIZER (Infrastructure-based)")
    print("="*70)
    
    df = pd.read_csv(cell_summary_csv)
    
    # Scoring
    df['pop_norm'] = df['cell_population'] / df['cell_population'].max()
    if 'avg_cf_bicycle' in df.columns:
        df['cf_inv_norm'] = (1 / df['avg_cf_bicycle']) / (1 / df['avg_cf_bicycle']).max()
    else:
        df['cf_inv_norm'] = 0.5
    if 'avg_speed_bicycle' in df.columns:
        df['speed_norm'] = df['avg_speed_bicycle'] / df['avg_speed_bicycle'].max()
    else:
        df['speed_norm'] = 0.5
    
    df['hub_score'] = (
        df['pop_norm'] * 0.5 +
        df['cf_inv_norm'] * 0.3 +
        df['speed_norm'] * 0.2
    )
    
    # Greedy selection
    selected_hubs = []
    covered_cells = set()
    df_sorted = df.sort_values('hub_score', ascending=False)
    
    for hub_num in range(num_hubs):
        for _, candidate in df_sorted.iterrows():
            if candidate['cell_id'] in covered_cells:
                continue
            
            selected_hubs.append({
                'hub_id': f"HUB_{hub_num+1}",
                'cell_id': candidate['cell_id'],
                'lat': candidate['cell_lat'],
                'lon': candidate['cell_lon'],
                'score': candidate['hub_score']
            })
            
            for _, cell in df.iterrows():
                dist = haversine_distance(
                    candidate['cell_lat'], candidate['cell_lon'],
                    cell['cell_lat'], cell['cell_lon']
                )
                if dist <= coverage_km:
                    covered_cells.add(cell['cell_id'])
            break
    
    coverage_pct = (len(covered_cells) / len(df)) * 100
    
    print(f"\n✓ {len(selected_hubs)} hubs selected")
    print(f"  Coverage: {coverage_pct:.1f}%")
    
    hubs_df = pd.DataFrame(selected_hubs)
    output_file = os.path.join(RESULTS_DIR, "optimal_hub_locations.csv")
    hubs_df.to_csv(output_file, index=False)
    
    return selected_hubs


# ============================================================================
# MODEL 4: DELIVERY TIME ESTIMATOR
# ============================================================================

def train_delivery_time_estimator(detailed_routes_csv):
    """
    DELIVERY TIME ESTIMATION MODEL
    
    Formula: time = f(distance, mode, circuity, population)
    
    ⭐ LOGIC: Lines 650-780
    """
    print("\n" + "="*70)
    print("MODEL 4: DELIVERY TIME ESTIMATOR")
    print("="*70)
    
    df = pd.read_csv(detailed_routes_csv)
    print(f"\nLoaded {len(df)} routes")
    
    # Feature engineering
    df['mode_auto'] = (df['mode'] == 'auto').astype(int)
    df['mode_bicycle'] = (df['mode'] == 'bicycle').astype(int)
    df['mode_truck'] = (df['mode'] == 'truck').astype(int)
    df['population_k'] = df['cell_population'] / 1000
    df['distance_x_circuity'] = df['distance_km'] * df['circuity']
    df['distance_x_pop'] = df['distance_km'] * df['population_k']
    
    feature_cols = [
        'mode_auto', 'mode_bicycle', 'mode_truck',
        'distance_km', 'circuity', 'population_k',
        'distance_x_circuity', 'distance_x_pop'
    ]
    target = 'time_minutes'
    
    df_model = df[feature_cols + [target]].dropna()
    print(f"Complete records: {len(df_model)}")
    
    X = df_model[feature_cols]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train (Gradient Boosting)
    print("\nTraining model...")
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n{'='*70}")
    print(f"MAE: {mae:.2f} minutes  |  R²: {r2:.3f}")
    print('='*70)
    
    return model, feature_cols


# ============================================================================
# MODEL 5: DEMAND-AWARE HUB OPTIMIZER
# ============================================================================

def optimize_hubs_with_demand(cell_summary_csv, demand_csv, num_hubs=5, coverage_km=3.0):
    """
    DEMAND-AWARE HUB OPTIMIZER
    
    What it does:
    - Places hubs where parcel demand is highest
    - Weights by expected_parcels column
    
    Why useful:
    - Serves most parcels with fewer hubs
    - Better capacity planning
    
    ⭐ LOGIC: Lines 850-1000
    """
    print("\n" + "="*70)
    print("MODEL 5: DEMAND-AWARE HUB OPTIMIZER")
    print("="*70)
    
    df_cells = pd.read_csv(cell_summary_csv)
    df_demand = pd.read_csv(demand_csv)
    
    print(f"\nLoaded {len(df_cells)} cells")
    print(f"Loaded {len(df_demand)} demand records")
    
    # Aggregate demand per cell
    demand_summary = df_demand.groupby('de_grid_id').agg({
        'expected_parcels': 'sum'
    }).reset_index()
    
    # Merge
    df = df_cells.merge(
        demand_summary, 
        left_on='cell_id', 
        right_on='de_grid_id', 
        how='left'
    )
    df['expected_parcels'] = df['expected_parcels'].fillna(0)
    
    print(f"Cells with demand: {(df['expected_parcels'] > 0).sum()}")
    print(f"Total parcels: {df['expected_parcels'].sum():.0f}")
    
    # Demand-weighted scoring
    df['demand_norm'] = df['expected_parcels'] / (df['expected_parcels'].max() + 1)
    
    if 'avg_speed_bicycle' in df.columns:
        df['speed_norm'] = df['avg_speed_bicycle'] / df['avg_speed_bicycle'].max()
    else:
        df['speed_norm'] = 0.5
    
    if 'avg_cf_bicycle' in df.columns:
        df['cf_inv_norm'] = (1 / df['avg_cf_bicycle']) / (1 / df['avg_cf_bicycle']).max()
    else:
        df['cf_inv_norm'] = 0.5
    
    # 60% demand weight
    df['hub_score_demand'] = (
        df['demand_norm'] * 0.60 +
        df['speed_norm'] * 0.25 +
        df['cf_inv_norm'] * 0.15
    )
    
    # Greedy selection
    selected_hubs = []
    covered_cells = set()
    total_served_demand = 0
    
    df_sorted = df.sort_values('hub_score_demand', ascending=False)
    
    for hub_num in range(num_hubs):
        for _, candidate in df_sorted.iterrows():
            if candidate['cell_id'] in covered_cells:
                continue
            
            hub_demand = 0
            
            for _, cell in df.iterrows():
                if cell['cell_id'] in covered_cells:
                    continue
                
                dist = haversine_distance(
                    candidate['cell_lat'], candidate['cell_lon'],
                    cell['cell_lat'], cell['cell_lon']
                )
                
                if dist <= coverage_km:
                    covered_cells.add(cell['cell_id'])
                    hub_demand += cell['expected_parcels']
            
            selected_hubs.append({
                'hub_id': f"HUB_{hub_num+1}",
                'cell_id': candidate['cell_id'],
                'lat': candidate['cell_lat'],
                'lon': candidate['cell_lon'],
                'score': candidate['hub_score_demand'],
                'demand_served': hub_demand
            })
            
            total_served_demand += hub_demand
            break
    
    coverage_pct = (total_served_demand / df['expected_parcels'].sum()) * 100 if df['expected_parcels'].sum() > 0 else 0
    
    print(f"\n✓ {len(selected_hubs)} hubs selected")
    print(f"  Demand coverage: {coverage_pct:.1f}%")
    print(f"  Total parcels served: {total_served_demand:.0f}")
    
    hubs_df = pd.DataFrame(selected_hubs)
    output_file = os.path.join(RESULTS_DIR, "demand_aware_hubs.csv")
    hubs_df.to_csv(output_file, index=False)
    
    return selected_hubs


# ============================================================================
# MODEL 6: DEMAND PREDICTOR
# ============================================================================

def train_demand_predictor(demand_csv, cell_summary_csv):
    """
    PARCEL DEMAND PREDICTOR
    
    What it does:
    - Predicts parcel volume from demographics + infrastructure
    
    Formula: parcels = f(population, age_distribution, speed, circuity)
    
    ⭐ LOGIC: Lines 1080-1200
    """
    print("\n" + "="*70)
    print("MODEL 6: DEMAND PREDICTOR")
    print("="*70)
    
    df_demand = pd.read_csv(demand_csv)
    df_cells = pd.read_csv(cell_summary_csv)
    
    print(f"\nLoaded {len(df_demand)} demand records")
    
    # Aggregate demand
    demand_summary = df_demand.groupby('de_grid_id').agg({
        'expected_parcels': 'sum'
    }).reset_index()
    
    # Merge
    df = df_cells.merge(
        demand_summary,
        left_on='cell_id',
        right_on='de_grid_id',
        how='inner'
    )
    
    print(f"Complete records: {len(df)}")
    
    # Features
    feature_cols = ['cell_population', 'cell_road_length']
    
    if 'avg_speed_auto' in df.columns:
        feature_cols.append('avg_speed_auto')
    if 'avg_cf_bicycle' in df.columns:
        feature_cols.append('avg_cf_bicycle')
    
    target = 'expected_parcels'
    
    df_model = df[feature_cols + [target]].dropna()
    print(f"Records for training: {len(df_model)}")
    
    X = df_model[feature_cols]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    print("\nTraining model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n{'='*70}")
    print(f"MAE: {mae:.0f} parcels/year  |  R²: {r2:.3f}")
    print('='*70)
    
    return model, feature_cols


# ============================================================================
# MODEL 7: DEMAND + INFRASTRUCTURE CLUSTERING
# ============================================================================

def cluster_by_demand_and_infrastructure(cell_summary_csv, demand_csv):
    """
    DEMAND + INFRASTRUCTURE CLUSTERING
    
    What it does:
    - Identifies priority intervention areas
    - Finds "high demand + poor infrastructure"
    
    ⭐ LOGIC: Lines 1280-1420
    """
    print("\n" + "="*70)
    print("MODEL 7: DEMAND + INFRASTRUCTURE CLUSTERING")
    print("="*70)
    
    df_cells = pd.read_csv(cell_summary_csv)
    df_demand = pd.read_csv(demand_csv)
    
    # Aggregate demand
    demand_summary = df_demand.groupby('de_grid_id').agg({
        'expected_parcels': 'sum'
    }).reset_index()
    
    # Merge
    df = df_cells.merge(
        demand_summary,
        left_on='cell_id',
        right_on='de_grid_id',
        how='left'
    )
    df['expected_parcels'] = df['expected_parcels'].fillna(0)
    
    print(f"\nLoaded {len(df)} cells")
    
    # Features for clustering
    features = ['expected_parcels']
    
    if 'avg_speed_bicycle' in df.columns:
        features.append('avg_speed_bicycle')
    if 'avg_cf_bicycle' in df.columns:
        features.append('avg_cf_bicycle')
    if 'cell_population' in df.columns:
        features.append('cell_population')
    
    df_features = df[features].dropna()
    df_clean = df.loc[df_features.index].copy()
    
    print(f"Complete records: {len(df_features)}")
    
    # Standardize and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    df_clean['demand_infra_cluster'] = labels
    
    print(f"\n{'='*70}")
    print("CLUSTER CHARACTERISTICS")
    print('='*70)
    
    for cluster_id in range(4):
        cluster_data = df_clean[df_clean['demand_infra_cluster'] == cluster_id]
        
        avg_demand = cluster_data['expected_parcels'].mean()
        avg_speed = cluster_data['avg_speed_bicycle'].mean() if 'avg_speed_bicycle' in cluster_data.columns else 0
        
        print(f"\n📦 Cluster {cluster_id} ({len(cluster_data)} cells)")
        print(f"  Avg demand: {avg_demand:.0f} parcels/year")
        print(f"  Avg speed: {avg_speed:.1f} km/h")
        
        # Categorize
        demand_median = df_clean['expected_parcels'].median()
        speed_median = df_clean['avg_speed_bicycle'].median() if 'avg_speed_bicycle' in df_clean.columns else 0
        
        if avg_demand > demand_median and avg_speed < speed_median:
            cluster_type = "🔴 HIGH DEMAND + POOR INFRA (PRIORITY!)"
        elif avg_demand > demand_median:
            cluster_type = "🟢 HIGH DEMAND + GOOD INFRA (Optimal)"
        elif avg_speed >= speed_median:
            cluster_type = "🟡 LOW DEMAND + GOOD INFRA (Expansion)"
        else:
            cluster_type = "⚪ LOW DEMAND + POOR INFRA (Monitor)"
        
        print(f"  Type: {cluster_type}")
    
    # Save
    output_file = os.path.join(RESULTS_DIR, "demand_infra_clusters.csv")
    df_clean.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    return df_clean


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all 7 ML models."""
    print("="*70)
    print("MACHINE LEARNING MODELS FOR LAST-MILE LOGISTICS")
    print("With Demand Prediction")
    print("="*70)
    
    # File paths
    cell_summary = "results/speed/cell_summary_1000m_high.csv"
    detailed_routes = "results/speed/detailed_routes_1000m_high.csv"
    demand_data = "data/demand_1km_bremen.csv"
    
    # Check infrastructure data
    if not os.path.exists(cell_summary) or not os.path.exists(detailed_routes):
        print("\n❌ ERROR: Infrastructure data not found!")
        print("   Run interactive_analysis_enhanced.py first!")
        return
    
    # ===== INFRASTRUCTURE MODELS =====
    print("\n" + "="*70)
    print("PART 1: INFRASTRUCTURE MODELS")
    print("="*70)
    
    df_clustered = cluster_neighborhoods(cell_summary)
    speed_model, speed_features = train_speed_predictor(detailed_routes)
    hubs_infra = optimize_hub_locations(cell_summary, num_hubs=5, coverage_km=3.0)
    time_model, time_features = train_delivery_time_estimator(detailed_routes)
    
    # ===== DEMAND MODELS =====
    if os.path.exists(demand_data):
        print("\n" + "="*70)
        print("PART 2: DEMAND-AWARE MODELS")
        print("="*70)
        
        hubs_demand = optimize_hubs_with_demand(cell_summary, demand_data, num_hubs=5, coverage_km=3.0)
        demand_model, demand_features = train_demand_predictor(demand_data, cell_summary)
        df_demand_clusters = cluster_by_demand_and_infrastructure(cell_summary, demand_data)
    else:
        print(f"\n⚠️  Demand data not found: {demand_data}")
        print("   Skipping models 5, 6, 7")
    
    # Summary
    print("\n" + "="*70)
    print("✅ ALL MODELS COMPLETE")
    print("="*70)
    print(f"\nOutputs: {RESULTS_DIR}/")
    print("\nInfrastructure Models:")
    print("  1. clustered_cells.csv")
    print("  2. [Speed model trained]")
    print("  3. optimal_hub_locations.csv")
    print("  4. [Time model trained]")
    
    if os.path.exists(demand_data):
        print("\nDemand Models:")
        print("  5. demand_aware_hubs.csv")
        print("  6. [Demand model trained]")
        print("  7. demand_infra_clusters.csv")
    
    print("\n✅ Ready for predictions!")


if __name__ == '__main__':
    main()