"""
Machine Learning Models for Last-Mile Logistics
================================================

Four Models:
1. Clustering - Group similar neighborhoods
2. Speed Predictor - Predict delivery speed
3. Hub Optimizer - Find best hub locations
4. Delivery Time Estimator - Estimate trip duration

Usage: python ml_models_analysis.py
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
import os

RESULTS_DIR = "results/ml_models"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# MODEL 1: CLUSTERING - Group Similar Cells
# ============================================================================

def cluster_neighborhoods(cell_summary_csv):
    """
    CLUSTERING MODEL
    
    What it does:
    - Groups cells with similar characteristics
    - Uses K-means clustering
    - Features: speed, circuity, population, road density
    
    Why useful:
    - Identify neighborhood types (dense urban, suburban, industrial)
    - Target delivery strategies per cluster
    - Understand city structure
    
    ⭐ LOGIC LOCATION: Lines 60-150
    """
    print("="*70)
    print("MODEL 1: NEIGHBORHOOD CLUSTERING")
    print("="*70)
    
    # Load data
    df = pd.read_csv(cell_summary_csv)
    
    print(f"\nLoaded {len(df)} cells")
    
    # ========================================
    # FEATURE ENGINEERING - What to cluster on?
    # ========================================
    # Lines 72-85
    
    features = []
    feature_names = []
    
    # Speed features (all modes)
    for mode in ['auto', 'bicycle', 'truck']:
        if f'avg_speed_{mode}' in df.columns:
            features.append(f'avg_speed_{mode}')
            feature_names.append(f'Speed ({mode})')
    
    # Circuity features (all modes)
    for mode in ['auto', 'bicycle', 'truck']:
        if f'avg_cf_{mode}' in df.columns:
            features.append(f'avg_cf_{mode}')
            feature_names.append(f'Circuity ({mode})')
    
    # Context features
    if 'cell_population' in df.columns:
        features.append('cell_population')
        feature_names.append('Population')
    
    if 'cell_road_length' in df.columns:
        features.append('cell_road_length')
        feature_names.append('Road Length')
    
    print(f"\nFeatures for clustering: {len(features)}")
    for fname in feature_names:
        print(f"  - {fname}")
    
    # Prepare data
    df_features = df[features].copy()
    
    # Remove rows with missing values
    df_features = df_features.dropna()
    df_clean = df.loc[df_features.index].copy()
    
    print(f"\nCells with complete data: {len(df_features)}")
    
    # ========================================
    # STANDARDIZATION - Scale features
    # ========================================
    # Lines 110-115
    # Why: Speed (0-50) and population (0-5000) have different scales
    #      K-means is sensitive to scale, so we normalize
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # ========================================
    # K-MEANS CLUSTERING
    # ========================================
    # Lines 120-140
    # Try different numbers of clusters (3-6)
    
    best_k = None
    best_score = -999
    
    for k in range(3, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = kmeans.inertia_  # Lower is better
        
        print(f"\nK={k} clusters: inertia={score:.2f}")
        
        # Use silhouette score or elbow method in practice
        # For simplicity, using k=4 (typical: urban/suburban/industrial/mixed)
        if k == 4:
            best_k = k
            best_labels = labels
            best_kmeans = kmeans
    
    df_clean['cluster'] = best_labels
    
    # ========================================
    # CLUSTER ANALYSIS
    # ========================================
    # Lines 145-175
    
    print(f"\n{'='*70}")
    print(f"CLUSTER CHARACTERISTICS (K={best_k})")
    print('='*70)
    
    for cluster_id in range(best_k):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        
        print(f"\n📊 Cluster {cluster_id} ({len(cluster_data)} cells):")
        
        # Characterize each cluster
        if 'avg_speed_auto' in df_clean.columns:
            print(f"  Avg car speed: {cluster_data['avg_speed_auto'].mean():.1f} km/h")
        
        if 'avg_cf_auto' in df_clean.columns:
            print(f"  Avg circuity: {cluster_data['avg_cf_auto'].mean():.2f}")
        
        if 'cell_population' in df_clean.columns:
            print(f"  Avg population: {cluster_data['cell_population'].mean():.0f}")
        
        if 'cell_road_length' in df_clean.columns:
            print(f"  Avg road length: {cluster_data['cell_road_length'].mean():.0f}m")
        
        # Give cluster a name based on characteristics
        avg_speed = cluster_data['avg_speed_auto'].mean() if 'avg_speed_auto' in df_clean.columns else 0
        avg_pop = cluster_data['cell_population'].mean() if 'cell_population' in df_clean.columns else 0
        
        if avg_pop > 1000 and avg_speed < 30:
            cluster_name = "Dense Urban Core"
        elif avg_pop > 500 and avg_speed > 35:
            cluster_name = "Residential Suburban"
        elif avg_pop < 200:
            cluster_name = "Industrial/Sparse"
        else:
            cluster_name = "Mixed Use"
        
        print(f"  Type: {cluster_name}")
    
    # Save results
    output_file = os.path.join(RESULTS_DIR, "clustered_cells.csv")
    df_clean.to_csv(output_file, index=False)
    print(f"\n✓ Saved clustered data: {output_file}")
    
    # Create visualization map
    create_cluster_map(df_clean)
    
    return df_clean, best_kmeans, scaler


def create_cluster_map(df_clustered):
    """Create map showing clusters with different colors."""
    
    print("\nCreating cluster visualization map...")
    
    # Bremen center
    center_lat = df_clustered['cell_lat'].mean()
    center_lon = df_clustered['cell_lon'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Colors for clusters
    cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    for _, row in df_clustered.iterrows():
        cluster_id = int(row['cluster'])
        color = cluster_colors[cluster_id % len(cluster_colors)]
        
        folium.CircleMarker(
            location=[row['cell_lat'], row['cell_lon']],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=f"""
                <b>Cluster:</b> {cluster_id}<br>
                <b>Cell:</b> {row['cell_id']}<br>
                <b>Population:</b> {row.get('cell_population', 'N/A')}
            """
        ).add_to(m)
    
    map_file = os.path.join(RESULTS_DIR, "cluster_map.html")
    m.save(map_file)
    print(f"✓ Cluster map: {map_file}")


# ============================================================================
# MODEL 2: SPEED PREDICTOR
# ============================================================================

def train_speed_predictor(detailed_routes_csv):
    """
    SPEED PREDICTION MODEL
    
    What it does:
    - Predicts speed given route characteristics
    - Uses Random Forest regression
    - Features: mode, distance, circuity, population, road density
    
    Why useful:
    - Estimate delivery time without running Valhalla
    - Fast what-if scenarios
    - Identify speed bottlenecks
    
    Formula: speed = f(mode, distance, circuity, population, road_density)
    
    ⭐ LOGIC LOCATION: Lines 240-360
    """
    print("\n" + "="*70)
    print("MODEL 2: SPEED PREDICTOR")
    print("="*70)
    
    # Load detailed routes
    df = pd.read_csv(detailed_routes_csv)
    
    print(f"\nLoaded {len(df)} routes")
    
    # ========================================
    # FEATURE ENGINEERING
    # ========================================
    # Lines 260-285
    # Create features that might affect speed
    
    # One-hot encode transport mode
    df['mode_auto'] = (df['mode'] == 'auto').astype(int)
    df['mode_bicycle'] = (df['mode'] == 'bicycle').astype(int)
    df['mode_truck'] = (df['mode'] == 'truck').astype(int)
    
    # Normalize population (per 1000 people)
    df['population_k'] = df['cell_population'] / 1000
    
    # Road density (meters per population)
    df['road_density'] = df['cell_road_length'] / (df['cell_population'] + 1)
    
    # Features for model
    feature_cols = [
        'mode_auto', 'mode_bicycle', 'mode_truck',  # Transport mode
        'distance_km',                               # Route distance
        'circuity',                                  # Route efficiency
        'population_k',                              # Area density
        'road_density'                               # Infrastructure
    ]
    
    # Target variable
    target = 'speed_kmh'
    
    # Remove rows with missing values
    df_model = df[feature_cols + [target]].dropna()
    
    print(f"\nRoutes with complete data: {len(df_model)}")
    print(f"\nFeatures:")
    for col in feature_cols:
        print(f"  - {col}")
    
    # ========================================
    # TRAIN/TEST SPLIT
    # ========================================
    # Lines 305-310
    # 80% training, 20% testing
    
    X = df_model[feature_cols]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # ========================================
    # TRAIN MODEL
    # ========================================
    # Lines 320-330
    # Random Forest: ensemble of decision trees
    # Why: Handles non-linear relationships, robust to outliers
    
    print("\nTraining Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=100,      # 100 trees
        max_depth=15,          # Prevent overfitting
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # ========================================
    # EVALUATE MODEL
    # ========================================
    # Lines 340-365
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE")
    print('='*70)
    print(f"\nTraining Set:")
    print(f"  MAE: {train_mae:.2f} km/h (average error)")
    print(f"  R²: {train_r2:.3f} (variance explained)")
    
    print(f"\nTest Set:")
    print(f"  MAE: {test_mae:.2f} km/h")
    print(f"  R²: {test_r2:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model (would use pickle/joblib in practice)
    print(f"\n✓ Model trained successfully")
    
    return model, feature_cols


# ============================================================================
# MODEL 3: HUB OPTIMIZER
# ============================================================================

def optimize_hub_locations(cell_summary_csv, num_hubs=5, coverage_km=3.0):
    """
    HUB LOCATION OPTIMIZER
    
    What it does:
    - Finds optimal hub locations
    - Greedy algorithm maximizing coverage
    - Considers population, speed, circuity
    
    Why useful:
    - Data-driven hub placement
    - Maximize service quality
    - Minimize delivery distance
    
    Formula: best_location = maximize(coverage × quality)
    
    ⭐ LOGIC LOCATION: Lines 400-520
    """
    print("\n" + "="*70)
    print("MODEL 3: HUB LOCATION OPTIMIZER")
    print("="*70)
    
    df = pd.read_csv(cell_summary_csv)
    
    print(f"\nLoaded {len(df)} cells")
    print(f"Target: {num_hubs} hubs")
    print(f"Coverage radius: {coverage_km} km")
    
    # ========================================
    # CALCULATE CELL SCORES
    # ========================================
    # Lines 425-450
    # Score = desirability of placing a hub in this cell
    
    # Normalize features (0-1 scale)
    df['pop_norm'] = df['cell_population'] / df['cell_population'].max()
    
    # Inverse circuity (lower is better, so invert)
    if 'avg_cf_bicycle' in df.columns:
        df['cf_inv_norm'] = (1 / df['avg_cf_bicycle']) / (1 / df['avg_cf_bicycle']).max()
    else:
        df['cf_inv_norm'] = 0.5
    
    # Speed score (higher is better)
    if 'avg_speed_bicycle' in df.columns:
        df['speed_norm'] = df['avg_speed_bicycle'] / df['avg_speed_bicycle'].max()
    else:
        df['speed_norm'] = 0.5
    
    # Combined score (weighted)
    df['hub_score'] = (
        df['pop_norm'] * 0.5 +        # 50% weight on population (demand)
        df['cf_inv_norm'] * 0.3 +     # 30% weight on low circuity
        df['speed_norm'] * 0.2         # 20% weight on speed
    )
    
    print("\nHub scoring complete")
    
    # ========================================
    # GREEDY HUB SELECTION
    # ========================================
    # Lines 460-510
    # Algorithm:
    # 1. Pick highest-score uncovered cell
    # 2. Mark all cells within coverage_km as covered
    # 3. Repeat until num_hubs selected
    
    selected_hubs = []
    covered_cells = set()
    
    # Sort by score
    df_sorted = df.sort_values('hub_score', ascending=False)
    
    for hub_num in range(num_hubs):
        # Find best uncovered cell
        for _, candidate in df_sorted.iterrows():
            if candidate['cell_id'] in covered_cells:
                continue
            
            # This is our next hub
            selected_hubs.append({
                'hub_id': f"HUB_{hub_num+1}",
                'cell_id': candidate['cell_id'],
                'lat': candidate['cell_lat'],
                'lon': candidate['cell_lon'],
                'score': candidate['hub_score'],
                'population': candidate['cell_population']
            })
            
            # Mark coverage area
            for _, cell in df.iterrows():
                # Calculate distance from hub to this cell
                dist = haversine_distance(
                    candidate['cell_lat'], candidate['cell_lon'],
                    cell['cell_lat'], cell['cell_lon']
                )
                
                if dist <= coverage_km:
                    covered_cells.add(cell['cell_id'])
            
            break
    
    # ========================================
    # RESULTS
    # ========================================
    
    coverage_pct = (len(covered_cells) / len(df)) * 100
    
    print(f"\n{'='*70}")
    print("OPTIMAL HUB LOCATIONS")
    print('='*70)
    
    for hub in selected_hubs:
        print(f"\n{hub['hub_id']}:")
        print(f"  Cell: {hub['cell_id']}")
        print(f"  Location: ({hub['lat']:.4f}, {hub['lon']:.4f})")
        print(f"  Score: {hub['score']:.3f}")
        print(f"  Population: {hub['population']:.0f}")
    
    print(f"\nCoverage: {len(covered_cells)}/{len(df)} cells ({coverage_pct:.1f}%)")
    
    # Save
    hubs_df = pd.DataFrame(selected_hubs)
    output_file = os.path.join(RESULTS_DIR, "optimal_hub_locations.csv")
    hubs_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved hub locations: {output_file}")
    
    # Create map
    create_hub_map(df, selected_hubs, covered_cells, coverage_km)
    
    return selected_hubs


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def create_hub_map(df_cells, hubs, covered_cells, coverage_km):
    """Create map showing hub locations and coverage."""
    
    print("\nCreating hub visualization map...")
    
    center_lat = df_cells['cell_lat'].mean()
    center_lon = df_cells['cell_lon'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Draw coverage circles
    for hub in hubs:
        folium.Circle(
            location=[hub['lat'], hub['lon']],
            radius=coverage_km * 1000,  # km to meters
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.1,
            popup=f"{hub['hub_id']}<br>Coverage: {coverage_km} km"
        ).add_to(m)
    
    # Draw hub markers
    for hub in hubs:
        folium.Marker(
            location=[hub['lat'], hub['lon']],
            popup=f"""
                <b>{hub['hub_id']}</b><br>
                Score: {hub['score']:.3f}<br>
                Population: {hub['population']:.0f}
            """,
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m)
    
    # Draw covered cells (green) and uncovered (gray)
    for _, cell in df_cells.iterrows():
        color = 'green' if cell['cell_id'] in covered_cells else 'gray'
        
        folium.CircleMarker(
            location=[cell['cell_lat'], cell['cell_lon']],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6
        ).add_to(m)
    
    map_file = os.path.join(RESULTS_DIR, "hub_locations_map.html")
    m.save(map_file)
    print(f"✓ Hub map: {map_file}")


# ============================================================================
# MODEL 4: DELIVERY TIME ESTIMATOR
# ============================================================================

def train_delivery_time_estimator(detailed_routes_csv):
    """
    DELIVERY TIME ESTIMATION MODEL
    
    What it does:
    - Predicts trip duration (time_minutes)
    - Uses Gradient Boosting
    - Features: distance, mode, circuity, population
    
    Why useful:
    - Quick time estimates without Valhalla
    - Delivery scheduling
    - Customer ETA predictions
    
    Formula: time = f(distance, mode, circuity, traffic_proxy)
    
    ⭐ LOGIC LOCATION: Lines 580-680
    """
    print("\n" + "="*70)
    print("MODEL 4: DELIVERY TIME ESTIMATOR")
    print("="*70)
    
    df = pd.read_csv(detailed_routes_csv)
    
    print(f"\nLoaded {len(df)} routes")
    
    # ========================================
    # FEATURE ENGINEERING
    # ========================================
    # Lines 600-620
    # Similar to speed predictor but targeting TIME
    
    df['mode_auto'] = (df['mode'] == 'auto').astype(int)
    df['mode_bicycle'] = (df['mode'] == 'bicycle').astype(int)
    df['mode_truck'] = (df['mode'] == 'truck').astype(int)
    
    df['population_k'] = df['cell_population'] / 1000
    
    # Interaction features
    df['distance_x_circuity'] = df['distance_km'] * df['circuity']
    df['distance_x_pop'] = df['distance_km'] * df['population_k']
    
    feature_cols = [
        'mode_auto', 'mode_bicycle', 'mode_truck',
        'distance_km',
        'circuity',
        'population_k',
        'distance_x_circuity',  # Longer routes in complex areas take even more time
        'distance_x_pop'         # Dense areas slow things down
    ]
    
    target = 'time_minutes'
    
    df_model = df[feature_cols + [target]].dropna()
    
    print(f"\nRoutes with complete data: {len(df_model)}")
    
    # ========================================
    # TRAIN MODEL
    # ========================================
    # Lines 640-655
    # Gradient Boosting: iteratively improves predictions
    # Why: Better than Random Forest for this regression task
    
    X = df_model[feature_cols]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nTraining Gradient Boosting model...")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # ========================================
    # EVALUATE
    # ========================================
    
    y_pred_test = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE")
    print('='*70)
    print(f"\nTest Set:")
    print(f"  MAE: {mae:.2f} minutes (average error)")
    print(f"  R²: {r2:.3f}")
    
    print(f"\n✓ Delivery time estimator trained")
    
    return model, feature_cols


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run all four ML models.
    
    Requires:
    - cell_summary CSV (from your analysis)
    - detailed_routes CSV (from your analysis)
    """
    print("="*70)
    print("MACHINE LEARNING MODELS FOR LAST-MILE LOGISTICS")
    print("="*70)
    
    # File paths (update these!)
    cell_summary = "results/speed/cell_summary_1000m_high.csv"
    detailed_routes = "results/speed/detailed_routes_1000m_high.csv"
    
    # Check files exist
    if not os.path.exists(cell_summary):
        print(f"\n❌ ERROR: {cell_summary} not found!")
        print("   Run interactive_analysis_enhanced.py first!")
        return
    
    if not os.path.exists(detailed_routes):
        print(f"\n❌ ERROR: {detailed_routes} not found!")
        print("   Run interactive_analysis_enhanced.py first!")
        return
    
    # Model 1: Clustering
    df_clustered, kmeans_model, scaler = cluster_neighborhoods(cell_summary)
    
    # Model 2: Speed Predictor
    speed_model, speed_features = train_speed_predictor(detailed_routes)
    
    # Model 3: Hub Optimizer
    hubs = optimize_hub_locations(cell_summary, num_hubs=5, coverage_km=3.0)
    
    # Model 4: Delivery Time Estimator
    time_model, time_features = train_delivery_time_estimator(detailed_routes)
    
    print("\n" + "="*70)
    print("✅ ALL MODELS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {RESULTS_DIR}/")
    print("  - clustered_cells.csv")
    print("  - cluster_map.html")
    print("  - optimal_hub_locations.csv")
    print("  - hub_locations_map.html")
    print("\nModels ready for predictions!")


if __name__ == '__main__':
    main()