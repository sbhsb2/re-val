# TEMPORAL DEMAND CODE - CORRECTED FOR YOUR ACTUAL CSV STRUCTURE

## Your CSV Structure:
# Monthly columns: Jan, Feb, Mar, ..., Dec (total parcels per month)
# Daily breakdown: Jan_Mon, Jan_Tue, ..., Dec_Sat (parcels per day of week per month)
# Plus: de_grid_id, city_name, population, avg_age, expected_parcels, geometry

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# ============================================================================
# PART 1: PARSE AND AGGREGATE YOUR TEMPORAL DEMAND DATA
# ============================================================================

def process_temporal_demand_CORRECTED(demand_csv_path):
    """
    Process demand CSV with your actual structure.
    
    Input columns:
    - Monthly: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
    - Daily: Jan_Mon, Jan_Tue, Jan_Wed, Jan_Thu, Jan_Fri, Jan_Sat, 
             Feb_Mon, Feb_Tue, ..., Dec_Sat
    - Other: de_grid_id, city_name, population, avg_age, expected_parcels
    
    Output: Multiple aggregations for clustering + forecasting
    """
    
    df = pd.read_csv(demand_csv_path)
    
    print(f"Loaded {len(df)} cells")
    print(f"Columns: {len(df.columns)}")
    
    # ========================================
    # AGGREGATION 1: Annual Total (for base clustering)
    # ========================================
    
    monthly_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    df['annual_parcels'] = df[monthly_cols].sum(axis=1)
    
    print(f"\nANNUAL AGGREGATION:")
    print(f"  Total cells: {len(df)}")
    print(f"  Total annual parcels: {df['annual_parcels'].sum():,.0f}")
    print(f"  Avg per cell: {df['annual_parcels'].mean():,.0f}")
    
    # ========================================
    # AGGREGATION 2: Monthly Pattern (seasonality)
    # ========================================
    
    monthly_pattern = df[monthly_cols].mean()
    
    print(f"\nMONTHLY PATTERN (Average per cell):")
    for month, parcels in monthly_pattern.items():
        pct_of_avg = 100 * parcels / monthly_pattern.mean()
        print(f"  {month}: {parcels:,.0f} parcels/month ({pct_of_avg:.0f}% of avg)")
    
    # Identify peak and trough months
    peak_month = monthly_pattern.idxmax()
    trough_month = monthly_pattern.idxmin()
    peak_ratio = monthly_pattern.max() / monthly_pattern.mean()
    
    print(f"\n  Peak month: {peak_month} ({peak_ratio:.1%} above average)")
    print(f"  Trough month: {trough_month} ({(1-monthly_pattern.min()/monthly_pattern.mean()):.1%} below average)")
    
    # ========================================
    # AGGREGATION 3: Day-of-Week Pattern
    # ========================================
    
    # Extract all day-of-week columns
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    
    # Collect all month_day columns
    dow_cols = []
    for month in monthly_cols:
        for dow in dow_names:
            col_name = f'{month}_{dow}'
            if col_name in df.columns:
                dow_cols.append(col_name)
    
    print(f"\nDAY-OF-WEEK PATTERN:")
    print(f"  Found {len(dow_cols)} day-of-week columns")
    
    # Calculate average by day of week
    dow_data = {}
    for dow in dow_names:
        # All columns like Jan_Mon, Feb_Mon, Mar_Mon, etc.
        cols_for_dow = [f'{m}_{dow}' for m in monthly_cols if f'{m}_{dow}' in df.columns]
        
        # Average across all months for this day
        avg_parcels = df[cols_for_dow].mean(axis=1).mean()
        dow_data[dow] = avg_parcels
    
    for day, parcels in dow_data.items():
        pct_of_avg = 100 * parcels / np.mean(list(dow_data.values()))
        print(f"  {day}: {parcels:,.0f} avg parcels ({pct_of_avg:.0f}% of avg)")
    
    peak_day = max(dow_data, key=dow_data.get)
    trough_day = min(dow_data, key=dow_data.get)
    
    print(f"\n  Peak day: {peak_day} ({100*dow_data[peak_day]/np.mean(list(dow_data.values())):.0f}% of avg)")
    print(f"  Trough day: {trough_day} ({100*dow_data[trough_day]/np.mean(list(dow_data.values())):.0f}% of avg)")
    
    # ========================================
    # AGGREGATION 4: Cell-Level Day Statistics
    # ========================================
    
    # For each cell, calculate statistics by day of week
    for dow in dow_names:
        cols_for_dow = [f'{m}_{dow}' for m in monthly_cols if f'{m}_{dow}' in df.columns]
        df[f'avg_{dow}_parcels'] = df[cols_for_dow].mean(axis=1)
    
    # Peak and offpeak days per cell
    dow_avg_cols = [f'avg_{dow}_parcels' for dow in dow_names]
    df['peak_day_parcels'] = df[dow_avg_cols].max(axis=1)
    df['offpeak_day_parcels'] = df[dow_avg_cols].min(axis=1)
    df['dow_variability'] = df['peak_day_parcels'] / df['offpeak_day_parcels']
    
    print(f"\nCELL-LEVEL VARIABILITY:")
    print(f"  Avg day-of-week ratio (peak/offpeak): {df['dow_variability'].mean():.2f}x")
    print(f"  Min: {df['dow_variability'].min():.2f}x")
    print(f"  Max: {df['dow_variability'].max():.2f}x")
    
    # ========================================
    # AGGREGATION 5: Monthly Statistics
    # ========================================
    
    for month in monthly_cols:
        df[f'{month}_dow_cols'] = [
            f'{month}_{dow}' for dow in dow_names
        ]
    
    print(f"\nSUMMARY:")
    print(f"  Cells: {len(df)}")
    print(f"  Annual demand range: {df['annual_parcels'].min():,.0f} to {df['annual_parcels'].max():,.0f}")
    print(f"  Cities: {df['city_name'].unique().tolist()}")
    
    return {
        'df': df,
        'monthly_cols': monthly_cols,
        'dow_names': dow_names,
        'monthly_pattern': monthly_pattern,
        'dow_data': dow_data,
        'peak_month': peak_month,
        'trough_month': trough_month,
        'peak_day': peak_day,
        'trough_day': trough_day
    }


# ============================================================================
# PART 2: VISUALIZATIONS
# ============================================================================

def visualize_temporal_demand_CORRECTED(demand_dict):
    """Create 4-panel visualization of temporal patterns."""
    
    df = demand_dict['df']
    monthly_pattern = demand_dict['monthly_pattern']
    dow_data = demand_dict['dow_data']
    monthly_cols = demand_dict['monthly_cols']
    dow_names = demand_dict['dow_names']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Monthly Seasonality
    ax = axes[0, 0]
    colors = ['red' if m == demand_dict['peak_month'] else 'steelblue' 
              for m in monthly_cols]
    ax.bar(monthly_cols, monthly_pattern.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(monthly_pattern.mean(), color='orange', linestyle='--', linewidth=2, label='Average')
    ax.set_ylabel('Average Parcels/Cell/Month', fontsize=11, fontweight='bold')
    ax.set_title('Monthly Seasonality Pattern', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # 2. Day-of-Week Pattern
    ax = axes[0, 1]
    colors_dow = ['orange' if d == demand_dict['peak_day'] else 
                  'lightcoral' if d == demand_dict['trough_day'] else 'steelblue'
                  for d in dow_names]
    ax.bar(dow_names, [dow_data[d] for d in dow_names], color=colors_dow, edgecolor='black', linewidth=1.5)
    ax.axhline(np.mean(list(dow_data.values())), color='green', linestyle='--', linewidth=2, label='Average')
    ax.set_ylabel('Average Parcels/Cell/Day', fontsize=11, fontweight='bold')
    ax.set_title('Day-of-Week Pattern', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # 3. Cell Variability Distribution
    ax = axes[1, 0]
    ax.hist(df['dow_variability'], bins=30, color='green', edgecolor='black', alpha=0.7)
    ax.axvline(df['dow_variability'].mean(), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {df["dow_variability"].mean():.2f}x')
    ax.set_xlabel('Peak/Offpeak Ratio (Variability)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Cells', fontsize=11, fontweight='bold')
    ax.set_title('Day-of-Week Variability Across Cells', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Peak vs Offpeak Distribution
    ax = axes[1, 1]
    bp = ax.boxplot([df['peak_day_parcels'], df['offpeak_day_parcels']], 
                     labels=['Peak Day (Avg)', 'Offpeak Day (Avg)'],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['orange', 'lightblue']):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Parcels/Cell/Day', fontsize=11, fontweight='bold')
    ax.set_title('Peak vs Offpeak Demand Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('temporal_demand_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved: temporal_demand_analysis.png")


# ============================================================================
# PART 3: CREATE PEAK/OFFPEAK DATASETS FOR CLUSTERING
# ============================================================================

def create_peak_offpeak_datasets(demand_dict):
    """
    Create separate demand datasets for peak and offpeak clustering.
    
    Peak = Friday (typically highest demand day)
    Offpeak = Saturday (typically lowest demand day)
    
    In December (peak month) vs July (offpeak month)
    """
    
    df = demand_dict['df']
    peak_day = demand_dict['peak_day']
    trough_day = demand_dict['trough_day']
    peak_month = demand_dict['peak_month']
    trough_month = demand_dict['trough_month']
    
    # Peak scenario: Peak month + peak day of week
    # Usually Dec_Fri
    df['peak_period_parcels'] = df[f'{peak_month}_{peak_day}']
    
    # Offpeak scenario: Trough month + offpeak day of week
    # Usually Jul_Sat
    df['offpeak_period_parcels'] = df[f'{trough_month}_{trough_day}']
    
    # Create separate CSVs for clustering
    df_peak = df[['de_grid_id', 'peak_period_parcels']].copy()
    df_peak.columns = ['de_grid_id', 'expected_parcels']
    df_peak['period'] = f'Peak ({peak_month}_{peak_day})'
    
    df_offpeak = df[['de_grid_id', 'offpeak_period_parcels']].copy()
    df_offpeak.columns = ['de_grid_id', 'expected_parcels']
    df_offpeak['period'] = f'Offpeak ({trough_month}_{trough_day})'
    
    print(f"\nPEAK vs OFFPEAK SCENARIOS:")
    print(f"  Peak ({peak_month}_{peak_day}) total: {df_peak['expected_parcels'].sum():,.0f}")
    print(f"  Offpeak ({trough_month}_{trough_day}) total: {df_offpeak['expected_parcels'].sum():,.0f}")
    print(f"  Ratio: {df_peak['expected_parcels'].sum() / df_offpeak['expected_parcels'].sum():.2f}x")
    
    df_peak.to_csv('demand_peak_period.csv', index=False)
    df_offpeak.to_csv('demand_offpeak_period.csv', index=False)
    
    print(f"\nSaved:")
    print(f"  demand_peak_period.csv ({len(df_peak)} cells)")
    print(f"  demand_offpeak_period.csv ({len(df_offpeak)} cells)")
    
    return df_peak, df_offpeak


# ============================================================================
# PART 4: WORKFORCE SCHEDULING OPTIMIZATION
# ============================================================================

def calculate_staffing_levels_CORRECTED(demand_dict, baseline_staff=100):
    """
    Calculate optimal staffing by day of week.
    
    Args:
        demand_dict: Output from process_temporal_demand_CORRECTED()
        baseline_staff: Staff needed for average day
    """
    
    dow_data = demand_dict['dow_data']
    dow_names = demand_dict['dow_names']
    
    avg_demand = np.mean(list(dow_data.values()))
    
    staffing = {}
    print(f"\nSTAFFING OPTIMIZATION (baseline={baseline_staff} for avg day):")
    print(f"  Average daily demand: {avg_demand:,.0f} parcels")
    
    for day in dow_names:
        demand = dow_data[day]
        staff_needed = int(baseline_staff * demand / avg_demand)
        pct_change = 100 * (staff_needed - baseline_staff) / baseline_staff
        staffing[day] = staff_needed
        
        print(f"  {day}: {staff_needed} staff ({pct_change:+.0f}%)")
    
    # Cost analysis
    annual_shifts = 52 * 6  # 52 weeks, 6 working days (Mon-Sat)
    hourly_wage = 15  # EUR/hour
    hours_per_shift = 8
    shifts_per_day = 3  # Morning, afternoon, night
    
    cost_baseline = baseline_staff * annual_shifts * shifts_per_day * hours_per_shift * hourly_wage
    cost_optimized = sum(staffing.values()) * shifts_per_day * hours_per_shift * hourly_wage / 6 * annual_shifts / 52
    
    print(f"\nCOST IMPACT:")
    print(f"  Baseline (constant {baseline_staff}): EUR {cost_baseline:,.0f}/year")
    print(f"  Optimized (variable): EUR {cost_optimized:,.0f}/year")
    print(f"  Savings: EUR {cost_baseline - cost_optimized:,.0f}/year ({100*(cost_baseline-cost_optimized)/cost_baseline:.1f}%)")
    
    return staffing


# ============================================================================
# MAIN: RUN COMPLETE ANALYSIS
# ============================================================================

if __name__ == '__main__':
    
    # 1. Process your temporal demand data
    print("="*70)
    print("PROCESSING TEMPORAL DEMAND DATA")
    print("="*70)
    
    demand_dict = process_temporal_demand_CORRECTED('data/parcel_demand_by_age.csv')
    
    # 2. Visualize patterns
    print("\n" + "="*70)
    print("VISUALIZING PATTERNS")
    print("="*70)
    
    visualize_temporal_demand_CORRECTED(demand_dict)
    
    # 3. Create peak/offpeak datasets for clustering
    print("\n" + "="*70)
    print("CREATING PEAK/OFFPEAK DATASETS")
    print("="*70)
    
    df_peak, df_offpeak = create_peak_offpeak_datasets(demand_dict)
    
    # 4. Calculate staffing optimization
    print("\n" + "="*70)
    print("STAFFING OPTIMIZATION")
    print("="*70)
    
    staffing = calculate_staffing_levels_CORRECTED(demand_dict, baseline_staff=100)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use demand_peak_period.csv for peak clustering")
    print("  2. Use demand_offpeak_period.csv for offpeak clustering")
    print("  3. Compare results to identify dynamic capacity cells")
    print("  4. Visualizations saved: temporal_demand_analysis.png")