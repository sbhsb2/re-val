import numpy as np
import pandas as pd
import ast

def safe_parse(val):
    try:
        return ast.literal_eval(val)
    except:
        return {}

def data_trans(df):
    selected_df = df[["de_grid_id", "population","avg_age", "city_name","age_distribution","geometry_4326"]]
    selected_df = selected_df.dropna(subset=["population"])
    
    selected_df['age_parsed'] = selected_df['age_distribution'].apply(safe_parse)
    age_expanded = selected_df['age_parsed'].apply(pd.Series)

    rename_map = {
        'unter10': 'under_10',
        'a10bis19': '10_19',
        'a20bis29': '20_29',
        'a30bis39': '30_39',
        'a40bis49': '40_49',
        'a50bis59': '50_59',
        'a60bis69': '60_69',
        'a70bis79': '70_79',
        'a80undaelter': '80_plus'
    }
    age_expanded = age_expanded.rename(columns=rename_map)
    selected_df = selected_df.join(age_expanded) 
    return selected_df



def annual_pd(selected_df):

    parcel_levels = np.array([20,28,39])

    def expected_parcels_from_probs(probabilities):
        return np.sum(parcel_levels * probabilities)
    
    age_probabilities = {
        '16-20': np.array([0.27, 0.41, 0.32]),
        '21-30': np.array([0.23, 0.35, 0.42]),
        '31-40': np.array([0.16, 0.36, 0.49]),
        '41-50': np.array([0.18, 0.43, 0.39]),
        '51-60': np.array([0.23, 0.44, 0.33]),
        '61+':   np.array([0.31, 0.43, 0.26])
        }
    
    def parcels_from_age(age):
        if age < 21:
            probs = age_probabilities['16-20']
        elif age < 31:
            probs = age_probabilities['21-30']
        elif age < 41:
            probs = age_probabilities['31-40']
        elif age < 51:
            probs = age_probabilities['41-50']
        elif age < 61:
            probs = age_probabilities['51-60']
        else:
            probs = age_probabilities['61+']

        return expected_parcels_from_probs(probs)

    results = []

    for idx, row in selected_df.iterrows():
        if row['age_parsed'] == {}:
            results.append(row['population']* parcels_from_age(row['avg_age']))
            continue

        #young   = (row['under_10'] or 0) + (row['10_19'] or 0)
        young   = row['10_19']/2 
        a20_29  = row['20_29']
        a30_39  = row['30_39']
        a40_49  = row['40_49']
        a50_59  = row['50_59']
        elderly = (row['60_69']) + (row['70_79']) + (row['80_plus'])
        total_parcels = (
            young   * expected_parcels_from_probs(age_probabilities['16-20']) +
            a20_29  * expected_parcels_from_probs(age_probabilities['21-30']) +
            a30_39  * expected_parcels_from_probs(age_probabilities['31-40']) +
            a40_49  * expected_parcels_from_probs(age_probabilities['41-50']) +
            a50_59  * expected_parcels_from_probs(age_probabilities['51-60']) +
            elderly * expected_parcels_from_probs(age_probabilities['61+'])
        )

        results.append(total_parcels)

    selected_df['expected_parcels'] = results
    selected_df['expected_parcels_rounded'] = [round(x) for x in results]
    return selected_df

def monthly_daily_pd(selected_df):

    monthly_props = {
    'Jan': 0.076361645, 'Feb': 0.07506666, 'Mar': 0.078234065, 
    'Apr': 0.087409351, 'May': 0.077291486, 'Jun': 0.075869126, 
    'Jul': 0.078463341, 'Aug': 0.081966169, 'Sep': 0.0803485, 
    'Oct': 0.076323432, 'Nov': 0.0919142, 'Dec': 0.120752025
    }

    daily_props = {
        'Mon': 0.155, 'Tue': 0.2, 'Wed': 0.21, 
        'Thu': 0.215, 'Fri': 0.195, 'Sat': 0.025
    }

    for m_name, m_prop in monthly_props.items():
        selected_df[m_name] = round(selected_df['expected_parcels'] * m_prop)
        
        
        for d_name, d_prop in daily_props.items():
            selected_df[f"{m_name}_{d_name}"] = round(selected_df[m_name] * d_prop)

    return selected_df

def main():
    df= pd.read_csv('data/100m.csv')
    df = data_trans(df)
    df = annual_pd(df)
    selected_df_final = df[["de_grid_id", "city_name", "population", "avg_age","age_distribution", "expected_parcels"]]
    df = monthly_daily_pd(selected_df_final)
    df.to_csv('result/100m_with_parcels_7july.csv', index=False)
    

if __name__ == "__main__":
    main()