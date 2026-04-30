from connection.db_utils import fetch_data

query = """
SELECT de_grid_id, population
FROM analytics.grid_metrics_1km
LIMIT 5;
"""

df = fetch_data(query)

print("Data fetched successfully!")
print(df)
