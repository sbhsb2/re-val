from connection.config import read_db_config

import pandas as pd
import psycopg2

db_config = read_db_config("postgis")

connection = psycopg2.connect(
    dbname=db_config["dbname"],
    user=db_config["user"],
    password=db_config["password"],
    host=db_config["host"],
    port=db_config["port"]
)
cursor = connection.cursor()

# Konfiguration laden
db_config = read_db_config("postgis")

try:
    # Verbindung aufbauen
    connection = psycopg2.connect(
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"]
    )
    
    # Test-Abfrage mit Pandas (bequemer für SELECT)
    # Hinweis: Prüfe, ob das Schema wirklich 'anayltis' (Tippfehler?) oder 'analysis' heißt.
    query = """SELECT de_grid_id, geometry_3035, population FROM analytics.grid_metrics_1km 
        WHERE city_name LIKE 'Bremen%' LIMIT 500;"""
    
    df = pd.read_sql_query(query, connection)
    
    print("Verbindung erfolgreich! Hier sind die ersten Zeilen:")
    print(df.head())

except Exception as e:
    print(f"Fehler bei der Verbindung oder Abfrage: {e}")

finally:
    if 'connection' in locals() and connection:
        connection.close()
        print("Datenbankverbindung geschlossen.")

