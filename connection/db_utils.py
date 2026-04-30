from connection.config import create_connection
import pandas as pd

def fetch_data(query):
    conn = create_connection()

    if conn is None:
        raise Exception("Database connection failed")

    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()
