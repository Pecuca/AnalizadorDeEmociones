import sqlite3
import pandas as pd

conn = sqlite3.connect("app_data.db")

print("\n--- Personas registradas ---")
df_personas = pd.read_sql_query("SELECT * FROM personas;", conn)
print(df_personas)

print("\n--- Detecciones registradas ---")
df_det = pd.read_sql_query("SELECT * FROM detecciones;", conn)
print(df_det)

conn.close()