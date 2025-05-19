import sqlite3
import pandas as pd

# Path to SQLite database
db_path = "PPG_ACC_database.db"

# Connect to SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create a table for the dataset
cursor.execute("""
CREATE TABLE IF NOT EXISTS ppg_acc_data (
    PPG BLOB,
    ACC BLOB,
    HeartRate REAL,
    Activity INTEGER,
    SubjectID TEXT,
    Age INTEGER,
    Gender TEXT,
    Height INTEGER,
    Weight INTEGER,
    SkinType INTEGER,
    SportLevel INTEGER
)
""")

# Load the dataset in chunks to the SQL database
chunk_size = 10000  # Define a reasonable chunk size
csv_path = "PPG_ACC_processed_data/data.csv"  # Path to the large dataset CSV

for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    chunk.to_sql("ppg_acc_data", conn, if_exists="append", index=False)

print(f"Data successfully loaded into the SQLite database at {db_path}")

# Close the connection
conn.close()
