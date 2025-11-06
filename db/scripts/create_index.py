import duckdb
import os

# Define the database file name
db_file = '../carciscan.db'

# Delete the old database file if it exists to start fresh
if os.path.exists(db_file):
    os.remove(db_file)

# Connect to a persistent database file
con = duckdb.connect(database=db_file)

# Import the Parquet files into native DuckDB tables
print("Importing Parquet files...")
con.execute("CREATE TABLE smiles AS SELECT * FROM '../smiles.parquet';")
con.execute("CREATE TABLE synonyms AS SELECT * FROM '../synonyms.parquet';")
con.execute("CREATE TABLE t3db AS SELECT * FROM '../t3db.parquet';")

# Create indexes on the columns used for searching and joining
print("Creating indexes... (This may take a moment)")
# Index for the WHERE clause
con.execute("CREATE INDEX idx_synonyms_name ON synonyms(Synonyms);")
# Index for the JOIN condition
con.execute("CREATE INDEX idx_synonyms_cid ON synonyms(CID);")
con.execute("CREATE INDEX idx_smiles_cid ON smiles(CID);")
con.execute("CREATE INDEX idx_t3db_cid ON t3db(CID);")

print("Setup complete! Database 'carciscan.db' is ready.")

con.close()