import pathlib
import time

import pandas as pd
from pgvector.psycopg import Bit

from src.retrieval.src.pgsql import Connection

parquet_folder = pathlib.Path(r"D:\Projects\Uni\Chessy3D\data\retrieval\parqueet_out")

conn = Connection(r"host=localhost user=postgres password=password dbname=chessy")

for parquet in parquet_folder.glob("*.parquet"):
    current = time.time()
    df = pd.read_parquet(parquet)

    conn.begin()
    with conn.cursor.copy("""COPY naivevectors (embeddingid, embedding) FROM STDIN""") as copy:
        for index, row in df.iterrows():
            id = int(row['id'])
            id8 = id if id <= 9223372036854775807 else id - 18446744073709551616

            copy.write_row((id8, Bit(row['vector']).to_text()))
    conn.commit()

    print(f"{parquet.name} {time.time() - current:.2f} seconds")

print("DONE")
