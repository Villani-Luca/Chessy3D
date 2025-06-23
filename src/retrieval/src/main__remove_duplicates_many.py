'''
removes duplicates created from load_milvus
'''
import pathlib
import time

import numpy as np
import pandas as pd

from src.retrieval.src.pgsql import Connection

ROOT = pathlib.Path.cwd().parent.parent.parent
PARQUEET_FOLDER = (ROOT / 'data/retrieval/parqueet')
PARQUEET_OUT_FOLDER = (ROOT / 'data/retrieval/parqueet_out')
OUTPUT_BATCH_SIZE = 200_000
PG_CONN = r"host=localhost user=postgres password=password dbname=chessy"
STARTING_INDEX_OUTPUT = 0

def notfound(series: pd.Series):
    values = list(series)

    not_in_db = conn.cursor.execute(r"SELECT embeddingid from dedup WHERE embeddingid = ANY(%s)", [values]).fetchall()
    # not_in_db_list = set([x[0].strip() for x in not_in_db])
    not_in_db_list = set([x[0] for x in not_in_db])
    return ~series.isin(not_in_db_list)

def savepg(ids: list):
    # conn.cursor.execute("""CREATE TEMPORARY TABLE temp_dedup AS SELECT * from dedup LIMIT 0""")
    # with conn.cursor.copy("""COPY temp_dedup (embeddingid) FROM STDIN""") as copy:
    #     for item in ids:
    #         copy.write_row((item,))
    #
    # conn.cursor.execute("""INSERT INTO dedup (embeddingid)
    #                        SELECT embeddingid
    #                        FROM temp_dedup
    #                        ON CONFLICT DO NOTHING""")
    # conn.cursor.execute("""DROP TABLE temp_dedup""")

    start = time.time()
    with conn.cursor.copy("""COPY dedup (embeddingid) FROM STDIN""") as copy:
        for item in ids:
            copy.write_row((item,))
    print(f'SAVEPG DURATION {time.time() - start} seconds')

def save(df: pd.DataFrame, current_index: int):
    if df.shape[0] < OUTPUT_BATCH_SIZE:
        return current_index

    start = time.time()
    print(f'Saving {df.shape[0]} rows')
    idx = current_index

    df = df.drop_duplicates(subset=['id'])
    for chunk in [df.iloc[i:i + OUTPUT_BATCH_SIZE] for i in range(0, len(df), OUTPUT_BATCH_SIZE)]:
        pd.DataFrame(chunk).to_parquet((PARQUEET_OUT_FOLDER / f'_{idx}.parquet'), index=False)
        idx += 1

    unsigned_hashes = df['id'].to_numpy().astype(np.uint64)
    signed_hashes = unsigned_hashes.astype(np.int64).tolist()

    conn.begin()
    savepg(signed_hashes)
    conn.commit()
    print(f'Saved {df.shape[0]} rows in {time.time() - start} seconds [hashes {unsigned_hashes.shape[0]}]')

    return idx


if __name__ == '__main__':
    #files = [f for f in filter(filter_func, PARQUEET_FOLDER.glob('*.parquet'))]
    #dt = pd.concat(pd.read_parquet(parquet_file) for parquet_file in files)
    conn = Connection(PG_CONN)

    current_df = pd.DataFrame()
    index = 0

    for parquet in PARQUEET_FOLDER.glob('*.parquet'):
        current = time.time()
        print(f'Processing {parquet.name} - current index: {index}')
        parquet_df = pd.read_parquet(parquet)

        # convert to 64bit unsigned
        temp = parquet_df.copy()

        unsigned_hashes = parquet_df['id'].to_numpy().astype(np.uint64)
        signed_hashes = pd.Series(unsigned_hashes.astype(np.int64), dtype=np.int64)

        filtered_rows = notfound(signed_hashes)
        not_found_db = parquet_df[filtered_rows]
        not_found_hashes = signed_hashes[filtered_rows]

        print(f'Removed {parquet_df.shape[0] - not_found_db.shape[0]} rows - {time.time() - current} seconds')
        current_df = pd.concat([current_df, not_found_db])

        new_index = save(current_df, index)
        if index != new_index:
            current_df.drop(current_df.index,inplace=True)
            index = new_index

        print(f'Done processing {parquet.name} - current index: {index} - {time.time() - current}')

