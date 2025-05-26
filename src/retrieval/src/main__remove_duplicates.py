'''
removes duplicates created from load_milvus
'''

import pathlib
import pandas as pd
import numpy as np

ROOT = pathlib.Path.cwd().parent.parent.parent
PARQUEET_FOLDER = (ROOT / 'data/retrieval/parqueet/0000')
PARQUEET_OUT_FOLDER = (ROOT / 'data/retrieval/parqueet_out')
OUTPUT_BATCH_SIZE = 2000
STARTING_INDEX_OUTPUT = 0

def filter_func(x: pathlib.Path):
    value = int(x.name.split('.')[0])
    return 0 <= value <= 200


if __name__ == '__main__':
    #files = [f for f in filter(filter_func, PARQUEET_FOLDER.glob('*.parquet'))]
    #dt = pd.concat(pd.read_parquet(parquet_file) for parquet_file in files)
    dt = pd.read_parquet(PARQUEET_FOLDER.as_posix())
    dt.drop_duplicates('id', inplace=True)


    for index, chunk in enumerate(np.array_split(dt, dt.shape[0] // OUTPUT_BATCH_SIZE)):
        pd.DataFrame(chunk).to_parquet((PARQUEET_OUT_FOLDER / f'_{index + 1 + STARTING_INDEX_OUTPUT}.parquet'), index=False)

