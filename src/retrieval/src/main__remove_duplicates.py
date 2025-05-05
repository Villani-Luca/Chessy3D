'''
removes duplicates created from load_milvus
'''

import pathlib
import pandas as pd
import numpy as np

ROOT = pathlib.Path.cwd().parent.parent.parent
PARQUEET_FOLDER = (ROOT / 'data/retrieval/parqueet')
PARQUEET_OUT_FOLDER = (ROOT / 'data/retrieval/parqueet_out')
OUTPUT_BATCH_SIZE = 2000


if __name__ == '__main__':
    dt = pd.concat(pd.read_parquet(parquet_file) for parquet_file in PARQUEET_FOLDER.glob('*.parquet'))
    dt.drop_duplicates('id', inplace=True)


    for index, chunk in enumerate(np.array_split(dt, dt.shape[0] // OUTPUT_BATCH_SIZE)):
        pd.DataFrame(chunk).to_parquet((PARQUEET_OUT_FOLDER / f'_{index + 1}.parquet'))

