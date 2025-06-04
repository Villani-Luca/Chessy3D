'''
start milvus import jobs using minio uploaded files
'''

from pymilvus.bulk_writer import bulk_import
from src.retrieval.src.milvus import MilvusSetup, NAIVE_COLLECTION_NAME

N_FILES = 2047

url = f"http://127.0.0.1:19530"

MilvusSetup.setup_milvus(reset=True)

page_size = 10
max_page_run = 3
already_processed = 0

processed = already_processed
while processed < N_FILES:
    remaining = N_FILES - processed
    range_end = remaining if remaining <= page_size else page_size

    resp = bulk_import(
        url=url,
        collection_name=NAIVE_COLLECTION_NAME,
        files=[[f'_{processed + x + 1}.parquet'] for x in range(0, range_end)],
    )
    job_id = resp.json()['data']['jobId']
    print(job_id)

    processed += range_end
    if processed >= already_processed + max_page_run * page_size:
        break



