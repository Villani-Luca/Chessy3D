'''
start milvus import jobs using minio uploaded files
'''

from pymilvus.bulk_writer import bulk_import
from src.retrieval.src.milvus import MilvusSetup, NAIVE_COLLECTION_NAME

N_FILES = 2047

url = f"http://127.0.0.1:19530"

MilvusSetup.setup_milvus()

processed = 0
while processed < N_FILES:
    remaining = N_FILES - processed
    range_end = remaining if remaining <= 1024 else 1024

    resp = bulk_import(
        url=url,
        collection_name=NAIVE_COLLECTION_NAME,
        files=[[f'_{processed + x + 1}.parquet'] for x in range(0, range_end)],
    )

    processed += range_end

job_id = resp.json()['data']['jobId']
print(job_id)
