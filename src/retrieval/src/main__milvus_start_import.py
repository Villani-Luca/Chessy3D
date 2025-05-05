'''
start milvus import jobs using minio uploaded files
'''

from pymilvus.bulk_writer import bulk_import
from src.retrieval.src.milvus import MilvusSetup, NAIVE_COLLECTION_NAME

N_FILES = 106

url = f"http://127.0.0.1:19530"

MilvusSetup.setup_milvus()


resp = bulk_import(
    url=url,
    collection_name=NAIVE_COLLECTION_NAME,
    files=[[f'_{x + 1}.parquet'] for x in range(N_FILES)],
)

job_id = resp.json()['data']['jobId']
print(job_id)
