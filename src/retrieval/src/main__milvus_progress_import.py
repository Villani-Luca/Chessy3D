'''
script to check the status of the import jobs
'''

from pymilvus.bulk_writer import list_import_jobs, get_import_progress
from src.retrieval.src.milvus import NAIVE_COLLECTION_NAME
import json

url = f"http://127.0.0.1:19530"

resp = list_import_jobs(
    url=url,
    collection_name=NAIVE_COLLECTION_NAME,
)

print(json.dumps(resp.json(), indent=4))

resp = get_import_progress(
    url=url,
    job_id="457819205515459134",
)

js = resp.json()
total = len(js['data']['details'])
progress = len([1 for x in js['data']['details'] if x['state'] == 'InProgress'])
completed = len([1 for x in js['data']['details'] if x['state'] == 'Completed'])
failed = len([1 for x in js['data']['details'] if x['state'] == 'Failed'])

print(json.dumps(resp.json(), indent=4))
print(f'COMPLETED: {completed} / {total} PROGRESS: {progress} / {total} FAILED: {failed} / {total}')
