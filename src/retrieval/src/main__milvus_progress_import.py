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

r = resp.json()
job_ids = [(x["jobId"], x["state"], x["progress"]) for x in r["data"]["records"]]

for (job_id, state, job_progress) in job_ids:
    resp = get_import_progress(
        url=url,
        job_id=job_id,
    )

    js = resp.json()
    total = len(js['data']['details'])
    progress = len([1 for x in js['data']['details'] if x['state'] == 'InProgress'])
    completed = len([1 for x in js['data']['details'] if x['state'] == 'Completed'])
    failed = len([1 for x in js['data']['details'] if x['state'] == 'Failed'])
    pending = len([1 for x in js['data']['details'] if x['state'] == 'Pending'])
    importing = len([1 for x in js['data']['details'] if x['state'] == 'Importing'])

    print(f'JOB ID: {job_id} STATE: {state} PROGRESS: {job_progress}')
    print(f'COMPLETED: {completed} / {total} PROGRESS: {progress} / {total} FAILED: {failed} / {total} PENDING: {pending} / {total} IMPORTING: {importing} / {total}')
