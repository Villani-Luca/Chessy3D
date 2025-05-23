import numpy as np
import pymilvus as milvus
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType

NAIVE_COLLECTION_NAME = 'naive_embeddings'
AE_COLLECTION_NAME = 'ae_embeddings'

class MilvusRepository:
    def __init__(self, url = 'http://localhost:19530', collection = NAIVE_COLLECTION_NAME):
        """
        :param url: url to milvus, defaults to http://localhost:19530
        :param collection: collection name, defaults to 'positions'
        """

        self.collection = collection
        self.client = milvus.MilvusClient(url)

    def load_collection(self):
        self.client.load_collection(self.collection)

    def search_embeddings(self, embeddings: np.ndarray | list[np.ndarray], limit = 5):
        data = [x.tolist() for x in ([embeddings] if isinstance(embeddings, np.ndarray) else embeddings)]

        return self.client.search(
            collection_name=self.collection,
            data=data,
            limit=limit,
            anns_field="vector",
        )

    def get_embedding(self, ids: list[str]):
        return self.client.get(
            collection_name=self.collection,
            ids = ids,
            output_fields=["vector"]
        )

    def close(self):
        self.client.close()


class MilvusBulkWriter:
    writer: LocalBulkWriter

    def __init__(self,
                 local_path: str = '/',
                 collection = NAIVE_COLLECTION_NAME):
        """
        :param access_key: minio access key, defaults to 'minioadmin'
        :param secret_key: minio secret key, defaults to 'minioadmin'
        :param bucket_name: minio bucket name, defaults to 'a-bucket'
        :param endpoint: minio endpoint, defaults to http://localhost:9000
        :param secure: whether to use secure connection, defaults to False
        :param collection: collection name, defaults to 'positions'
        """

        schema = MilvusSetup.milvus_naive_embedding() if collection == NAIVE_COLLECTION_NAME else MilvusSetup.milvus_ae_embedding()
        self.writer = LocalBulkWriter(schema=schema, local_path=local_path, file_type=BulkFileType.PARQUET, chunk_size=512*1024*1024)

    def append(self, data: list[tuple[str, np.ndarray]]):
        for item in data:
            self.writer.append_row({
                'id': item[0],
                'vector': item[1],
            })

    def commit(self):
        self.writer.commit()

    def batch_files(self):
        return self.writer.batch_files

class MilvusSetup:
    @staticmethod
    def milvus_naive_embedding(vector_size=768):
        """
        :param vector_size: vector size, defaults to 768
        """
        schema = milvus.MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=milvus.DataType.VARCHAR,
            max_length=20,
            # highlight-start
            is_primary=True,
            auto_id=False,
            # highlight-end
        )
        schema.add_field(
            field_name="vector",
            datatype=milvus.DataType.FLOAT_VECTOR,
            # highlight-next-line
            dim=vector_size
        )
        schema.verify()
        return schema

    @staticmethod
    def milvus_ae_embedding(vector_size=768):
        """
        :param vector_size: vector size, defaults to 768
        """
        schema = milvus.MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=milvus.DataType.VARCHAR,
            max_length=20,
            # highlight-start
            is_primary=True,
            auto_id=False,
            # highlight-end
        )
        schema.add_field(
            field_name="vector",
            datatype=milvus.DataType.FLOAT_VECTOR,
            # highlight-next-line
            dim=vector_size
        )
        schema.verify()
        return schema

    @staticmethod
    def setup_milvus(reset=False):
        """
        :param reset: deletes <collection>, defaults to False
        """

        mclient = milvus.MilvusClient()

        if reset:
            if mclient.has_collection(collection_name=NAIVE_COLLECTION_NAME):
                mclient.drop_collection(collection_name=NAIVE_COLLECTION_NAME)

            if mclient.has_collection(collection_name=AE_COLLECTION_NAME):
                mclient.drop_collection(collection_name=AE_COLLECTION_NAME)


        mclient.create_collection(collection_name=NAIVE_COLLECTION_NAME, schema=MilvusSetup.milvus_naive_embedding())
        mclient.create_collection(collection_name=AE_COLLECTION_NAME, schema=MilvusSetup.milvus_ae_embedding())

        naive_index_params = milvus.MilvusClient.prepare_index_params()
        naive_index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="naive_vector_index",
            params={"nlist": 128}
        )

        # 4.3. Create an index file
        mclient.create_index(
            collection_name=NAIVE_COLLECTION_NAME,
            index_params=naive_index_params,
            sync=False  # Whether to wait for index creation to complete before returning. Defaults to True.
        )

        index_params = milvus.MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="ae_vector_index",
            params={"nlist": 128}
        )

        # 4.3. Create an index file
        mclient.create_index(
            collection_name=AE_COLLECTION_NAME,
            index_params=index_params,
            sync=False  # Whether to wait for index creation to complete before returning. Defaults to True.
        )

        mclient.close()