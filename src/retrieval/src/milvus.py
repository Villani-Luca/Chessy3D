import pymilvus as milvus
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType

COLLECTION_NAME = 'positions'

class MilvusRepository:
    def __init__(self, url = 'http://localhost:19530', collection = COLLECTION_NAME):
        """
        :param url: url to milvus, defaults to http://localhost:19530
        :param collection: collection name, defaults to 'positions'
        """

        self.collection = collection
        self.client = milvus.MilvusClient(url)

    def load_collection(self):
        self.client.load_collection(self.collection)

    def search_embeddings(self, embeddings: list[int] | list[list[int]], limit = 5):
        return self.client.search(collection_name=self.collection, data=embeddings, limit=limit)
        # TODO: convertire il risultato a qualcosa di piú facilmente utilizzabile

    def get_embedding(self, ids: list[str]):
        return self.client.get(
            collection_name=self.collection,
            ids = ids,
            output_fields=["vector"]
        )

    def close(self):
        self.client.close()


class MilvusBulkWriter:
    writer: RemoteBulkWriter

    def __init__(self,
                 access_key: str = 'minioadmin',
                 secret_key: str = 'minioadmin',
                 bucket_name: str = 'a-bucket',
                 endpoint: str = 'localhost:9000',
                 secure: bool = False,
                 collection = COLLECTION_NAME):
        """
        :param access_key: minio access key, defaults to 'minioadmin'
        :param secret_key: minio secret key, defaults to 'minioadmin'
        :param bucket_name: minio bucket name, defaults to 'a-bucket'
        :param endpoint: minio endpoint, defaults to http://localhost:9000
        :param secure: whether to use secure connection, defaults to False
        :param collection: collection name, defaults to 'positions'
        """

        conn  = RemoteBulkWriter.S3ConnectParam(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name,
            secure=secure
        )
        schema = MilvusSetup.milvus_collection_schema(collection)
        self.writer = RemoteBulkWriter(schema=schema, remote_path='/', connect_param=conn, file_type=BulkFileType.PARQUET)

    def append(self, data: list[tuple[str, list[int]]]):
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
    def milvus_collection_schema(collection=COLLECTION_NAME, vector_size=768):
        """
        :param collection: collection name, defaults to 'positions'
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
    def setup_milvus(reset=False, collection=COLLECTION_NAME, vector_size=768):
        """
        :param reset: deletes <collection>, defaults to False
        :param collection: collection name, defaults to 'positions'
        """

        mclient = milvus.MilvusClient()

        if reset and mclient.has_collection(collection_name=collection):
            mclient.drop_collection(collection_name=collection)

        schema = MilvusSetup.milvus_collection_schema(collection, vector_size)
        mclient.create_collection(collection_name=collection, schema=schema)

        index_params = milvus.MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="vector_index",
            params={"nlist": 128}
        )

        # 4.3. Create an index file
        mclient.create_index(
            collection_name=collection,
            index_params=index_params,
            sync=False  # Whether to wait for index creation to complete before returning. Defaults to True.
        )

        mclient.close()