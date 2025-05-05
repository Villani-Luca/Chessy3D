1. dati i file pgn, avviare lo split pgn passandogli i parametri
2. poi avviare il progetto c# per importare le partire su postgresql
3. avviare milvus process per caricare i file su minio e su postgresl delle mosse
    TODO: creare il local bulk writer 
4. scaricare i file di minio e avviare remove_duplicates
5. ricaricare i file su minio e avviare import