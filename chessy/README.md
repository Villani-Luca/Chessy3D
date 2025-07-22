# CHESSY3D

Authors:
- Davide Della Casa Venturelli
- Luca Villani
- Alessandro Mezzogori

## Setup

1. Setup server postgres PostgreSQL 17
   1. install postgres extension pgvector to support efficient storage of the embeddings https://github.com/pgvector/pgvector?tab=readme-ov-file
   2. create an empty database
   3. restore the following backup https://drive.google.com/file/d/1SWYzWhkMFzg8yCRK40U0ezMAgDsi4Nlu/view?usp=sharing
   4. NOTE: if the database does not align with the following connection string "host=localhost user=postgres password=password dbname=chessy" it is possibile to modify it by changing the 'pgconn' inside the args dictionary of main.py  (line 209)
2. install requirements.txt
   1. separate installation of torch with cuda 
   ```sh
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. application start
   1. activate virtual environment if used
   2. navigate to project root folder
   3. set environment variable PYTHONPATH="<path to project root folder (contains main.py)>" \ 
   ```powershell
   #powershell example
   $env:PYTHONPATH="E:\projects\uni\chessy"
   ```
   4. execute main.py script

## NOTE
The project contains the minimum amount of code possible to be able to run, we 
skimmed off all the historical tests and tries, all the unused .pt and all the codes that 
was used to create the training datasets and retrieval database.