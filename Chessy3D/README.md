# CHESSY3D

Authors:
- Davide Della Casa Venturelli
- Luca Villani
- Alessandro Mezzogori

## Setup

1. Setup server postgres PostgreSQL 17
   1. install postgres extension pgvector to support efficient storage of the embeddings https://github.com/pgvector/pgvector?tab=readme-ov-file
   2. create an empty database
   3. restore the following backup https://drive.google.com/file/d/1SWYzWhkMFzg8yCRK40U0ezMAgDsi4Nlu/view?usp=sharing (~15min)
   4. NOTE: if the database does not align with the following connection string "host=localhost user=postgres password=password dbname=chessy" it is possibile to modify it by changing the 'pgconn' inside the args dictionary of main.py  (line 209)
2. install requirements.txt
   1. separate installation of torch with cuda 
   ```sh
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. application start
   1. activate virtual environment if used
   2. navigate to "chessy" folder
   3. set environment variable PYTHONPATH="<path to project root folder "Chessy3D" (contains readme.md) >"
   ```powershell
   #powershell example
   $env:PYTHONPATH="<path to extracted Zip folder ("Chessy3D")>"
   ```
   4. execute main.py script

## Application usage
The application is divided in 3 subsections:
- image to the left
- chessboard to the right
- retrieval datagrid to the bottom

to start drag and drop and image, there are test images both bad and good inside the "test_images" folder, 
this will populate if a chessboard is found all the tabs with the respective stage of the pipeline.
If no image is found most of the tabs will be set to the original image.

After a chessboard is extracted the right hand side will rerender with the chess state, **this should be oriented from
white to black** using the buttons in the bottom left that rotate the position ( this step is needed to build the embedding correctly ).
Beside the rotations buttons the fen representation can be copied from the textbox or the dedicated button.

After rotating the chessboard by submitting a refresh request the database will be queried to retrieve the top 5 games that contain a similar position.
it is possible that the same game will be returned if the positions in it are the best ranked.

By double clicking on of the returned rows the position of the retrieved game will be shown, with its relative fen, to go back click on the green arrow

## NOTE
The project contains the minimum amount of code possible to be able to run, we 
skimmed off all the historical tests and tries, all the unused .pt and all the codes that 
was used to create the training datasets and retrieval database.