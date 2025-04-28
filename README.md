# Chessy3D

The project is structure in the following modules:

1. Chessboard localization inside the image.
    - Provides a list of found chessboards, containing the coordinates of the 4 corners.
    - Provides information about the camera/scene.
2. Chessboard cells localization inside the chessboard.
    - For every valid chessboard provides a list of cells, containing the coordinates of the 4 cell corners and the color of che cell.
3. Chess pieces detection and localization inside the image.
   - Find all chess pieces and provides information about the type, the color and the bounding box.
4. Mapping detected chess pieces inside the correct chessboard cells.
   - Based on the camera/scene conditions.
5. Chessboard reference detection and creation of the state in the FEN notation.
6. Retrieval of chess games with a similar state.
7. UI for the visualization of the results.