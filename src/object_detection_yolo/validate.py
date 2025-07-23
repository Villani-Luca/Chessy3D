from gc import freeze
from pathlib import Path

from ultralytics import YOLO
import wandb

# Load model
if __name__ == '__main__':
    chessred2k = r"E:\projects\uni\Chessy3D\data\chessred_yolo\data.yaml"
    models = [
        ("validation/external", r"E:\projects\uni\Chessy3D\chessy\chesspiece_detection\model.pt"),
        ("validation/altro_nome", r"E:\projects\uni\Chessy3D\src\object_detection_yolo\ultimo.pt")
    ]

    for (project, model_path) in models:
        path = Path(model_path)
        model = YOLO(model_path)
        result = model.val(data=chessred2k, project=project)
        print(result)
