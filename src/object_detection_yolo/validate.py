from gc import freeze
from pathlib import Path

from ultralytics import YOLO
import wandb

# Load model
if __name__ == '__main__':
    chessred2k = r"D:\CodeProjects\University\Chessy3d\Chessy3D\data\datachess_redYolo\data.yaml"
    models = [
        ("validation/primaRunYolo", r"C:\Users\villa\Desktop\PTs\01_primaRunYolo.pt"),
        ("validation/chessPieceDetection", r"C:\Users\villa\Desktop\PTs\03_datasetnew_2706.pt"),
        ("validation/mergedfromScratch", r"C:\Users\villa\Desktop\PTs\04_3datasetmergedfromscratch.pt"),
        ("validation/mergedfromPreTrained", r"C:\Users\villa\Desktop\PTs\05_3dataset_merged_frompretrained.pt"),
        ("validation/model", r"C:\Users\villa\Desktop\PTs\model.pt")
    ]

    for (project, model_path) in models:
        path = Path(model_path)
        model = YOLO(model_path)
        result = model.val(data=chessred2k, project=project)
        print(result.results_dict)
