from gc import freeze

from ultralytics import YOLO
import wandb

# Load model
if __name__ == '__main__':
    model = YOLO("yolov8m.pt")
    #freeze_layers = 21 # or 18 pre neck in name:

    wandb.login(key="ec8527bc43fcbc7aef5dfa1fa83e93e0cab69ce8")

    # Train
    model.train(
        data=r"C:\Users\villa\Downloads\DatasetsPerTrainig\mergedDataset\data.yaml",
        epochs=600,
        batch=32,
        imgsz=640,
        patience=100,
        project="chess_finetuning_model",
        name="mergedDataset_frompretrainedmodel",
        cache=False,
        resume=False
        #freeze=[x for x in range(freeze_layers + 1)],
    )

