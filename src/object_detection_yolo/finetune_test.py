from gc import freeze

from ultralytics import YOLO

# Load model
if __name__ == '__main__':
    model = YOLO(r"D:\Projects\Uni\Chessy3D\src\object_detection_yolo\chess-model-yolov8m.pt")
    freeze_layers = 21 # or 18 pre neck in name:

    # Train
    model.train(
        data=r"D:\Projects\Uni\Chessy3D\data\roboflow-dataset\data.yaml",
        epochs=30,
        batch=16,
        freeze=[x for x in range(freeze_layers + 1)],
    )

    model.save(r"D:\Projects\Uni\Chessy3D\src\object_detection_yolo\freeze12-e20-roboflow.pt")
