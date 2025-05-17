from ultralytics import YOLO

def train():
    # Load a model
    yolo_model = YOLO("yolo11n-pose.pt")

    # Update keypoint shape
    # yolo_model.model.args["kpt_shape"] = [4, 3]

    # Train the model
    results = yolo_model.train(
        data="src/chessboard_localization/keypoint_approach/data.yaml",
        epochs=10,  # 100
        batch=-1,
        imgsz=640,
    )

def validate():
    model = YOLO("runs/pose/train/weights/best.pt") 

    # Validate on test set
    metrics = model.val(
        data="src/chessboard_localization/keypoint_approach/data.yaml",
        split="test",
        imgsz=640,
        conf=0.5,
        iou=0.5, 
        device=0
    )

    print(metrics.box.map)
    print(metrics.pose.map)


if __name__ == "__main__":
    train()
    validate()