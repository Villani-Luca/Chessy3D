from ultralytics import YOLO
import ChessPiece
import matplotlib.pyplot as plt

def detect_objects(image):
    # model = YOLO("chess-model-yolov8m.pt")
    model = YOLO("D:\\CodeProjects\\University\\Chessy3D\\src\\object_detection_yolo\\runs\\detect\\train3\w\eights\\best.pt")
    results = model(image)
    cps = []

    for box in results[0].boxes:
        xyxy = box.xyxy[0]
        xyxy_n = box.xyxyn[0]
        color_name = results[0].names[int(box.cls[0])].split("-")
        cp = ChessPiece.ChessPiece(
            x_min=xyxy[0],
            y_min=xyxy[1],
            x_max=xyxy[2],
            y_max=xyxy[3],
            class_number=box.cls[0],
            xn_min=xyxy_n[0],
            yn_min=xyxy_n[1],
            xn_max=xyxy_n[2],
            yn_max=xyxy_n[3],
            class_name=color_name[1],
            class_confidence=box.conf[0],
            class_color=color_name[0],
            xyxy_box_position=box.xyxy[0],
            xyxy_n_box_position=box.xyxyn[0]
        )
        cps.append(cp)

    im_array = results[0].plot()  # plot a BGR numpy array of predictions

    plt.imshow(im_array)

    return cps