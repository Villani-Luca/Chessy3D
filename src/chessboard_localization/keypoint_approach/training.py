from ultralytics import YOLO
import wandb
import cv2
import pyautogui

COLOR_PALETTE = [
    (255, 0, 0),  # Blue
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (0, 128, 255),  # Orange-ish
    (255, 128, 0),  # Light Orange
    (0, 255, 128),  # Mint
]

def display_image_cv2(image: cv2.typing.MatLike, max_height=1024, max_width=1024):
    window_name = "Image"

    img_height, img_width = image.shape[:2]

    scale_width = max_width / img_width
    scale_height = max_height / img_height

    scale_factor = min(scale_width, scale_height)

    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_width, new_height)
    cv2.imshow(window_name, image)

    screen_width, screen_height = pyautogui.size()

    pos_x = (screen_width - new_width) // 2
    pos_y = (screen_height - new_height) // 2

    cv2.moveWindow(window_name, pos_x, pos_y)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def train():
    #wandb.login(key="")

    # Load a model
    #yolo_model = YOLO("yolo11n-pose.pt")

    yolo_model = YOLO("chessy3D/chessboard_localization2/weights/last.pt")

    # Train the model
    results = yolo_model.train(
        data="src/chessboard_localization/keypoint_approach/data.yaml",
        imgsz=640,
        epochs=100,
        #lr0 = 0.0005,
        #lrf = 0.01,
        patience=0,
        batch=-1,
        hsv_h=0.0, # Keep colors consistent
        hsv_s=0.0, # Keep colors consistent
        hsv_v=0.0, # Keep colors consistent

        flipud=0.0, # Avoid vertical flips (unnatural for chessboards)
        fliplr=0.5, # Horizontal flips are okay (symmetry)

        mosaic=0.0, # breaks geometry
        mixup=0.0, # makes corner identity unclear

        perspective=0.0, # Avoid perspective distortion (breaks structure)
        translate=0.05, # Small shift is okay, avoid cutting corners

        scale=0.05, # Small zoom is okay
        shear=0.0,  # Avoid shearing — distorts lines
        #degrees=5.0, # disable strong rotation

        project="chessy3D",
        name="chessboard_localization2.1"
    )

def validate():
    #model = YOLO("runs/pose/train/weights/best.pt")
    model = YOLO("chessy3D/chessboard_localization2.1/weights/best.pt")

    # Validate on test set
    metrics = model.val(
        data="src/chessboard_localization/keypoint_approach/data.yaml",
        split="test",
        imgsz=640,
        conf=0.5,
        iou=0.5,
    )

def predict():
    image_path = "data/chessred2k/images/0/G000_IMG001.jpg"

    model = YOLO("chessy3D/chessboard_localization2.1/weights/best.pt")
    res = model.predict(
        image_path,
        imgsz=640,
    )

    img = cv2.imread(image_path)

    for kp in res[0].keypoints.xy:
        for x, y in kp:
            x = int(x)
            y = int(y)
            cv2.circle(img, (x, y), 20, (0, 255,0), -1)

    display_image_cv2(img)


if __name__ == "__main__":
    #train()
    #validate()
    predict()
