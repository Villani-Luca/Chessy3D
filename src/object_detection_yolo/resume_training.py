from gc import freeze

from ultralytics import YOLO
import wandb

# Load model
if __name__ == '__main__':
    model = YOLO(r"<pathToPt>")
    #freeze_layers = 21 # or 18 pre neck in name:

    wandb.login(key="ec8527bc43fcbc7aef5dfa1fa83e93e0cab69ce8")
    run = wandb.init(entity="269419-unimore-cvcs2025",
        project="chess_finetuning_model", id="<runId>", resume="allow")

    # Train
    model.train(
        data=r"<yamlPath>",
        epochs=600,
        batch=32,
        imgsz=640,
        patience=100,
        project="<projectName>",
        name="<runName>",
        cache=False,
        resume=True
        #freeze=[x for x in range(freeze_layers + 1)],
    )

