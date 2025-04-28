import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json

# Import your DETR model
from model.detr import build_detr
from model.matcher import HungarianMatcher
from model.criterion import SetCriterion

# ---------------- Dataset ---------------- #
class ChessDataset(Dataset):
    def __init__(self, img_folder, annotations, transforms=None):
        """
        img_folder: path to images
        annotations: list of dicts [{'image': str, 'boxes': list, 'labels': list}]
        transforms: torchvision transforms
        """
        self.img_folder = img_folder
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        img_path = os.path.join(self.img_folder, annot['image'])
        img = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(annot['boxes'], dtype=torch.float32)
        labels = torch.tensor(annot['labels'], dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transforms:
            img = self.transforms(img)

        return img, target

# ---------------- Main Fine-tuning Code ---------------- #
def main():
    # ----- Configuration ----- #
    num_classes = 6  # pawn, rook, knight, bishop, queen, king
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    weight_decay = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- Paths (fill these) ----- #
    img_folder = "/Users/luca/Downloads"
    # Reading JSON from a file
    with open('data/annotation.json', 'r') as file:
        loaded_annotation = json.load(file)

    # Displaying the loaded data
    annotations = loaded_annotation
    # ----- Dataset and Dataloader ----- #
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
    ])

    dataset = ChessDataset(img_folder, annotations, transforms=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # ----- Load Pretrained DETR ----- #
    # model = build_detr(num_classes=num_classes)
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    # checkpoint = torch.load('PATH_TO_PRETRAINED_DETR/detr-r50-e632da11.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['model'], strict=False)


    # Replace classification head
    hidden_dim = model.class_embed.in_features
    model.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for 'no object'

    model.to(device)

    # ----- Set Loss and Optimizer ----- #
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=losses,
    )
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ----- Training Loop ----- #
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(imgs)
            loss_dict = criterion(outputs, targets)
            losses_total = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses_total.backward()
            optimizer.step()

            running_loss += losses_total.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint every few epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"detr_chess_epoch{epoch+1}.pth")

    # Final save
    torch.save(model.state_dict(), "detr_chess_finetuned.pth")
    print("Training complete! Fine-tuned model saved.")

if __name__ == "__main__":
    main()
