import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import build_model
from util.misc import nested_tensor_from_tensor_list
from util.box_ops import box_cxcywh_to_xyxy

# ==== CONFIG ====
img_path = "D:\\Downloads\\1a530d578f3f0bf3497bfeff3d953025_jpg.rf.c74da55881f0dba02f72b74b96444c00.jpg"
checkpoint_path = "D:\\detr_output\\checkpoint.pth"
num_classes = 14  # Include background as needed

# ==== STEP 1: Costruzione modello ====
class Args:
    # Imposta i parametri necessari
    backbone = "resnet50"
    dilation = False
    position_embedding = "sine"
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    enc_layers = 6
    dec_layers = 6
    pre_norm = False
    return_intermediate_dec = False
    masks = False
    num_classes = num_classes
    num_queries = 100
    lr_backbone = 1e-5
    dataset_file = "custom"
    device = "cuda"
    aux_loss = True
    set_cost_class = 2.0
    set_cost_bbox = 5.0
    set_cost_giou = 2.0
    set_cost_mask = 1.0
    bbox_loss_coef = 5.0
    giou_loss_coef = 2.0
    eos_coef = 2.0

args = Args()
model, _, _ = build_model(args)

# ==== STEP 2: Caricamento pesi ====
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["model"]
state_dict["class_embed.weight"] = model.class_embed.weight
state_dict["class_embed.bias"] = model.class_embed.bias
model.load_state_dict(state_dict, strict=False)
model.eval()

# ==== STEP 3: Preprocessing immagine ====
transform = Compose([
    Resize(800),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open(img_path).convert("RGB")
img_tensor = transform(img)
inputs = nested_tensor_from_tensor_list([img_tensor])

# ==== STEP 4: Inferenza ====
with torch.no_grad():
    outputs = model(inputs)

logits = outputs["pred_logits"][0]
boxes = outputs["pred_boxes"][0]
x = outputs["aux_outputs"][0]

probas = torch.softmax(logits, 1)
scores, labels = probas.max(-1)

# ==== STEP 5: Filtro con soglia ====
threshold = 0.9
#keep = scores > threshold

keep = (labels != 13) & (scores > 0.2)


print("Predetti:", labels[:10])
print("Punteggi:", scores[:10])

scores = scores[keep]
labels = labels[keep]
boxes = boxes[keep]
boxes = box_cxcywh_to_xyxy(boxes)

# Rescale boxes
w, h = img.size
boxes = boxes * torch.tensor([w, h, w, h])

# ==== STEP 6: Visualizzazione ====
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(img)

for box, score, label in zip(boxes, scores, labels):
    xmin, ymin, xmax, ymax = box.tolist()
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin, f'{label.item()} ({score:.2f})', 
            fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

plt.axis('off')
plt.show()
