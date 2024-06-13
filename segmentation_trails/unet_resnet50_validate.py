import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AEROSCAPES IN NAME ONLY. CHANGE INPUTS AS NEEDED
class AeroscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask

def transform(image, mask):
    resize = transforms.Resize(size=(576, 1024))
    image = resize(image)
    mask = resize(mask)

    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int32))
    mask = mask.long()
    return image, mask

def load_trained_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=12
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

model = load_trained_model('unet_resnet50.pth')

# Validation dataset # CURRENTLY SET FOR AEROSCAPES. CHANGE FOR DATASET BEING USED
val_dataset = AeroscapesDataset(
    'aeroscapes_val/JPEGImages', 
    'aeroscapes_val/SegmentationClass', 
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Function to calculate IoU
def calculate_iou(pred, gt, num_classes=12):
    intersections = torch.zeros(num_classes, dtype=torch.float32, device=device)
    unions = torch.zeros(num_classes, dtype=torch.float32, device=device)
    frequencies = torch.zeros(num_classes, dtype=torch.float32, device=device)

    pred = pred.view(-1)
    gt = gt.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        gt_inds = (gt == cls)
        intersections[cls] += (pred_inds & gt_inds).sum().item()
        unions[cls] += (pred_inds | gt_inds).sum().item()
        frequencies[cls] += gt_inds.sum().item()

    # Avoid division by zero
    unions = unions.clamp(min=1)
    iou_list = (intersections / unions).tolist()
    freq_weights = frequencies / frequencies.sum()
    freq_weighted_iou = (intersections / unions * freq_weights).sum()

    return iou_list, np.nanmean(iou_list), freq_weighted_iou.item()


# Evaluate the model
# Load the first batch
images, masks = next(iter(val_loader))
images, masks = images.to(device), masks.to(device)

# Process the batch
with torch.no_grad():
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)

# Calculate IoU for the first example - one example for visualization
iou_list, mean_iou, freq_weighted_iou = calculate_iou(predictions, masks, num_classes=12)
print(f"Class IoUs: {iou_list}")
print(f"Mean IoU: {mean_iou}")
print(f"Frequency Weighted IoU: {freq_weighted_iou}")

# Visualization
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
image = TF.to_pil_image(images[0].cpu())
gt = masks[0].cpu().numpy()
pred = predictions[0].cpu().numpy()

axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(gt, cmap='gray')
axs[1].set_title('Ground Truth')
axs[1].axis('off')

axs[2].imshow(pred, cmap='gray')
axs[2].set_title(f'Model Output - IoU: {mean_iou:.4f}')
axs[2].axis('off')

plt.tight_layout()
plt.show()
