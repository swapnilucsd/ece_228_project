import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import os
from torchvision import transforms
import torch
import torchvision.models as models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fcn_resnet50(num_classes, pretrained_path):
    # Load FCN model with a ResNet-50 backbone
    model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=num_classes)
    state_dict = torch.load(pretrained_path)
    
    # Remove auxiliary classifier weights if they exist
    state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}
    model.load_state_dict(state_dict, strict=False)
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_fcn_resnet50(num_classes=12, pretrained_path='fcn_resnet50.pth')
model = model.to(device)
model.eval()

# Load pre-trained ResNet and remove fully connected layers
resnet50 = models.resnet50(pretrained=True)
modules = list(resnet50.children())[:-2]  # remove last fc layer and avg pool
resnet50 = nn.Sequential(*modules)

# AEROSCAPES IN NAME ONLY. CHANGE INPUTS AS NEEDED
class AeroscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask
    
# Define transformations
def transform(image, mask):
    # Resize the image and mask to the required dimensions
    resize = transforms.Resize(size=(576, 1024)) # THIS SIZE FOR AEROSCAPES. CHANGE AS NEEDED FOR TU GRAZ
    image = resize(image)
    mask = resize(mask)

    image = TF.to_tensor(image)

    mask_array = np.array(mask, dtype=np.int64)  
    mask_tensor = torch.from_numpy(mask_array)  
    mask_tensor = mask_tensor.long() 
    return image, mask_tensor


# Function to calculate IoU
def calculate_iou(pred, gt, num_classes):
    iou_list = []
    pred = pred.view(-1)
    gt = gt.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        gt_inds = gt == cls
        intersection = (pred_inds[gt_inds]).sum() 
        union = pred_inds.sum() + gt_inds.sum() - intersection 
        iou = intersection.float() / union.float()
        iou_list.append(iou.item())
    
    return iou_list

# Set up the validation dataset and loader
val_dataset = AeroscapesDataset('aeroscapes_val/JPEGImages', 'aeroscapes_val/SegmentationClass', transform) # CHANGE HERE IF USING TU GRAZ
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

import torch
import matplotlib.pyplot as plt

# Function to calculate IoU
def calculate_iou(pred, gt, num_classes=12):
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        gt_inds = (gt == cls)
        intersection = (pred_inds & gt_inds).sum().item()
        union = (pred_inds | gt_inds).sum().item()
        
        if union == 0:
            iou = float('nan')  # Avoid division by zero
        else:
            iou = intersection / union
        
        iou_list.append(iou)
        print(f"Class {cls}: Intersection = {intersection}, Union = {union}, IoU = {iou}")

    mean_iou = np.nanmean(iou_list)
    print(f"Mean IoU (ignoring NaNs): {mean_iou}")
    return mean_iou

images, masks = next(iter(val_loader))
images, masks = images.to(device), masks.to(device)

with torch.no_grad():
    outputs = model(images)['out']
    predictions = torch.argmax(outputs, dim=1)

iou = calculate_iou(predictions[0], masks[0], num_classes=12)

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
axs[2].set_title(f'Model Output - IoU: {iou:.4f}')
axs[2].axis('off')

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Flatten the ground truth and prediction arrays
gt_flat = gt.flatten()
pred_flat = pred.flatten()

# Compute confusion matrix
conf_matrix = confusion_matrix(gt_flat, pred_flat, labels=range(12))  # Assuming 12 classes

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()