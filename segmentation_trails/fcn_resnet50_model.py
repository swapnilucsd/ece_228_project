import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os

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
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask

def transform(image, mask):
    resize = transforms.Resize(size=(576, 1024)) # CHANGE TRANSFORM IF USING TU GRAZ
    image = resize(image)
    mask = resize(mask)

    image = TF.to_tensor(image)
    mask_array = np.array(mask, dtype=np.int64)
    mask_tensor = torch.from_numpy(mask_array).long()
    return image, mask_tensor

def create_fcn_resnet50(num_classes, pretrained=True):
    model = models.segmentation.fcn_resnet50(pretrained=pretrained)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_fcn_resnet50(num_classes=12)  # Specify the number of classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_dataset = AeroscapesDataset('aeroscapes_train/JPEGImages', 'aeroscapes_train/SegmentationClass', transform)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

num_epochs = 25  # Define number of epochs
model.train()
for epoch in range(num_epochs):
    start_time = time.time()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        output = model(images)['out']
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch {epoch+1}/{num_epochs} completed. Loss: {loss.item()}. Completed in {epoch_time:.2f} seconds')

torch.save(model.state_dict(), 'fcn_resnet50.pth')
