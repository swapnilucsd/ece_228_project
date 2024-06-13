import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image
import cv2
from os.path import join


class DataGenerator(Dataset):
    def __init__(
        self,
        img_path,
        mask_path,
        X,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        transform=None,
        normalize=False,
        patch=False,
    ):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.mean = mean
        self.std = std
        self.transform = transform
        self.normalize = normalize
        self.patches = patch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(join(self.img_path, self.X[idx] + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            join(self.mask_path, self.X[idx] + ".png"), cv2.IMREAD_GRAYSCALE
        )

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = Image.fromarray(augmented["image"])
            mask = augmented["mask"]
        else:
            img = Image.fromarray(img)

        if self.normalize:
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
            img = t(img)

        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.get_img_patches(img, mask)

        return img, mask

    def get_img_patches(self, img, mask):
        kh, kw = 512, 768
        dh, dw = 512, 768

        img_patches = img.unfold(1, kh, dh).unfold(2, kw, dw)
        img_patches = img_patches.contiguous().view(3, -1, kh, kw)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, kh, dh).unfold(1, kw, dw)
        mask_patches = mask_patches.contiguous().view(-1, kh, kw)

        return img_patches, mask_patches
