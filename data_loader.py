import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image
import cv2
from os.path import join


class DataGenerator(Dataset):
    """
    Custom dataset class for image and mask data.

    Parameters:
    img_path (str): Path to the directory containing images.
    mask_path (str): Path to the directory containing masks.
    X (list): List of image filenames (without extensions).
    mean (list, optional): Mean values for normalization.
    std (list, optional): Standard deviation values for normalization.
    transform (callable, optional): Optional transform to be applied on a sample.
    normalize (bool, optional): Whether to apply normalization. Default is False.
    patch (bool, optional): Whether to generate patches. Default is False.

    Returns:
    img (torch.Tensor): Transformed image tensor or patches.
    mask (torch.Tensor): Transformed mask tensor or patches.
    """

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
        # Load image and mask
        img = cv2.imread(join(self.img_path, self.X[idx] + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            join(self.mask_path, self.X[idx] + ".png"), cv2.IMREAD_GRAYSCALE
        )

        # Apply transformation if specified
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = Image.fromarray(augmented["image"])
            mask = augmented["mask"]
        else:
            img = Image.fromarray(img)

        # Convert image to tensor and normalize if needed
        if self.normalize:
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
            img = t(img)

        mask = torch.from_numpy(mask).long()

        # Generate patches if specified
        if self.patches:
            img, mask = self.get_img_patches(img, mask)

        return img, mask

    def get_img_patches(self, img, mask):
        """
        Generate patches from the image and mask.

        Parameters:
        img (torch.Tensor): Image tensor.
        mask (torch.Tensor): Mask tensor.

        Returns:
        img_patches (torch.Tensor): Image patches.
        mask_patches (torch.Tensor): Mask patches.
        """
        kh, kw = 512, 768  # Kernel size
        dh, dw = 512, 768  # Strides

        img_patches = img.unfold(1, kh, dh).unfold(2, kw, dw)
        img_patches = img_patches.contiguous().view(3, -1, kh, kw)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, kh, dh).unfold(1, kw, dw)
        mask_patches = mask_patches.contiguous().view(-1, kh, kw)

        return img_patches, mask_patches
