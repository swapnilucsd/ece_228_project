from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T
import torch.nn.functional as F

from os.path import isfile, join
from os import listdir
from re import I


def create_image_id_dataframe(image_path):
    """
    Creates a DataFrame containing the image IDs.

    Parameters:
    image_path (str): Path to the directory containing image files.

    Returns:
    pd.DataFrame: DataFrame with a single column 'id' containing the image IDs.
    """
    image_ids = []
    filenames = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    for filename in filenames:
        image_ids.append(filename.split(".")[0])
    df = pd.DataFrame({"id": image_ids}, index=np.arange(0, len(image_ids)))
    return df


def get_pixel_accuracy(predicted_image, mask):
    """
    Calculates the pixel accuracy between the predicted image and the ground truth mask.

    Parameters:
    predicted_image (torch.Tensor): The predicted segmentation map.
    mask (torch.Tensor): The ground truth segmentation mask.

    Returns:
    float: Pixel accuracy as a float.
    """
    with torch.no_grad():
        predicted_image = torch.argmax(F.softmax(predicted_image, dim=1), dim=1)
        correct = torch.eq(predicted_image, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def get_mean_iou(predicted_label, label, eps=1e-8, num_classes=23):
    """
    Calculates the mean Intersection over Union (IoU) for the predicted labels.

    Parameters:
    predicted_label (torch.Tensor): The predicted segmentation map.
    label (torch.Tensor): The ground truth segmentation map.
    eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-8.
    num_classes (int, optional): Number of classes. Default is 23.

    Returns:
    float: Mean IoU across all classes.
    """
    with torch.no_grad():
        predicted_label = F.softmax(predicted_label, dim=1)
        predicted_label = torch.argmax(predicted_label, dim=1)

        predicted_label = predicted_label.contiguous().view(-1)
        label = label.contiguous().view(-1)

        iou_per_class = []
        for class_number in range(num_classes):
            true_predicted_class = predicted_label == class_number
            true_label = label == class_number

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersection = (
                    torch.logical_and(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item()
                )
                union = (
                    torch.logical_or(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item()
                )
                iou = (intersection + eps) / (union + eps)
                iou_per_class.append(iou)
        mean_iou_across_classes = np.nanmean(iou_per_class)
        return mean_iou_across_classes


def plot_loss_vs_epoch(history):
    """
    Plots training and validation loss per epoch.

    Parameters:
    history (dict): Dictionary containing training and validation loss values per epoch.

    Returns:
    None
    """
    plt.plot(history["val_loss"], label="Validation Loss", marker="o")
    plt.plot(history["train_loss"], label="Training Loss", marker="o")
    plt.title("Loss per Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.show()


def plot_iou_score_vs_epoch(history):
    """
    Plots mean Intersection over Union (mIoU) score per epoch for training and validation data.

    Parameters:
    history (dict): Dictionary containing training and validation mIoU values per epoch.

    Returns:
    None
    """
    plt.plot(history["train_miou"], label="Training mIoU", marker="*")
    plt.plot(history["val_miou"], label="Validation mIoU", marker="*")
    plt.title("mIoU Score per Epoch")
    plt.ylabel("Mean IoU")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.show()


def plot_accuracy_vs_epoch(history):
    """
    Plots training and validation accuracy per epoch.

    Parameters:
    history (dict): Dictionary containing training and validation accuracy values per epoch.

    Returns:
    None
    """
    plt.plot(history["train_acc"], label="Training Accuracy", marker="*")
    plt.plot(history["val_acc"], label="Validation Accuracy", marker="*")
    plt.title("Accuracy per Epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_image_mask(image, mask, pred_mask):
    """
    Visualizes the original image, ground truth mask, and predicted mask.

    Parameters:
    image (PIL.Image or ndarray): The input image.
    mask (torch.Tensor): The ground truth segmentation mask.
    pred_mask (torch.Tensor): The predicted segmentation mask.

    Returns:
    None
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(mask)
    ax2.set_title("Ground Truth Mask")
    ax2.axis("off")

    ax3.imshow(pred_mask)
    ax3.set_title("Predicted Mask")
    ax3.axis("off")

    plt.show()


def predict_image_mask_miou_pixel_accuracy(
    model, image, mask, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """
    Predicts the segmentation mask for an image and calculates the mean IoU and pixel accuracy.

    Parameters:
    model (torch.nn.Module): The trained segmentation model.
    image (PIL.Image or ndarray): The input image.
    mask (torch.Tensor): The ground truth segmentation mask.
    mean (list, optional): List of mean values for normalization. Default is [0.485, 0.456, 0.406].
    std (list, optional): List of standard deviation values for normalization. Default is [0.229, 0.224, 0.225].

    Returns:
    tuple: (predicted mask, mean IoU score, pixel accuracy)
    """
    model.eval()
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_image = model(image)
        mean_iou_score = get_mean_iou(predicted_image, mask)
        pix_accuracy = get_pixel_accuracy(predicted_image, mask)
        predicted_mask = torch.argmax(predicted_image, dim=1).cpu().squeeze(0)

    return predicted_mask, mean_iou_score, pix_accuracy


def get_miou_pixel_accuracy_scores_from_trained_model(model, test_set, device):
    """
    Evaluates a trained model on a test dataset to calculate mean IoU and pixel accuracy.

    Parameters:
    model (torch.nn.Module): The trained segmentation model.
    test_set (Dataset): The test dataset containing images and masks.

    Returns:
    tuple: (list of mean IoU scores, list of pixel accuracy scores)
    """
    score_iou = []
    accuracy = []

    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, miou, pix_acc = predict_image_mask_miou_pixel_accuracy(
            model, img, mask, device
        )
        score_iou.append(miou)
        accuracy.append(pix_acc)

    return score_iou, accuracy
