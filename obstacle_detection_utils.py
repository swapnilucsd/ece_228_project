import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms as T

import cv2
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
import ssl

import math
from enum import Enum

import cv2 as cv
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from utils import *
from data_loader import *
from train import *
from segmentation import *

ssl._create_default_https_context = ssl._create_unverified_context

labels = {
    "unlabeled": 0,
    "pavedArea": 1,
    "dirt": 2,
    "grass": 3,
    "gravel": 4,
    "water": 5,
    "rocks": 6,
    "pool": 7,
    "vegetation": 8,
    "roof": 9,
    "wall": 10,
    "window": 11,
    "door": 12,
    "fence": 13,
    "fencePole": 14,
    "person": 15,
    "dog": 16,
    "car": 17,
    "bicycle": 18,
    "tree": 19,
    "baldTree": 20,
    "arMarker": 21,
    "obstacle": 22,
    "conflicting": 23,
}


class RiskLevel(Enum):
    VERY_HIGH = 100
    HIGH = 20
    MEDIUM = 10
    LOW = 5
    ZERO = 0


risk_table = {
    "unlabeled": RiskLevel.ZERO,
    "pavedArea": RiskLevel.LOW,
    "dirt": RiskLevel.LOW,
    "grass": RiskLevel.ZERO,
    "gravel": RiskLevel.LOW,
    "water": RiskLevel.HIGH,
    "rocks": RiskLevel.MEDIUM,
    "pool": RiskLevel.HIGH,
    "vegetation": RiskLevel.LOW,
    "roof": RiskLevel.HIGH,
    "wall": RiskLevel.HIGH,
    "window": RiskLevel.HIGH,
    "door": RiskLevel.HIGH,
    "fence": RiskLevel.HIGH,
    "fencePole": RiskLevel.HIGH,
    "person": RiskLevel.VERY_HIGH,
    "dog": RiskLevel.VERY_HIGH,
    "car": RiskLevel.VERY_HIGH,
    "bicycle": RiskLevel.VERY_HIGH,
    "tree": RiskLevel.HIGH,
    "baldTree": RiskLevel.HIGH,
    "arMarker": RiskLevel.ZERO,
    "obstacle": RiskLevel.HIGH,
    "conflicting": RiskLevel.HIGH,
}

categories_of_interest = {1: "person", 2: "bicycle", 3: "car", 7: "truck"}


class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

    def forward(self, x):
        return self.model(x)


# No change
def getDistance(img, pt1, pt2):
    dim = img.shape
    furthestDistance = math.hypot(dim[0], dim[1])
    dist = distance.euclidean(pt1, pt2)
    return 1 - abs(dist / furthestDistance)


# obstacles is output of define_no_landing_zones
def dist_to_obs(lz, obstacles, img):
    posLz = lz.get("position")
    norm_dists = []
    if not obstacles:
        return 0
    else:
        for ob in obstacles:
            dist = getDistance(img, (ob[0], ob[1]), posLz)
            norm_dists.append(1 - dist)
        return np.mean(norm_dists)


# No change
def getDistanceCenter(img, pt):
    dim = img.shape
    furthestDistance = math.hypot(dim[0] / 2, dim[1] / 2)
    dist = distance.euclidean(pt, [dim[0] / 2, dim[1] / 2])
    return 1 - abs(dist / furthestDistance)


# No change
def circles_intersect(x1, x2, y1, y2, r1, r2):
    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if d < r1 - r2:
        return -3
    elif d < r2 - r1:
        return -2
    elif d > r1 + r2:
        return 0
    else:
        return -1


def get_risk_map(seg_img, labels, risk_table, gaussian_sigma=25):
    image = seg_img
    risk_array = image.astype("float32")
    for label in labels:
        np.where(risk_array == labels[label], risk_table[label], risk_array)
    risk_array = gaussian_filter(risk_array, sigma=gaussian_sigma)
    risk_array = (risk_array / risk_array.max()) * 255
    risk_array = np.uint8(risk_array)
    return risk_array


def visualize_heatmap(risk_heatmap):
    plt.imshow(risk_heatmap, cmap="coolwarm")
    plt.colorbar()
    plt.title("Heatmap")
    plt.axis("off")
    plt.show()


def define_no_landing_zones(boxes, labels):
    no_landing_zones = []
    for box, label in zip(boxes, labels):
        if label in categories_of_interest:
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
            if label == 1:  # person
                radius = 120
            elif label == 2:  # bicycle
                radius = 135
            elif label in [3, 7]:  # car or truck
                radius = 150
            no_landing_zones.append((x_center, y_center, radius))
    return no_landing_zones


def meets_min_safety_requirement(zone_proposed, obstacles_list):
    posLz = zone_proposed.get("position")
    radLz = zone_proposed.get("radius")
    for obstacle in obstacles_list:
        touch = circles_intersect(
            posLz[0], obstacle[0], posLz[1], obstacle[1], radLz, obstacle[2]
        )
        if touch < 0:
            return False
    return True


def get_landing_zones_proposals(high_risk_obstacles, stride, r_landing, image):
    if not torch.is_tensor(image):
        t = T.Compose([T.ToTensor()])
        image = t(image)
    height, width = image.shape[1:]
    zones_proposed = []

    for y in range(r_landing, height - r_landing, stride):
        for x in range(r_landing, width - r_landing, stride):
            lzProposed = {
                "confidence": math.nan,
                "radius": r_landing,
                "position": (x, y),
            }
            if not meets_min_safety_requirement(lzProposed, high_risk_obstacles):
                lzProposed["confidence"] = 0
            zones_proposed.append(lzProposed)
    return zones_proposed


def detect_obstacles(model, image, device):
    t = T.Compose([T.ToTensor()])
    image = t(image)
    image = image.to(device)
    with torch.no_grad():
        prediction = model([image])
    boxes = prediction[0]["boxes"].cpu().numpy()
    labels = prediction[0]["labels"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()

    score_threshold = 0.5
    boxes = boxes[scores > score_threshold]
    labels = labels[scores > score_threshold]

    return boxes, labels


def draw_lzs_obs(list_lzs, list_obs, img, thickness=3):
    t = T.Compose([T.ToTensor()])
    img = t(img)

    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for obstacle in list_obs:
        cv.circle(
            img,
            (obstacle[0], obstacle[1]),
            obstacle[2],
            (0, 0, 255),
            thickness=thickness,
        )
    for lz in list_lzs:
        posLz = lz.get("position")
        radLz = lz.get("radius")
        cv.circle(img, (posLz[0], posLz[1]), radLz, (0, 255, 0), thickness=thickness)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def risk_map_eval_basic(img, areaLz):
    maxRisk = areaLz * 255
    totalRisk = np.sum(img)
    return 1 - (totalRisk / maxRisk)


def rank_lzs(lzsProposals, riskMap, obstacles, weightDist=5, weightRisk=15, weightOb=5):
    for lz in lzsProposals:
        riskFactor, distanceFactor, obFactor = 0, 0, 0
        lzRad = lz.get("radius")
        lzPos = lz.get("position")
        mask = np.zeros_like(riskMap)
        mask = cv.circle(mask, (lzPos[0], lzPos[1]), lzRad, (255, 255, 255), -1)
        areaLz = math.pi * lzRad * lzRad
        crop = cv.bitwise_and(riskMap, mask)

        if weightRisk != 0:
            riskFactor = risk_map_eval_basic(crop, areaLz)
        if weightDist != 0:
            distanceFactor = getDistanceCenter(riskMap, (lzPos[0], lzPos[1]))
        if weightOb != 0:
            obFactor = dist_to_obs(lz, obstacles, riskMap)

        if lz["confidence"] is math.nan:
            lz["confidence"] = abs(
                (
                    weightRisk * riskFactor
                    + weightDist * distanceFactor
                    + weightOb * obFactor
                )
                / (weightRisk + weightDist + weightOb)
            )

    lzsSorted = sorted(lzsProposals, key=lambda k: k["confidence"])
    return lzsSorted
