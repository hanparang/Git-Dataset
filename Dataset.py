from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import cv2
import matplotlib.patches as patches

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

def get_bbox_patch(image, bbox_points):
    x1 = bbox_points[0][0]
    y1 = bbox_points[0][1]
    x2 = bbox_points[1][0]
    y2 = bbox_points[1][1]
    rect = patches.Rectangle((x1, y2), x2-x1, y1-y2, linewidth=1,edgecolor='r',facecolor='none')

    return image, rect





class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform=transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:5]
        landmarks = np.array([landmarks]).astype("float").reshape(-1, 2)
        sample = {'image': image, 'landmarks':landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(csv_file='C:/Users/asdzx/Downloads/test_torch/face_detect.csv', root_dir='C:/Users/asdzx/Downloads/test_torch')
fig = plt.figure()

for i, each_sample in enumerate(face_dataset):
    print(i, each_sample['image'].shape, each_sample['landmarks'])

    image, bbox = get_bbox_patch(each_sample['image'], each_sample['landmarks'])

    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.imshow(image)
    ax.add_patch(bbox)
    ax.set_title("Sample#{}".format(i))
    ax.axis("off")

    if i == 3:
        plt.show()
        break