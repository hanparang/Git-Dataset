import dlib
from skimage import io, transform
import cv2
import os
import pandas as pd
import torch
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset

## face detector와 landmark predictor 정의
def get_bbox(bbox_points):
    x1 = bbox_points[0][0]
    y1 = bbox_points[0][1]
    x2 = bbox_points[1][0]
    y2 = bbox_points[1][1]
    rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
    return rect

class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform=transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 2:6]
        landmarks = np.array([landmarks]).astype("float").reshape(-1, 2)
        sample = {"name": img_name,'image': image, 'landmarks':landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_dataset = FaceLandmarksDataset(csv_file='lable.csv', root_dir='C:/Users/asdzx/Downloads/images')
file = open("Five_landmarks.csv", "w")

for i, frame in enumerate(face_dataset) :
    image = frame['image']
    Lable = list()
    Lable.append(frame['landmarks'])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = image

    dlib_rect = get_bbox(frame['landmarks'])
    file.write(frame["name"])

    shape = predictor(resized, dlib_rect)
    landmark_idx = [36, 39, 42, 45, 30, 48, 54]
    for j in landmark_idx:
       
        x, y = shape.part(j).x, shape.part(j).y
        file.write("," + str(x) + "," + str(y))
        cv2.circle(resized, (x, y), 3, (0, 0, 255), -1)
    file.write("\n")
    r = 500. / image.shape[1]
    dim = (500, int(image.shape[0] * r))

    resized = cv2.resize(resized, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow('frame', resized)
    
    #cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
file.close()
cv2.destroyAllWindows()