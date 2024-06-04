import dataclasses
import json
import os

import cv2
import numpy as np
import torch
from scipy.interpolate import splprep, splev
from torch.utils.data import Dataset


def resample_curve(points, num_points):
    tck, u = splprep(points.T, s=0, per=0)
    u_new = np.linspace(u.min(), u.max(), num_points)
    new_points = splev(u_new, tck)
    return np.array(new_points).T


def get_image_info(number_key_points):
    project_info: list[dict] = json.load(open("../data/vaptcha-recover/project-2-at-2024-06-03-17-52-37c15ff6.json"))
    image_filename_list = set(os.listdir("../data/vaptcha-recover/images"))
    return [ImageInfo(image_path=f"../data/vaptcha-recover/images/{image_info['file_upload']}",
                      normalized_points=np.array([[result['value']['x'], result['value']['y']] for result in image_info["annotations"][0]["result"]]),
                      number_key_points=number_key_points)
            for image_info in project_info if image_info["cancelled_annotations"] == 0 and image_info['file_upload'] in image_filename_list]


@dataclasses.dataclass
class ImageInfo:
    image_path: str
    normalized_points: np.ndarray
    number_key_points: int

    @property
    def image(self):
        return cv2.imread(self.image_path)

    @property
    def real_points(self):
        return self.normalized_points * np.array([self.image.shape[1] / 100.0, self.image.shape[0] / 100.0])

    @property
    def resampled_normalized_points(self):
        return resample_curve(self.normalized_points, self.number_key_points)

    @property
    def resampled_real_points(self):
        return self.resampled_normalized_points * np.array([self.image.shape[1] / 100.0, self.image.shape[0] / 100.0])

    def show(self):
        image = self.image.copy()
        for x, y in self.real_points:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class CurveDataset(Dataset):
    def __init__(self, number_key_points=10):
        self.project_info = get_image_info(number_key_points)

    def __len__(self):
        return len(self.project_info)

    def __getitem__(self, idx):
        image_info = self.project_info[idx]
        image = image_info.image
        points = image_info.resampled_real_points
        # image, points = random_persepective_transform(image, points)
        return torch.from_numpy(image.transpose(2, 0, 1)).float(), torch.from_numpy(points).float()
