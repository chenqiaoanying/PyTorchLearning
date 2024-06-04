import random

import cv2
import numpy as np


# 旋转图像
def rotate_image(image, points, angle):
    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算旋转后的图像尺寸
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # 旋转图像
    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    # 旋转点
    new_points = []
    for (x, y) in points:
        v = np.array([x, y, 1])
        new_x, new_y = M.dot(v)
        new_points.append((int(new_x), int(new_y)))

    return rotated, new_points


# 裁切图像
def crop_image(image, points, x, y, w, h):
    cropped = image[y:y + h, x:x + w]

    new_points = [(px - x, py - y) for (px, py) in points if x <= px < x + w and y <= py < y + h]

    return cropped, new_points


# 膨胀图像
def dilate_image(image, points, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)

    return dilated, points


def random_perspective_transform(image, points):
    height, width = image.shape[:2]
    min_x = round(points[:, 0].min())
    max_x = round(points[:, 0].max())
    min_y = round(points[:, 1].min())
    max_y = round(points[:, 1].max())

    new_left_top = (random.randint(0, min_x), random.randint(0, min_y))
    new_left_bottom = (random.randint(0, min_x), random.randint(max_y, height))
    new_right_top = (random.randint(max_x, width), random.randint(0, min_y))
    new_right_bottom = (random.randint(max_x, width), random.randint(max_y, height))
    wrapper_points = np.array([new_left_top, new_left_bottom, new_right_bottom, new_right_top], dtype=np.float32)
    wrapper_points = wrapper_points if random.randint(0, 1) == 0 else wrapper_points[::-1]
    random_order = random.randint(0, 3)
    wrapper_points = np.concatenate((wrapper_points[random_order:], wrapper_points[:random_order]))
    target_points = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(wrapper_points, target_points)
    warped_image = cv2.warpPerspective(image, M, (width, height))
    transformed_points = cv2.perspectiveTransform(points[None, :, :], M).squeeze(0)

    return warped_image, transformed_points


if __name__ == '__main__':
    from src.dataset import get_image_info

    image_info_list = get_image_info(20)
    image_info = next(image_info for image_info in image_info_list if "f0847fd8-01700a69823c494abb82d990a442fe3a" in image_info.image_path)
    image = image_info.image
    points = image_info.resampled_real_points
    new_image, new_points = random_perspective_transform(image, points)

    cv2.imshow("image", image)
    for x, y in new_points:
        cv2.circle(new_image, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imshow("warped_image", new_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
