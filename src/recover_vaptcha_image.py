import copy
import os

import cv2
import numpy as np
from cv2 import Mat
from sortedcontainers import SortedList


def find_largest_class(src, tolerance):
    classes = []
    for num in src:
        for cls in classes:
            if abs(num - np.mean(cls)) <= tolerance:
                cls.append(num)
                break
        else:
            classes.append([num])
    return max(classes, key=len)


def find_split_count(gray: Mat):
    height, width = gray.shape
    diff_matrix = np.diff(gray.astype(np.int16), axis=1)
    diff_matrix = np.abs(diff_matrix).astype(np.uint8)
    _, binary = cv2.threshold(diff_matrix, 20, 255, cv2.THRESH_BINARY)
    # cv2.imshow("diff", binary)
    # cv2.waitKey(0)

    kernel = np.ones((3, 1), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=2)
    # cv2.imshow("dilation", dilation)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(dilation, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=3)
    lines = np.squeeze(lines, axis=1)
    # 在原图上绘制线条
    # line_background = np.zeros((height, width, 3), dtype=np.uint8)
    # for line in lines:
    #     x1, y1, x2, y2 = line
    #     cv2.line(line_background, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow("image", line_background)
    # cv2.waitKey(0)

    valid_lines = lines[np.abs(lines[:, 2] - lines[:, 0]) < 2]
    lines_x_avg = (valid_lines[:, 2] + valid_lines[:, 0]) // 2
    lines_x_interval = np.diff(np.sort(lines_x_avg) + 1, prepend=0, append=width)
    x_interval = np.mean(find_largest_class(lines_x_interval, 3))
    return round(width / x_interval)


def calculate_connection(connect_cost_matrix, piece_lr_count, piece_tb_count):
    connect_cost_matrix = np.square(connect_cost_matrix)
    lr_connect_cost_matrix, tb_connect_cost_matrix = connect_cost_matrix[:, :, 0], connect_cost_matrix[:, :, 1]
    sorted_lr_indices = np.argsort(lr_connect_cost_matrix)[:, :]
    sorted_tb_indices = np.argsort(tb_connect_cost_matrix)[:, :]
    calculated_count = 0

    class Combination:
        def __init__(self, lr_count_limit, tb_count_limit):
            self.status = np.full((tb_count_limit, lr_count_limit), -1)
            self.tb_count_limit = tb_count_limit
            self.lr_count_limit = lr_count_limit
            self.index = (-1, -1)
            self.cost = 0
            self.connect_count = 0

        @property
        def avg_cost(self):
            return self.cost / self.connect_count

        @property
        def next_index(self):
            tb_index, lr_index = self.index
            if lr_index == -1 and tb_index == -1:
                return 0, 0
            lr_index += 1
            tb_index = tb_index
            if lr_index >= self.lr_count_limit:
                lr_index = 0
                tb_index += 1
                if tb_index >= self.tb_count_limit:
                    return None

            return tb_index, lr_index

        @property
        def next_appended_values(self):
            if not self.next_index:
                return []
            tb_index, lr_index = self.next_index
            if lr_index == 0 and tb_index == 0:
                return []

            if lr_index == 0:
                return [(self.status[tb_index - 1, 0], "bottom")]
            if tb_index == 0:
                return [(self.status[tb_index, lr_index - 1], "right")]
            return [(self.status[tb_index, lr_index - 1], "right"), (self.status[tb_index - 1, lr_index], "bottom")]

        def append(self, value):
            if value in self.status:
                return False
            if not self.next_index:
                return False
            tb_index, lr_index = self.next_index
            clone = copy.deepcopy(self)
            for appended_value, direction in clone.next_appended_values:
                clone.connect_count += 1
                if direction == "bottom":
                    clone.cost += tb_connect_cost_matrix[appended_value, value]
                if direction == "right":
                    clone.cost += lr_connect_cost_matrix[appended_value, value]
            clone.status[tb_index, lr_index] = value
            clone.index = self.next_index
            return clone

    piece_count = connect_cost_matrix.shape[0]
    combination_list = SortedList[Combination](key=lambda x: x.avg_cost)
    min_combination: Combination | None = None
    for (left, sorted_right_indices) in enumerate(sorted_lr_indices):
        for right in sorted_right_indices:
            if left == right:
                continue
            combination = Combination(piece_lr_count, piece_tb_count).append(left).append(right)
            calculated_count += 1
            combination_list.add(combination)
    while combination_list:
        combination = combination_list.pop(0)
        append_candidates = set(range(piece_count))
        for appended_value, direction in combination.next_appended_values:
            if direction == "bottom":
                append_candidates.intersection_update([bottom for bottom in sorted_tb_indices[appended_value]])
            if direction == "right":
                append_candidates.intersection_update([right for right in sorted_lr_indices[appended_value]])
        for append_value in append_candidates:
            new_combination = combination.append(append_value)
            if new_combination:
                if not new_combination.next_index:
                    if not min_combination or new_combination.cost < min_combination.cost:
                        min_combination = new_combination
                elif not min_combination or new_combination.cost < min_combination.cost:
                    combination_list.add(new_combination)
                    calculated_count += 1
    print(f"calculated_count: {calculated_count}")
    return min_combination.status


def recover_image(src_image: Mat):
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    # piece_h_count, piece_w_count = find_split_count(gray.transpose()), find_split_count(gray)
    piece_h_count, piece_w_count = 2, 5
    piece_h, piece_w = src_image.shape[0] // piece_h_count, src_image.shape[1] // piece_w_count
    image_pieces = [(src_image[y:y + piece_h, x:x + piece_w], gray[y:y + piece_h, x:x + piece_w])
                    for x in range(0, src_image.shape[1], piece_w)
                    for y in range(0, src_image.shape[0], piece_h)]

    # for (image_piece, gray_piece) in image_pieces:
    #     cv2.imshow("image_piece", image_piece)
    #     cv2.waitKey(0)

    connect_diff_matrix = np.zeros((len(image_pieces), len(image_pieces), 2), dtype=np.int32)
    for i, (image_piece, gray_piece) in enumerate(image_pieces):
        for j, (image_piece2, gray_piece2) in enumerate(image_pieces):
            if i == j:
                connect_diff_matrix[i, j] = np.iinfo(connect_diff_matrix.dtype).max
                continue
            lr_cost = np.abs((gray_piece[:, -1] - gray_piece2[:, 0]).astype(np.int8)).sum()
            tb_cost = np.abs((gray_piece[-1, :] - gray_piece2[0, :]).astype(np.int8)).sum()
            connect_diff_matrix[i, j] = lr_cost, tb_cost

    min_status = calculate_connection(connect_diff_matrix, piece_w_count, piece_h_count)
    print(f"min_status: {min_status}")

    recovered_image = cv2.vconcat([cv2.hconcat([image_pieces[piece_index][0] for piece_index in row]) for row in min_status])
    # cv2.imshow("recover_image", recovered_image)
    # cv2.waitKey(0)
    return recovered_image


if __name__ == '__main__':
    file_list = os.listdir("../data/vaptcha")
    for file in file_list:
        image = cv2.imread(f"../data/vaptcha/{file}")
        image = recover_image(image)
        cv2.imwrite(f"../data/vaptcha-recover/{file}", image)
