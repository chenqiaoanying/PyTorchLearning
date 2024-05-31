# scan the picture from left fo right and find the edge of the image
from collections import Counter

import cv2
import numpy as np
from cv2 import Mat


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
    # line_background = np.zeros_like(gray)
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


def calculate_connection(connect_diff_matrix, max_link_count, max_link_size):
    connect_indices = np.argsort(connect_diff_matrix.flatten())
    prev_connect_indices, next_connect_indices = np.unravel_index(connect_indices, connect_diff_matrix.shape)
    node_dict = {}
    link_head_set = {}

    def find_head(node: dict):
        while node["prev"]:
            node = node["prev"]
        return node

    def get_or_create_node(value):
        if value not in node_dict:
            node_dict[value] = {"value": value, "prev": None, "next": None}
            link_head_set[value] = 1
        return node_dict[value]

    def to_list(node: dict):
        result = []
        while node:
            result.append(node["value"])
            node = node["next"]
        return result

    for i, j in zip(prev_connect_indices, next_connect_indices):
        if len([count for count in link_head_set.values() if count >= max_link_size]) >= max_link_count:
            break
        node_i, node_j = get_or_create_node(i), get_or_create_node(j)
        if not node_i["next"] and not node_j["prev"]:
            head = find_head(node_i)
            if link_head_set[head["value"]] < max_link_size and head != node_j:
                node_i["next"] = node_j
                node_j["prev"] = node_i
                count_from_node_j = link_head_set.pop(j)
                link_head_set[head["value"]] += count_from_node_j
    return [to_list(node_dict[index]) for index, count in link_head_set.items() if count >= max_link_size]


image = cv2.imread("1a5c9ecb5c54461daf1d18722c69ab4a.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
piece_h_count, piece_w_count = find_split_count(gray.transpose()), find_split_count(gray)
# piece_h_count, piece_w_count = 2, 5
piece_h, piece_w = image.shape[0] // piece_h_count, image.shape[1] // piece_w_count
image_pieces = [(image[y:y + piece_h, x:x + piece_w], gray[y:y + piece_h, x:x + piece_w])
                for x in range(0, image.shape[1], piece_w)
                for y in range(0, image.shape[0], piece_h)]

# for (image_piece, gray_piece) in image_pieces:
#     cv2.imshow("image_piece", image_piece)
#     cv2.waitKey(0)

connect_diff_matrix = np.zeros((len(image_pieces), len(image_pieces), 2), dtype=np.int32)
for i, (image_piece, gray_piece) in enumerate(image_pieces):
    for j, (image_piece2, gray_piece2) in enumerate(image_pieces):
        if i == j:
            connect_diff_matrix[i, j] = np.iinfo(connect_diff_matrix.dtype).max
            continue
        rl_score = np.abs((gray_piece[:, -1] - gray_piece2[:, 0]).astype(np.int8)).sum()
        bt_score = np.abs((gray_piece[-1, :] - gray_piece2[0, :]).astype(np.int8)).sum()
        connect_diff_matrix[i, j] = rl_score, bt_score
        print(f"i:{i}, j:{j}, rl_score: {rl_score}, bt_score: {bt_score}")
rl_connect_diff_matrix, bt_connect_diff_matrix = connect_diff_matrix[:, :, 0], connect_diff_matrix[:, :, 1]

row_list = calculate_connection(rl_connect_diff_matrix, piece_h_count, piece_w_count)
col_list = calculate_connection(bt_connect_diff_matrix, piece_w_count, piece_h_count)

first_col = next(col for col in col_list if row_list[0][0] in col)
row_list = sorted(row_list, key=lambda row: first_col.index(row[0]))

recovered_image = cv2.vconcat([cv2.hconcat([image_pieces[piece_index][0] for piece_index in row]) for row in row_list])
cv2.imshow("recover_image", recovered_image)
cv2.waitKey(0)
