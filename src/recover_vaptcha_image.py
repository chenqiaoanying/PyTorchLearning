# scan the picture from left fo right and find the edge of the image

import cv2
import numpy as np

image = cv2.imread("1805502d43514d9e9250633720eca013.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
diff_matrix = np.diff(gray, axis=1).astype(np.int8)
diff_matrix = np.abs(diff_matrix)
diff_matrix = diff_matrix.reshape(2, diff_matrix.shape[0] // 2, diff_matrix.shape[1]).sum(axis=1)
top_1_row_indices_in_per_col = np.argpartition(-diff_matrix, 1, axis=0)[:1]
diff_matrix = diff_matrix[top_1_row_indices_in_per_col, np.arange(diff_matrix.shape[1])]
diff_matrix = diff_matrix.sum(axis=0)
col_split_indices = np.sort(np.argpartition(-diff_matrix, 6)[:6] + 1)
col_split_indices_diff = np.diff(col_split_indices)
col_split_indices_diff_diff = np.abs(np.diff(col_split_indices_diff))
col_split_indices_indices = set([col_split_indices_index
                                 for col_split_indices_diff_diff_index, e in enumerate(col_split_indices_diff_diff) if e < 3
                                 for col_split_indices_diff_index in [col_split_indices_diff_diff_index, col_split_indices_diff_diff_index + 1]
                                 for col_split_indices_index in [col_split_indices_diff_index, col_split_indices_diff_index + 1]])
col_split_indices_indices = sorted(col_split_indices_indices)

print(col_split_indices[col_split_indices_indices])

gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("gray", gray)
cv2.imshow("gaussian", gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
