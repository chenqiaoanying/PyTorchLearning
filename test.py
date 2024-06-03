import numpy as np

# 创建原始矩阵
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 找到每一列的top k值的索引
k = 2
top_k_indices = np.argpartition(-matrix, k, axis=0)[:k]

# 使用这些索引来获取top k值
top_k_values = matrix[top_k_indices, np.arange(matrix.shape[1])]

print(top_k_values)