# Get the dia number [1 2 3 4]
# In order to turn to a sparse array, first turn it into [[1 2 3 4]]
# Use sp.dia_matrix turn it into a sparse array which dia and the shape are same as the origin array.origin
# Use origin - now = dia removed array

import numpy as np
import scipy.sparse as sp

adj = np.array([[1, 2, 3, 4], [2, 2, 1, 2], [3, 1, 3, 1], [4, 2, 1, 4]])
print(adj)
# [[1 2 3 4]
#  [2 2 1 2]
#  [3 1 3 1]
#  [4 2 1 4]]
print(adj.diagonal())
# [1 2 3 4]


print(adj.diagonal()[np.newaxis, :])
# [[1 2 3 4]]
print("---------------------")
print(sp.dia_matrix(
    (adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape))
print("---------------------")
# (0, 0)	1
# (1, 1)	2
# (2, 2)	3
# (3, 3)	4
print(adj -
      sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape))
# [[0 2 3 4]
#  [2 0 1 2]
#  [3 1 0 1]
#  [4 2 1 0]]


print(adj.diagonal()[:, np.newaxis])
# [[1]
#  [2]
#  [3]
#  [4]]


# data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
# print(data)  # , ..., sep, end, file, flush)
