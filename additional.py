# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from scipy.sparse import csr_matrix
import numpy as np

# +
# 1. creat Compressed Sparse Row matrix (csr)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

# create element = 0, shape = (3, 4), dtype=np.int8 csr matrix
sp_1 = csr_matrix((3, 4), dtype=np.int8)

# create 
# element = [1,2,3,4,5,6]
# position = [(0,0), (0,2), (1,2), (2,0), (2,1), (2,2)]
# dtype default = np.int64
# csr matrix
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
sp_2 = csr_matrix((data, (row, col)), shape=(3, 3))

# create 
# data = [1,2,3,4,5,6]
# column indices = [0,2,2,0,1,2]
# index pointer = [0,2,3,6] 
indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
sp_3 = csr_matrix((data, indices, indptr), shape=(3, 3))

sp_1, sp_1.toarray(), sp_2, sp_2.toarray(), sp_3, sp_3.toarray()
# -

# **2 what ia index pointer?**
#
# [ref](https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr)
#
# **what is it?**
#
# 1. If the sparse matrix has $M$ rows, `indptr` is an array contains $M+1$ elements
# 2. for row **i**, `[ indptr[i]:indptr[i+1] ]` returns the indices of element take from `data` and `indices` corresponding to row **i**
#
# <img src='./assets/scipy_1.png'></img>
#
# e.g. 
#
# `data = [a,b,c,d,e,f,g,h,i,j,k]`, `n_elements = 11` 
#
# `indices(columns) = [5,1,6,3,3,4,5,0,5,3,5]`, `n_elements = 11`
#
# `indptr = [0,1,3,4,7,8,9,11]`, `n_elements = 8` -> means there is 7 rows(row0 ~ row6)
#
# all right, how to fill the values?
#
# 1. `index[0:1]` row 0 filling data **a** at indices(columns) **5**
# 2. `index[1:3]` row 1 filling data **b, c** at indices(columns) **1, 6**
# 3. `index[3:4]` row 2 filling data **d** at indices(columns) **3**
# 4. `index[4:7]` row 3 filling data **e, f, g** at indices(columns) **3, 4, 5**
# 5. `index[7:8]` row 4 filling data **h** at indices(columns) **0**
# 6. `index[8:9]` row 5 filling data **i** at indices(columns) **5**
# 7. `index[9:11]` row 6 filling data **j, k** at indices(columns) **3, 5**
#
# Note : the values `indptr` are necessarily increasing and unique
#
# **why use this?**
#
# you can observse that the index stored size is more smaller than
#
# `data, row, column` approach. 
#
# use it when you need it.
#
#

# +
# 3 
# sampling the csr matrix by row
# Consider csr matrix with shape (User, Items) = (10, 20)
# Random Sampling 3 users, return a csr matrix format

# https://stackoverflow.com/questions/13843352/what-is-the-fastest-way-to-slice-a-scipy-sparse-matrix
########## Consider efficient slicing row index of csr
# Advantages of the CSR format
# efficient arithmetic operations CSR + CSR, CSR * CSR, etc.

# efficient row slicing

# fast matrix vector products
########## Begin
# element = [1, 1, 1, 1, 1, 1] - have interacted items
# position = [(0,0), (0,2), (1,2), (2,0), (2,1), (2,2)]
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 1, 1, 1, 1, 1])
A = csr_matrix((data, (row, col)), shape=(10, 20))

n_sampled_users = 3
n_total_users = A.shape[0]
rdn_user_idx = np.random.choice(np.arange(n_total_users),
                                size=n_sampled_users,replace=False) # without replacement
print(rdn_user_idx)
A[rdn_user_idx, :]

# +
# 4 
# the attributes indptr and indices
# element = [1, 1, 1, 1, 1, 1] - have interacted items
# position = [(0,0), (0,2), (1,2), (2,0), (2,1), (2,2)]
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 1, 1, 1, 1, 1])
A = csr_matrix((data, (row, col)), shape=(3,3))
display(A.todense())


for user in range(A.shape[0]):
        start_usr = A.indptr[user]
        end_usr = A.indptr[user + 1]
        print(A.indices[start_usr:end_usr]) # this will show the non zero column per user
# -


