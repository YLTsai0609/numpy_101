# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# ## There are alternative solution and hits: 
# ### more readable
#  > 26, 27, 38, 48, 49, 50, 52, 53, 63, 67
# ### more effient (vectorlized)
#  > 40
#  ### when to use it?
#  > 9, 15, 16, 44, 45, 46
#  ### hints
#  > 5, 7, 20, 21, 24, 30, 40, 42, 43, 45, 66

# # Outline
# * NumPy 高速運算徹底解說 
#
#
# Ch01 NumPy 的基礎
#
# 1-1 認識 NumPy 的基本操作
#
# 1-2 ndarray 多維陣列的基本概念
#
# 1-3 ndarray 的軸 (axis) 與維度 (dimension)
#
# 1-4 ndarray 的 dtype 屬性
#
# 1-5 ndarray 的切片 (Slicing) 操作
#
# 1-6 陣列擴張 (Broadcasting)
#
# Ch02 NumPy基本運算函式
#
# 2-1 陣列重塑 - reshape()、resize()
#
# 2-2 在陣列最後面加入元素 – append()
#
# 2-3 判斷陣列真假值 – all()、any()
#
# 2-4 找出符合條件的元素 – where()
#
# 2-5 取出最大值、最小值 – amax()、amin()
#
# 2-6 取出最大值、最小值的索引位置 – argmax()、argmin()
#
# 2-7 陣列轉置 – transpose()
#
# 2-8 陣列排序 – sort() 與 argsort()
#
# 2-9 陣列合併 – vstack()、hstack()
#
# 2-10 建立元素都是 0 的陣列 – zeros()
#
# 2-11 建立元素都是 1 的陣列 – ones()
#
# 2-12 建立「不限定元素值」的陣列 – empty()
#
# 2-13 建立指定範圍的等差陣列 – arange()
#
# 2-14 建立指定範圍的等差陣列 – linspace()
#
# 2-15 建立單位矩陣 – eye()、identity()
#
# 2-16 將陣列展平為 1D 陣列 – flatten()
#
# 2-17 將陣列展平為 1D 陣列 – ravel()
#
# 2-18 找出不是 0 的元素 – nonzero()
#
# 2-19 複製陣列元素, 拚貼成新陣列 – tile()
#
# 2-20 增加陣列的軸數 – np.newaxis
#
# 2-21 陣列合併 – np.r_ 與 np.c_ 物件
#
# 2-22 陣列的儲存與讀取 – save() 與 load()
#
# 2-23 以文字格式儲存、讀取陣列內容 – savetxt() 與 loadtxt()
#
# 2-24 建立隨機亂數的陣列 – random 模組
#
# Ch03 NumPy 的數學函式
#
# 3-1 基本的數學運算函式
#
# 3-2 計算元素平均值 – average() 與 mean()
#
# 3-3 計算中位數 – median()
#
# 3-4 計算元素總和 – sum()
#
# 3-5 計算標準差 – std()
#
# 3-6 計算變異數 – var()
#
# 3-7 計算共變異數 – cov() 
#
# 3-8 計算相關係數 – corrcoef()
#
# 3-9 網格陣列 – meshgrid()
#
# 3-10 點積運算 – dot()
#
# 3-11 計算矩陣的 determinant – linalg.det()
#
# 3-12 計算矩陣的「特徵值」與「特徵向量」 – linalg.eig()
#
# 3-13 計算矩陣的 rank – linalg.matrix_rank()
#
# 3-14 計算矩陣的「反矩陣」 – linalg.inv() 
#
# 3-15 計算張量積 – outer()
#
# 3-16 計算叉積 – cross() 
#
# 3-17 計算卷積 – convolve()
#
# 3-18 將連續值轉換為離散值 – digitize()
#
#

import numpy as np
np.random.seed(seed=42)

# +
# 1. Import numpy as np and see the version
# Difficulty Level: L1

# Q. Import numpy as `np` and print the version number.

np.__version__

# +
# 2. How to create a 1D array?
# Difficulty Level: L1

# Q. Create a 1D array of numbers from 0 to 9

np.arange(10)

# +
# 3. How to create a boolean array?
# Difficulty Level: L1

# Q. Create a 3×3 numpy array of all True’s

np.full((3,3), True)

# +
# 4. How to extract items that satisfy a given condition from 1D array?
# Difficulty Level: L1

# Q. Extract all odd numbers from arr
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

condintion = arr % 2 == 1
arr[condintion]

# +
# 5. How to replace items that satisfy a condition with another value in numpy array?
# Difficulty Level: L1

# Q. Replace all odd numbers in arr with -1

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.where(arr % 2 == 1, -1, arr)

# Hint
# 4,5題比較，如果只想要部分符合條件的結果 - 用 boolean array
# 如果想要整個array的結果, 符合與不符合分別map到不同的值 - 用 np.where
# When to use
# Feature engineering, 根據條件來變更array的值, 非常常用


# +
# 6. How to replace items that satisfy a condition without affecting the original array?
# Difficulty Level: L2
# Q. Replace all odd numbers in arr with -1 without changing arr

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

result = np.where(arr % 2 == 1, -1, arr)
print(arr, result, sep='\n')

# +
# 7. How to reshape an array?
# Difficulty Level: L1

# Q. Convert a 1D array to a 2D array with 2 rows

arr = np.arange(10)

arr.reshape(2, -1)

# hint 
# -1 表示剩餘的行/列可以被決定 在此例中，我們已經決定rows = 2
# 則 columns 為 10 / 2 = 5

# +
# 8. How to stack two arrays vertically?
# Difficulty Level: L2

# Q. Stack arrays a and b vertically

a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)

np.vstack([a, b])

# +
# 9. How to stack two arrays horizontally?
# Difficulty Level: L2

# Q. Stack the arrays a and b horizontally.

a = np.arange(10).reshape(2,-1)

b = np.repeat(1, 10).reshape(2,-1)

np.hstack([a, b])

# when to use?
# 在迴圈中，例如Cross-validation, 做陣列操作, 例如合併特徵重要度,
# 或是拼接特徵

# +
# 10. How to generate custom sequences in numpy without hardcoding?
# Difficulty Level: L2

# Q. Create the following pattern without hardcoding. Use only numpy functions and the below input array a.

a = np.array([1,2,3])

result = np.hstack([np.repeat(a, 3),
                    np.tile(a, 2)])
result

# +
# 11. How to get the common items between two python numpy arrays?
# Difficulty Level: L2

# Q. Get the common items between a and b

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.intersect1d(a, b)

# +
# 12. How to remove from one array those items that exist in another?
# Difficulty Level: L2

# Q. From array a remove all items present in array b

a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

intersect = np.intersect1d(a, b)
b[~np.isin(b, a)]


# +
# 13. How to get the positions where elements of two arrays match?
# Difficulty Level: L2
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.argwhere(a==b).reshape(-1)

# +
# 14. How to extract all numbers between a given range from a numpy array?
# Difficulty Level: L2
a = np.array([2, 6, 1, 9, 10, 3, 27])

idx = np.argwhere((5 <= a) & (a<=10)).reshape(-1)
a[idx]


# +
# 15. How to make a python function that handles scalars to work on numpy arrays?
# Difficulty Level: L2
def maxx(x, y):
    """Get the maximum of two items"""
    if x >= y:
        return x
    else:
        return y
v_maxx = np.vectorize(maxx, otypes=[int])

# when to use
# you want faster speed but no np function suit your case
# then define it, vectorlize it
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
v_maxx(a, b)
# -

# 16. How to swap two columns in a 2d numpy array?
# Difficulty Level: L2
arr = np.arange(9).reshape(3,3)
# when to use
# change your feature values if they are numpy.array
arr[:, [1,0,2]]

# 17. How to swap two rows in a 2d numpy array?
# Difficulty Level: L2
arr = np.arange(9).reshape(3,3)
arr[[1,0,2], :]

# 18. How to reverse the rows of a 2D array?
# Difficulty Level: L2
arr = np.arange(9).reshape(3,3)
arr[::-1, :]

# 19. How to reverse the columns of a 2D array?
# Difficulty Level: L2
arr = np.arange(9).reshape(3,3)
arr[:,::-1]

# +
# 20. How to create a 2D array containing random floats between 5 and 10?
# Difficulty Level: L2

# sol1
sol1 = np.random.randint(5, 10, size=(5,3)) + np.random.random(size=(5,3))
print(sol1)
sol2 = np.random.uniform(low=5, high=10, size=(5,3))
print(sol2)
# hint
# random methods are really make people confused
# we can check the method using dir and list-comprehension
# and it just saved your time
# method_list = dir(np.random)
# [method for method in method_list
#                    if method.startswith('rand')]
# quick note
# random.ranint(low, high, size) --> [low, high) interger only
# random.random_integers(low, high, size) --> [low, high] interger only
# random.uniform(low, high, size) --> [low, high) float
# random.random(size) --> [0,1) float
# https://www.jianshu.com/p/214798dd8f93 簡書上的numpy.random相關函數整理，作者推薦

# +
# 21. How to print only 3 decimal places in python numpy array?
# Difficulty Level: L1
rand_arr = np.random.random((5,3))
# sol1
np.round(rand_arr,3)
# sol2
np.set_printoptions(precision=3)
print(rand_arr)

# hint
# back_to_fefault option (by numpy documentation)
np.set_printoptions(edgeitems=3,infstr='inf',
linewidth=75, nanstr='nan', precision=8,
suppress=False, threshold=1000, formatter=None)

# +
# 22. How to pretty print a numpy array by suppressing the scientific notation (like 1e10)?
# Difficulty Level: L1
rand_arr = np.random.random([3,3])/1e4
rand_arr
np.set_printoptions(suppress=True, precision=5)
rand_arr

# hint
# back_to_fefault option (by numpy documentation)
np.set_printoptions(edgeitems=3,infstr='inf',
linewidth=75, nanstr='nan', precision=8,
suppress=False, threshold=1000, formatter=None)

# +
# 23. How to limit the number of items printed in output of numpy array?
# Difficulty Level: L1
a = np.arange(15)
np.set_printoptions(threshold=6)
print(a)

# hint
# back_to_fefault option (by numpy documentation)
np.set_printoptions(edgeitems=3,infstr='inf',
linewidth=75, nanstr='nan', precision=8,
suppress=False, threshold=1000, formatter=None)

# +
# 24. How to print the full numpy array without truncating
# Difficulty Level: L1
a = np.arange(15)
np.set_printoptions(threshold=None)
print(a)

# hint
# back_to_fefault option (by numpy documentation)
np.set_printoptions(edgeitems=3,infstr='inf',
linewidth=75, nanstr='nan', precision=8,
suppress=False, threshold=1000, formatter=None)
# -

# 25. How to import a dataset with numbers and texts keeping the text intact in python numpy?
# Difficulty Level: L2
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
iris[:3]

# +
# 26. How to extract a particular column from 1D array of tuples?
# Difficulty Level: L2

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',',
                        names=['a','b','c','d','txt'],
                        dtype=None)
# More readable
# use names filed 
# documentation about named fields 
# https://docs.scipy.org/doc/numpy-1.15.0/user/basics.rec.html
iris_1d['txt'][:5]


# +
# 27. How to convert a 1d array of tuples to a 2d numpy array?
# Difficulty Level: L2
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)

# More readable
# Just use asarray to unpack the element
result = np.asarray(iris_1d)
result[:5]


# +
# 28. How to compute the mean, median, standard deviation of a numpy array?
# Difficulty: L1

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
iris[:, 0:4] = iris[:, 0:4].astype(float)
print( 'mean ',iris[:, 0].mean())
print('median ', np.median(iris[:, 0]))
print('std ',iris[:, 0].std())


# +
# 29. How to normalize an array so the values range exactly between 0 and 1?
# Difficulty: L2
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

Smax, Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin) / (Smax - Smin)
S[:5]

# +
# 30. How to compute the softmax score?
# Difficulty Level: L3

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
softmax_socre = np.exp(sepallength) / sum(np.exp(sepallength))
softmax_socre[:5]

# hint 
# softmax score --> 一種歸一化, 將原本的元素壓縮在0,1之間, 且所有元素和為1
# 這樣的歸一化能夠強調最大的值, 並使其他值遠小於最大值
# Wiki : https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0
print('The max of softmax score : ',softmax_socre.max(),
      'The mean of softmax score : ', softmax_socre.mean(), sep='\n')
# when to use?
# Logistic regression, NN, Maive Byes ....

# +
# 31. How to find the percentile scores of a numpy array?
# Difficulty Level: L1
# Q. Find the 5th and 95th percentile of iris's sepallength


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
print(np.quantile(sepallength, 0.05))
print(np.quantile(sepallength, 0.95))


# +
# 32. How to insert values at random positions in an array?
# Difficulty Level: L2

# Q. Insert np.nan values at 20 random positions in iris_2d dataset

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

pos_x = np.random.randint(low=0, high=iris_2d.shape[0], size=20)
pos_y = np.random.randint(low=0, high=iris_2d.shape[1], size=20)
iris_2d[pos_x, pos_y] = np.nan
iris_2d[:10]

# +
# 33. How to find the position of missing values in numpy array?
# Difficulty Level: L2
# Q. Find the number and position of missing values in iris_2d's sepallength (1st column)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan


# +
# 34. How to filter a numpy array based on two or more conditions?
# Difficulty Level: L3

# Q. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
iris_2d[condition][:5]


# +
# 35. How to drop rows that contain a missing value from a numpy array?
# Difficulty Level: L3:
# Q. Select the rows of iris_2d that does not have any nan value.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# mask or np.where
# np.isnan will flattern our 2d-array
# so followed will return shape = 600
# condintion = ~ np.isnan(iris_2d)
# iris_2d[condintion].shape

# then we do it by rowise
mask = [~ np.any(np.isnan(row)) for row in iris_2d]
print(iris_2d[mask].shape)
iris_2d[mask][:5]

# +
# 36. How to find the correlation between two columns of a numpy array?
# Difficulty Level: L2
# Q. Find the correlation between SepalLength(1st column) and PetalLength(3rd column) in iris_2d

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

np.corrcoef(iris_2d[:, 0], iris_2d[:, 2])

# +
# 37. How to find if a given array has any null values?
# Difficulty Level: L2
# iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

np.isnan(iris_2d).any()



# -



# +
# 38. How to replace all missing values with 0 in a numpy array?
# Difficulty Level: L2

# Q. Replace all ccurrences of nan with 0 in numpy array

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan


# More readable 
iris_2d = np.where(np.isnan(iris_2d), 0, iris_2d)
print(np.isnan(iris_2d).any())


# +
# 39. How to find the count of unique values in a numpy array?
# Difficulty Level: L2

# Q. Find the unique values and the count of unique values in iris's species

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')


np.unique(iris[:, -1], return_counts=True)


# +
# 40. How to convert a numeric to a categorical (text) array?
# Difficulty Level: L2

# Q. Bin the petal length (3rd) column of iris_2d to form a text array, such that if petal length is:

# Less than 3 --> 'small'
# 3-5 --> 'medium'
# '>=5 --> 'large'

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Vectorlized

bins = np.array([0, 3, 5, 100])
def label_map(x):
    if x == 1:
        return 'small'
    elif x == 2:
        return 'medium'
    elif x == 3:
        return 'large'
    else:
        return np.nan
tmp = np.digitize(iris[:, 2].astype(float), bins)
v_label_map = np.vectorize(label_map)
v_label_map(tmp)[:5]

# Hint
# the comparision below when your data is really large
# use dictionary and list-comprehension



# +
# test_arr = np.random.randint(low=1, high=4, size=5000000)
# -

# # %%time
# label_map = {1 : 'small', 2 : 'medium', 3:'large'}
# result = [label_map[element] for element in test_arr]
# took 1.71s on my mac-air


# +
# # %%time
# def label_map(x):
#     if x == 1:
#         return 'small'
#     elif x == 2:
#         return 'medium'
#     elif x == 3:
#         return 'large'
#     else:
#         return np.nan
# v_label_map = np.vectorize(label_map)
# v_label_map(tmp)
# took 191 µs o my mac-air

# A brief report no stackoverflow
# https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
# -





# +
# 41. How to create a new column from existing columns of a numpy array?
# Difficulty Level: L2

# Q. Create a new column for volume in iris_2d, where volume is (pi x petallength x sepal_length^2)/3

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
pi = np.pi

result = pi * iris_2d[:, 2].astype(float) * (iris_2d[:, 0].astype(float) ** 2) / 3
result[:5]

# +
# 42. How to do probabilistic sampling in numpy?
# Difficulty Level: L3
    
# Q. Randomly sample iris's species such that setose is twice the number of versicolor and virginica
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')


spcies_unq = np.unique(iris[:, -1])
p = [0.5, 0.25, 0.25]
result = np.random.choice(spcies_unq, size=100, p=p)
print('result array distribution :',np.unique(result, return_counts=True))

# hint
# you could see sampling documentation 
# https://docs.scipy.org/doc/numpy-1.16.0/reference/routines.random.html

# +
# 43. How to get the second largest value of an array when grouped by another array?
# Difficulty Level: L2

# Q. What is the value of second longest petallength of species setosa

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

condition = iris[:, -1] == b'Iris-setosa'

tmp = iris[condition, 2].astype(float)
np.unique(np.sort(tmp))[-2]

# hint
# numpy中並沒有像pd.Series一樣有ascending選項


# +
# 44. How to sort a 2D array by a column
# Difficulty Level: L2

# Q. Sort the iris dataset based on sepallength column.


# More readable
# use name-filed
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
dtype=[('sepallength', '<f8'), ('sepalwidth', '<f8'),
       ('petallength', '<f8'), ('petalwidth', '<f8')
       ,('species', 'object')]

iris = np.genfromtxt(url, delimiter=',',names=names,
                    dtype = dtype)
np.sort(iris, order=['sepallength'])[:5]

# +
# 45. How to find the most frequent value in a numpy array?
# Difficulty Level: L1

# Q. Find the most frequent value of petal length (3rd column) in iris dataset.

# More readable
# name field
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
dtype=[('sepallength', '<f8'), ('sepalwidth', '<f8'),
       ('petallength', '<f8'), ('petalwidth', '<f8')
       ,('species', 'object')]

iris = np.genfromtxt(url, delimiter=',',names=names,
                    dtype = dtype)

vals_arr, counts_arr = np.unique(iris['petallength'], return_counts=True)

vals_arr[np.argmax(counts_arr)]

# Hint
# 函數回傳結果為turple時，
# 用兩個變數接收結果

# +
# 46. How to find the position of the first occurrence of a value greater than a given value?
# Difficulty Level: L2

# Q. Find the position of the first occurrence of a value greater than 1.0 in petalwidth 4th column of iris dataset.

# More readable
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
dtype=[('sepallength', '<f8'), ('sepalwidth', '<f8'),
       ('petallength', '<f8'), ('petalwidth', '<f8')
       ,('species', 'object')]

iris = np.genfromtxt(url, delimiter=',',names=names,
                    dtype = dtype)

np.argwhere(iris['petalwidth'] > 1.0)[0]

# +
# 47. How to replace all values greater than a given value to a given cutoff?
# Difficulty Level: L2

# Q. From the array a, replace all values greater than 30 to 30 and less than 10 to 10.

np.random.seed(100)
a = np.random.uniform(1,50, 20)

# 巢狀np.where
# 若 a > 30, 則換成30, 若 < 30, 執行另一函式
# 另一函式 : a < 10, 換成 10, 否, 為原本值
# 用途廣泛
np.where( a > 30, 
              30, 
          np.where(a < 10,
                       10,
                       a))
# clip
# when to use
# 純數字，取中間時
np.clip(a, a_min=10, a_max=30)[:5]

# +
# 48. How to get the positions of top n values from a numpy array?
# Difficulty Level: L2

# Q. Get the positions of top 5 maximum values in a given array a.

np.random.seed(100)
a = np.random.uniform(1,50, 20)

np.argsort(a)[-5:]

# Hint
# 條件 : 找值
# np.where 返回整個array的值
# np.argwhere 返回index
# 排序
# np.sort 值排序
# np.argsort 值排序，返回index
# When to use
# Eg. 返回特徵重要度最大的index, 用於取出特徵欄位
# +
# 49. How to compute the row wise counts of all possible values in an array?
# Difficulty Level: L4
# Q. Compute the counts of unique values row-wise.
np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))

# More readable
result = np.zeros_like(arr)
for row in range(arr.shape[0]):
    unq_arr, count_arr = np.unique(arr[row, :], return_counts=True)
    result[row, :][unq_arr - 1] = count_arr


result.tolist()


# +
# 50. How to convert an array of arrays into a flat 1d array?
# Difficulty Level: 2

# Q. Convert array_of_arrays into a flat linear 1d array.
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)

array_of_arrays = np.array([arr1, arr2, arr3])
# More readable
np.hstack(array_of_arrays)

# +
# 51. How to generate one-hot encodings for an array in numpy?
# Difficulty Level L4

# Q. Compute the one-hot encodings (dummy binary variables for each unique value in the array)
np.random.seed(101) 
arr = np.random.randint(1,4, size=6)
one_hot_feature = np.unique(arr)
one_hot_arr = np.zeros(shape=(arr.shape[0], one_hot_feature.shape[0]))

for i, k in enumerate(arr):
    one_hot_arr[i, k-1] = 1

one_hot_arr
# +
# 52. How to create row numbers grouped by a categorical variable?
# Difficulty Level: L3

# Q. Create row numbers grouped by a categorical variable. Use the following sample from iris species as input.


species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))
species_small

np.unique(species_small)

# More readable

def get_groupby_index(arr):
    unq = np.unique(arr)
    result = []
    for species_item in unq:
        filtered_arr = arr[arr == species_item]
        for idx in range(len(filtered_arr)):
            result.append(idx)
    return result

get_groupby_index(species_small)


# +
# 53. How to create groud ids based on a given categorical variable?
# Difficulty Level: L4

# Q. Create group ids based on a given categorical variable. Use the following sample from iris species as input.

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))
species_small

# More readable

def get_groupby_index(arr):
    unq = np.unique(arr)
    result = []
    for group_idx ,species_item in enumerate(unq):
        filtered_arr = arr[arr == species_item]
        group_idx_list = [group_idx for i in filtered_arr]
        result.extend(group_idx_list)
    return result

get_groupby_index(species_small)


# +
# 54. How to rank items in an array using numpy?
# Difficulty Level: L2

# Q. Create the ranks for the given numeric array a.

np.random.seed(10)
a = np.random.randint(20, size=10)
print(a)
a.argsort().argsort()
# -

np.random.seed(10)
a = np.random.randint(20, size=[2,5])
print(a.ravel().argsort().argsort().reshape(a.shape))

# +
# 56. How to find the maximum value in each row of a numpy array 2d?
# DifficultyLevel: L2

# Q. Compute the maximum for each row in the given array.

np.random.seed(100)
a = np.random.randint(1,10, [5,3])

a.max(axis=1)



# +
# 57. How to compute the min-by-max for each row for a numpy array 2d?
# DifficultyLevel: L3

# Q. Compute the min-by-max for each row for given 2d numpy array.
np.random.seed(100)
a = np.random.randint(1,10, [5,3])

a.min(axis=1) / a.max(axis=1)



# +
# 58. How to find the duplicate records in a numpy array?
# Difficulty Level: L3

# Q. Find the duplicate entries (2nd occurrence onwards) in the given numpy array and mark them as True. First time occurrences should be False.
np.random.seed(100)
a = np.random.randint(0, 5, 10)
print('Array: ', a)
result = np.full(a.shape[0], True)
unq, unq_idx = np.unique(a, return_index=True)
result[unq_idx] = False
print(result)
# +
# 59
# Difficulty Level L3

# Q. Find the mean of a numeric column grouped by a categorical column in a 2D numpy array

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
dtype=[('sepallength', '<f8'), ('sepalwidth', '<f8'),
       ('petallength', '<f8'), ('petalwidth', '<f8')
       ,('species', 'object')]

iris = np.genfromtxt(url, delimiter=',',names=names,
                    dtype = dtype)

# More readable
def groupby_numpy_mean(arr, by, caculate_col):
    uniuqe_species = np.unique(iris[by])
    result = []
    for unq in uniuqe_species:
        unq_mean = iris[iris[by] == unq][caculate_col].mean()
        result.append(
        [unq, unq_mean]
        )
    return result

groupby_numpy_mean(iris, by='species', caculate_col='sepallength')

# +
# 60. How to convert a PIL image to numpy array?
# Difficulty Level: L3

# Q. Import the image from the following URL and convert it to a numpy array.

URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
from PIL import Image
import requests
from io import BytesIO
response = requests.get(URL)
image = Image.open(BytesIO(response.content))
image_arr = np.array(image)
print(image_arr.shape)

# Optionaly Convert it back to an image and show
# im = PIL.Image.fromarray(np.uint8(arr))
# Image.Image.show(im)



# +
# 61. How to drop all missing values from a numpy array?
# Difficulty Level: L2

# Q. Drop all nan values from a 1D numpy array

arr = np.array([1,2,3,np.nan,5,6,7,np.nan])
arr[~ np.isnan(arr)]

# +
# 62. How to compute the euclidean distance between two arrays?
# Difficulty Level: L3

# Q. Compute the euclidean distance between two arrays a and b.

a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])
a = a.reshape(1, -1)
b = b.reshape(1, -1)
from scipy.spatial.distance import cdist as dist
dist(a,b, metric='euclidean')

# +
# 63. How to find all the local maxima (or peaks) in a 1d array?
# Difficulty Level: L4

# Q. Find all the peaks in a 1D numpy array a. Peaks are points surrounded by smaller values on both sides.

a = np.array([1, 3, 7, 1, 2, 6, 0, 1])


# More readable
def get_peak(arr, forward_diff,back_diff):
    def get_backword(arr, back_diff):
        import pandas as pd 
        return pd.Series(arr).diff(-1).values
    forward = np.sign(np.diff(a, n=forward_diff, prepend=np.nan) )
    backward = np.sign(get_backword(arr, back_diff))
    return np.argwhere((forward > 0) & (backward > 0)).reshape(-1)
    
peak_pos = get_peak(a, forward_diff=1, back_diff=1)
peak_pos

# +
# 64. How to subtract a 1d array from a 2d array, where each item of 1d array subtracts from respective row?
# Difficulty Level: L2

# Q. Subtract the 1d array b_1d from the 2d array a_2d, such that each item of b_1d subtracts from respective row of a_2d.

a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,1,1])
a_2d - b_1d



# +
# 65. How to find the index of n'th repetition of an item in an array
# Difficulty Level L2

x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])

def get_n_repetition_idx(arr, value, capture):
    tmp = x.copy()
    value_repetition = np.cumsum(x[x == value])
    result = np.argwhere(value_repetition == capture).reshape(-1)
    return result
get_n_repetition_idx(x, 1, 5)

# +
# 66. How to convert numpy's datetime64 object to datetime's datetime object?
# Difficulty Level: L2

# Q. Convert numpy's datetime64 object to datetime's datetime object
from datetime import datetime
dt64 = np.datetime64('2018-02-25 22:10:10')
dt64.astype(datetime)

# Hint
# 使用 print(dir(dt64)) 來找出可使用的方法及屬性
# datetime物件經常是很麻煩的處理, numpy的documentation可以在這裡找到
# https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.datetime.html


# +
# 67. How to compute the moving average of a numpy array?
# Difficulty Level: L3

# Q. Compute the moving average of window size 3, for the given 1D array.

np.random.seed(100)
Z = np.random.randint(10, size=10)

# more readable
def moving_average(arr, window):
    result = np.array([])
    for idx in range(arr.shape[0] - window):
        span = np.arange(idx, idx + window)
        result = np.append(result, arr[span].mean())
    return result
moving_average(Z, window=2)


# +
# 68. How to create a numpy array sequence given only the starting point, length and the step?
# Difficulty Level: L2
# Q. Create a numpy array of length 10, starting from 5 and has a step of 3 between consecutive numbers

def generatre_arr(start, step, length):
    stop = start + step*(length-1)
    return np.linspace(start=5, stop=stop, num=10)

generatre_arr(start=5, step=3, length=10)

# +
# 69. How to fill in missing dates in an irregular series of numpy dates?
# Difficulty Level: L3

# Q. Given an array of a non-continuous sequence of dates. Make it a continuous sequence of dates, by filling in the missing dates.
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)

import pandas as pd
start, end = dates[0], dates[-1]
pidx = pd.period_range(start=start, end=end, freq='D')
result = pidx.astype(np.datetime64).values
print(result)

# sol2

result = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-24'), 1)
print(result)

# +
# 70. How to create strides from a given 1D array?
# Difficulty Level: L4

# Q. From the given 1d array arr, generate a 2d matrix using strides, with a window length of 4 and strides of 2, like [[0,1,2,3], [2,3,4,5], [4,5,6,7]..]
arr = np.arange(15) 
arr

# More readable
def get_strides(arr, window_length, strides):
    result = []
    end_point_head_idx = len(arr) - window_length
    for head_idx in range(0, end_point_head_idx, strides):
        idx_arr = np.arange(0 + head_idx, head_idx + window_length)
        result.append(arr[idx_arr].tolist())
    return result

get_strides(arr, window_length=4, strides=2)
# -

# # Further reading 
#
# * NumPy 高速運算徹底解說 
#
# ## Outline
#
# Ch04 NumPy 的實務應用
#
# 4-1 資料的正規化 (Normalization)
#
# 4-1-1 z 分數正規化
#
# 4-1-2 最小值 - 最大值正規化
#
# 4-2 迴歸分析實作
#
# 4-2-1 迴歸概念解說 (簡單線性迴歸、多項式迴歸)
#
# 4-2-2 Step1：建立 20 個 (x, y) 組合的座標點
#
# 4-2-3 Step2：餵資料給機器學習, 求出能逼近 20 個點的迴歸方程式
#
# 4-2-4 Step3：完成學習, 驗證結果
#
#
# 4-3 機器學習實戰 (一)：使用神經網路替鳶尾花分類
#
# 4-3-1 神經網路的基本概念 (神經元、啟動函數、損失函數)
#
# 4-3-2 Step1：備妥訓練所需的資料
# 資料預處理
# 將資料集拆分為「訓練資料集」與「測試資料集」
#
# 4-3-3 Step2：開始訓練神經網路
# 訓練神經網路
# 更新權重
# 進入下一週期的訓練、更新權重
#
# 4-3-4 Step3：完成訓練, 驗證結果
#
# 4-4 機器學習實戰 (二)：使用神經網路辨識手寫數字圖片
#
# 4-4-1 多層神經網路的概念
#
# 4-4-2 Step1：備妥訓練所需的資料
#
# 4-4-3 Step2：開始訓練神經網路
# 了解反向傳播之前要先了解前向傳播 
# 以計算圖呈現損失函數 L 的算式
# 利用「反向傳播」求損失函數 L 對各權重的偏微分
# 開始訓練神經網路
#
# 4-4-4 Step3：完成訓練, 驗證結果
#
# 4-5 使用 NumPy 實作強化學習
#
# 4-5-1 OpenAI Gym 是什麼
#
# 4-5-2 安裝與執行遊戲
#
# 4-5-3 用 Q-Learning 實作強化學習
#
# 4-5-4 增進 Q-Learning 的學習成效
#
# 4-5-5 策略梯度法


