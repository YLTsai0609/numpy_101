# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## There are alternative solution and hits: 
# ### more readable
#  > 26, 27, 38
# ### more effient (vectorlized)
#  > 40
#  ### when to use it?
#  > 9, 15, 16
#  ### hints
#  > 5, 7, 20, 21, 24, 30, 40, 42

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

# -


