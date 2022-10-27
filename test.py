# -*- coding: utf-8 -*-
# file: test.py
# time: 2022/10/11
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import numpy as np


def count_common(A,B):
    assert A.ndim == 1 and B.ndim == 1, 'This function requires 1 dimensional arrays in input'
    # return sum(A == B)
    return np.count_nonzero(A == B)

def find_common(A,B):
    assert A.ndim == 1 and B.ndim == 1, 'This function requires 1 dimensional arrays in input'
    ids = np.where(A == B)[0]
    return A[ids], ids

A = np.arange(11)
B = A[::-1].copy()
print(A)
print(B)
out = count_common(A,B)

assert out == 1, 'the function should return 1'

out = find_common(A,B)
assert len(out) == 2, 'the function should return two values'
vals, ids = out
assert np.all(vals == [5]), 'the function should return 5'
assert np.all(ids == [5]), 'the function should return 5'

a = np.nonzero(np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]]))


def cutoff(A, lower, upper):
    B = A.copy()
    B[np.where(A>upper)]=upper
    B[np.where(A<lower)]=lower
    return B

A = np.arange(25).reshape(5, 5)
print(A)
B = cutoff(A, 5, 10)
print(B)
assert np.all(B[0] == [5]*5), 'the function should return a first row of 5s'
assert np.all(B[-1] == [10]*5), 'the function should return a last row of 10s'

def rank(A):
    assert A.ndim == 1, 'This function requires 1 dimensional arrays in input'
    return A.argsort().argsort()

A = np.arange(6)[::2]+0.3
B = np.arange(1,6)[::2]-0.3
C = np.hstack([A,B])
print(C)
out = rank(C)
print(out)

assert np.all(out == [0, 2, 4, 1, 3, 5]), 'the function should return [0 2 4 1 3 5]'

print('1)', C)
print('these are the indices that would sort the array')
print('2) argsort:',C.argsort())
print('i.e. to obtain:')
print('3)',C[C.argsort()])

print('What we want is the rank for the element in each position')
print('4) rank:',rank(C))
print('For example, we see that element with index 4 occurs in position 3 in 2)')
print('so we know that the rank of element with index 4 is 3')
print('argsort of 2) gives the index that would sort 2) in the order 0,1,2,3,4 ')
print('so asking argsort of 2), we are asking: which is the id that will go in position 4 (the 4 will go in position 4)?')
print(' it is 3')
print('which is the id that will go in position 3 (the 3 will go in position 3)')
print(' it is 1')