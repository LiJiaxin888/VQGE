import numpy as np
from vqgespy import *
import scipy.linalg as la
from scipy.linalg import eig
from scipy.io import loadmat
# from pyvqges import *

def make_matrix(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.complex128))


A = np.load('matrixA.npy')
B = np.load('matrixB.npy')

result = VQGES(A, B)


print(result)