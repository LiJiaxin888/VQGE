import numpy as np
from vqgespy import *
import scipy.linalg as la
from scipy.linalg import eig
from scipy.io import loadmat
# from pyvqges import *

def make_matrix(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.complex128))


data1 = loadmat("A_proj.mat")
data2 = loadmat("B_proj.mat")
A = data1["A_proj"]
B = data2["B_proj"]

result = VQGES(A, B)


print(result)