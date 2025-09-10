import numpy as np
from vqgespy import *
import scipy.linalg as la
from scipy.linalg import eig
from scipy.io import loadmat
# from pyvqges import *

def make_matrix(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.complex128))


A = np.array([[-0.846053,  -3.121318,  1.130982, -0.135525],[-0.274860, 0.540084, 0.832479, 0.530499],[-0.135770, 0.613640 , 0.947157, -0.638468],[1.730607 ,-1.242851, -2.299600, 0.060833]])
B = np.array([[0.217329,  0.418199,  1.206862, 1.458747],[-0.208682 , -1.124809, 0.288132, 2.032686],[1.272089, -0.145261, 1.799622, 1.183555],[0 ,0, 0, 0]])

result = VQGES(A, B)


print(result)