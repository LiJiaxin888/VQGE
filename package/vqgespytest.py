import numpy as np
from vqgespy import *
import scipy.linalg as la
from scipy.linalg import eig
from scipy.io import loadmat
import numpy as np
import pytest
import numpy.testing as npt


def test_Eigen_model1():
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    eigvals, eigvecs = la.eig(A, B)
    AA, BB, N = VQGESpreprocess(A, B)
    r,c = A.shape
    oldN = r
    learn_rate = np.array([1e-2, 1e-4, 1e-1, 1, 5])
    result = GeneralAnsatzGES(AA, BB, 6, DifferentStep = 0.0000008, LearnRateVector = learn_rate , LearnRateTrainSize= 5 ,LearnRateTrainOffset = 1, LossMonitor= 1e-8, useNoise = False , SingleGateNoise = 0.001, DoubleGateNoise = 0.003)
    result = VQGESpostprocess(result,oldN, N)

    sorted_vec = sorted(result, key=lambda x: (x.real, x.imag), reverse=True)
    sorted_eigvals = sorted(eigvals, key=lambda x: (x.real, x.imag), reverse=True)

    npt.assert_allclose(sorted_eigvals, sorted_vec, rtol=1e-2, atol=1e-3)


def test_Eigen_model2():
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    eigvals, eigvecs = la.eig(A, B)
    AA, BB, N = VQGESpreprocess(A, B)
    r,c = A.shape
    oldN = r
    learn_rate = np.array([1e-2, 1e-4, 1e-1, 1, 5])
    result = GeneralAnsatzGES(AA, BB, 6, DifferentStep = 0.0000008, LearnRateVector = learn_rate , LearnRateTrainSize= 5 ,LearnRateTrainOffset = 1, LossMonitor= 1e-8, useNoise = True , SingleGateNoise = 1e-6, DoubleGateNoise = 1e-8)
    result = VQGESpostprocess(result,oldN, N)

    sorted_vec = sorted(result, key=lambda x: (x.real, x.imag), reverse=True)
    sorted_eigvals = sorted(eigvals, key=lambda x: (x.real, x.imag), reverse=True)

    npt.assert_allclose(sorted_eigvals, sorted_vec, rtol=1e-2, atol=1e-3)

def test_Eigen_model3():
    A = np.random.rand(32, 32)
    B = np.random.rand(32, 32)
    eigvals, eigvecs = la.eig(A, B)
    AA, BB, N = VQGESpreprocess(A, B)
    r,c = A.shape
    oldN = r
    learn_rate = np.array([1e-2, 1e-4, 1e-1, 1, 5])
    S, T, Q, Z = la.qz(AA, BB, output='complex')
    result = optimizeAnsatzGES(AA, BB, Q, Z, eigvals, LossMonitor= 1e-15)
    # result = GeneralAnsatzGES(AA, BB, 6, DifferentStep = 0.0000008, LearnRateVector = learn_rate , LearnRateTrainSize= 5 ,LearnRateTrainOffset = 1, LossMonitor= 1e-8, useNoise = True , SingleGateNoise = 1e-6, DoubleGateNoise = 1e-8)
    result = VQGESpostprocess(result,oldN, N)

    sorted_vec = sorted(result, key=lambda x: (x.real, x.imag), reverse=True)
    sorted_eigvals = sorted(eigvals, key=lambda x: (x.real, x.imag), reverse=True)

    npt.assert_allclose(sorted_eigvals, sorted_vec, rtol=1e-2, atol=1e-3)