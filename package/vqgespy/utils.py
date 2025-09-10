# from pyvqges import *

from numpy.linalg import norm
from pyqpanda3.core import *
import random
import numpy as np
import math
import scipy.linalg as la
from scipy.linalg import eig
from pyvqges import *



def bitwise_difference(i, j, bit_count):
    """
    Generate an array of length bit_count, where each element indicates whether the corresponding bits of i and j differ.

    Parameters:
    i (int): Unsigned integer input i.
    j (int): Unsigned integer input j.
    bit_count (int): Number of bits to consider.

    Returns:
    list: An array of length bit_count. 0 indicates the bits are identical, 1 indicates the bits differ.
    """
    if i < 0 or j < 0:
        raise ValueError("Inputs i and j must be non-negative integers.")
    
    if bit_count <= 0:
        raise ValueError("bit_count must be a positive integer.")
    
    result = []
    for k in range(bit_count):
        bit_i = (i >> k) & 1
        bit_j = (j >> k) & 1
        # Append 1 if different, otherwise 0
        result.append(1 if bit_i != bit_j else 0)
    
    return result[::-1]  # Reverse to make the array match binary representation order

def create_ansatz_circuit4(qubit_num, theta):
    n = len(theta)
    m = n // qubit_num  # Number of columns per row
    # print(theta)
    AnsatzsCircuit = QCircuit()
    index = 0
    for i in range(qubit_num):
        sub_list = theta[index:index + m]
        AnsatzsCircuit << RZ(i,sub_list[0]) << RY(i,sub_list[1]) << RZ(i,sub_list[2]) 
        index += m

    for i in range(qubit_num-1):
        AnsatzsCircuit << CNOT(i,i+1)
	
    AnsatzsCircuit << CNOT(qubit_num-1,0)
    # print(AnsatzsCircuit)
    return AnsatzsCircuit

def create_WV_Circuit(qubit_num, theta):
    # W_AnsatzsCircuit = create_ansatz_circuit(qubit_num, theta[:len(theta)//2])
    # V_AnsatzsCircuit = create_ansatz_circuit(qubit_num, theta[len(theta)//2:])
    W_AnsatzsCircuit = create_ansatz_circuit4(qubit_num, theta[:len(theta)//2])
    V_AnsatzsCircuit = create_ansatz_circuit4(qubit_num, theta[len(theta)//2:])
    # W_AnsatzsCircuit = create_ansatz_circuit_1(qubit_num, theta[:len(theta)//2])
    # V_AnsatzsCircuit = create_ansatz_circuit_1(qubit_num, theta[len(theta)//2:])
    return W_AnsatzsCircuit,V_AnsatzsCircuit

def runAnsatzsCircuit(machine, AnsatzsCircuit, qubitNum, noise):
    x_Vec = []
    for i in range(2**qubitNum):
        prog = QProg()
        code = bitwise_difference(i, 0, qubitNum)
        for j in range(qubitNum):
            if code[j] == 1:
                prog << X(j)

        prog << AnsatzsCircuit

        machine.run(prog, 1, noise)
        result = machine.result().get_state_vector()
        x_Vec.append(np.array(result))

    return x_Vec

def runWVCircuit(machine, W_AnsatzsCircuit, V_AnsatzsCircuit , qubitNum, noise):
    x_Vec = runAnsatzsCircuit(machine, W_AnsatzsCircuit, qubitNum, noise)
    y_Vec = runAnsatzsCircuit(machine, V_AnsatzsCircuit, qubitNum, noise)
    return x_Vec, y_Vec

def getLoss(qubitNum, theta, A, B, noise, paraAnsatzs, layerAnsatzs = 1):
    rows, cols = A.shape
    W_AnsatzsCircuit = QCircuit()
    V_AnsatzsCircuit = QCircuit()
    for lay in range(layerAnsatzs):
        start_index = lay * 2 * paraAnsatzs
        end_index = (lay + 1) * 2 * paraAnsatzs
        sub_arr = theta[start_index:end_index]
        tmpAnsatzsCircuitW, tmpAnsatzsCircuitV = create_WV_Circuit(qubitNum, sub_arr)
        W_AnsatzsCircuit << tmpAnsatzsCircuitW
        V_AnsatzsCircuit << tmpAnsatzsCircuitV
	
    machine = CPUQVM()
    x_Vec, y_Vec = runWVCircuit(machine, W_AnsatzsCircuit, V_AnsatzsCircuit , qubitNum, noise)
    
    W = np.array(x_Vec).T
    V = np.array(y_Vec).T

    loss = 0
    T = np.dot(np.dot(W.conj().T,A),V)
    S = np.dot(np.dot(W.conj().T,B),V)
    
    for i in range(rows):   
        for j in range(i + 1, cols):  # Start from the element just above the diagonal

            loss += abs(T[j,i])**2 + abs(S[j,i])**2

    return loss

def transform_to_min_value(x):
    # Normalize the input number x to the range [0, 2π)
    transformed = x % (4 * math.pi)
    # If the transformed value is greater than π, subtract 2π to bring it into the range [-π, π)
    if transformed > math.pi:
        transformed -= 4 * math.pi
    return transformed

def generate_random_floats(seed, n, start=0, end= 4 * np.pi):
    return [random.uniform(start, end) for _ in range(n)]

def getBestTheta(randomNum, qubitNum, A, B, noise, paraAnsatzs, layerAnsatzs = 1, constraints = False):
    theta_list = []
    best_loss = float('inf')  # Initialize to positive infinity
    best_index = 0

    for i in range(randomNum):
        # Assume that generate_random_floats requires an integer parameter to specify the quantity to generate
        
        tmptheta = generate_random_floats(random.randint(0,100000), 2 * paraAnsatzs * layerAnsatzs)  

        if (constraints):    
            for id in range(len(tmptheta)):
                tmptheta[id] = transform_to_min_value(tmptheta[id])

        theta_list.append(tmptheta)
        tmploss = getLoss(qubitNum, tmptheta, A, B, noise, paraAnsatzs, layerAnsatzs)

        if tmploss < best_loss:
            best_loss = tmploss
            best_index = i
    
    return theta_list[best_index]


def multitrain(diff_step, qubitNum, theta, loss, learn_rate, A, B, noise, paraAnsatzs, layerAnsatzs = 1, usePSR = True):
    length = len(theta)

    gradient = []
    if (usePSR):

        for i in range(length):
            right_theta = theta.copy()
            left_theta = theta.copy()
            right_theta[i] = right_theta[i] + np.pi / 2
            left_theta[i] = left_theta[i] - np.pi / 2
            right_loss = getLoss(qubitNum, right_theta, A, B, noise, paraAnsatzs, layerAnsatzs)
            left_loss = getLoss(qubitNum, left_theta, A, B, noise, paraAnsatzs, layerAnsatzs)
            tmp_gradient = (right_loss - left_loss) / 2
            gradient.append(tmp_gradient)
    else:
        for i in range(length):

            tmp_theta = theta.copy()
            tmp_theta[i] = tmp_theta[i] + diff_step * 4.0 * np.pi
            tmp_loss = 0
            tmp_loss = getLoss(qubitNum, tmp_theta, A, B, noise, paraAnsatzs, layerAnsatzs)
            tmp_gradient = (tmp_loss - loss) / (diff_step*4.0*np.pi)
            gradient.append(tmp_gradient)

    theta_list = []
    best_loss = float('inf')  # Initialize to positive infinity
    best_index = 0
    for i in range(len(learn_rate)):
        tmptheta = theta.copy() - learn_rate[i] *  np.array(gradient)
        theta_list.append(tmptheta)
        tmploss = getLoss(qubitNum, tmptheta, A, B, noise, paraAnsatzs, layerAnsatzs)

        if tmploss < best_loss:
            best_loss = tmploss
            best_index = i

    return [learn_rate[best_index], theta_list[best_index]]

def compute_asymmetric_log_grid(rate, N, log_range, asym_ratio=0.3, include_rate=True):
    """
    Construct a logarithmic grid (of length N) centered around the rate, with an asymmetric bias toward the decreasing direction on the log10 scale.

    Parameters:
    - rate: Center point (positive number)
    - N: Number of grid points
    - log_range: Logarithmic range, i.e., ±log_range (base-10)
    - asym_ratio: Asymmetry ratio (0 = completely symmetric, >0 = biased toward smaller values; recommended 0.3~0.7)

    Returns:
    - result: A NumPy array of length N
    """
    assert rate > 0 and N >= 2
    log_center = np.log10(rate)
    
    # Center point offset: the larger asym_ratio, the more biased toward decreasing values
    # For example, asym_ratio = 0.3 indicates a 30% range offset
    shift = log_range * asym_ratio
    start = log_center - log_range + shift
    end = log_center + log_range + shift
    log_vals = np.linspace(start, end, N)

    result = 10 ** log_vals

    if include_rate:
        result = np.append(result, rate)

    return result

def clip_rate(rate, min_rate=1e-20, max_rate=1e2):
    """
    Constrain the rate within the [min_rate, max_rate] interval to prevent explosion or excessively small values
    """
    return np.clip(rate, min_rate, max_rate)

def compute_downward_log_grid(rate, N, log_range):
    """
    Construct a logarithmic grid (of length N) with rate as the maximum value, sampled downward on the log10 scale.

    Parameters:
    - rate: Maximum value (positive number)
    - N: Number of grid points
    - log_range: Logarithmic range, i.e., from log10(rate) - log_range to log10(rate)

    Returns:
    - result: A NumPy array of length N, sorted in increasing order (from smallest to rate)
    """
    assert rate > 0 and N >= 2
    log_max = np.log10(rate)
    log_min = log_max - log_range
    log_vals = np.linspace(log_min, log_max, N)  # From smallest to largest, approaching the rate
    result = 10 ** log_vals
    return result


def GeneralAnsatzGES(A, B, layerAnsatzs, isPrintScreen = True, DifferentStep = 1e-6, LearnRateVector = [1], \
                     LearnRateTrainSize = 5, LearnRateTrainOffset = 1.0, MaxIterStep = 0, LossMonitor = 1e-7, isOutputLoss = True,\
                     initialAnsatzsNum = 1000, useNoise = False, SingleGateNoise = 1e-6, DoubleGateNoise = 1e-8, usePSR = True, isPrintcfg = False):
    
    """
    Adopt the ansatz shown in Figure 11

    Parameters:
    - A, B: Matrix pair (A, B) for generalized eigenvalue problem
    - layerAnsatzs: Number of ansatz layers
    - isPrintScreen: Whether to output to screen
    - DifferentStep: Finite difference step size
    - LearnRateVector: Initial learning rate parameter list
    - LearnRateTrainSize: Number of learning rate parameters during training
    - LearnRateTrainOffset: Downward offset rate for learning rate during training
    - MaxIterStep: Maximum iteration steps (0 = no upper limit)
    - LossMonitor: Exit when loss function falls below this value
    - isOutputLoss: Whether to output loss function file
    - initialAnsatzsNum: Number of initial ansatz attempts
    - useNoise: Whether to use noise simulation
    - SingleGateNoise: Single-qubit gate noise strength
    - DoubleGateNoise: Two-qubit gate noise strength
    - usePSR: Whether to use parameter shift rule
    - isPrintcfg: Whether to print input parameter configuration

    Returns:
    - result: List of generalized eigenvalues
    """



    if (isPrintcfg):
        print("Function inputs:", locals())

    eigvals, eigvecs = eig(A, B)

    # seed = 1 # Set your desired seed

    # Get dimensions (returns a tuple, e.g., (2, 3))
    shape = A.shape
    rows, cols = shape

    qubitNum = math.ceil(math.log2(rows))  # Number of random floats to generate
    n = 2**qubitNum

    noise = NoiseModel()
    if (useNoise):
        noise.add_all_qubit_quantum_error(amplitude_damping_error(SingleGateNoise),GateType.RY)
        noise.add_all_qubit_quantum_error(amplitude_damping_error(SingleGateNoise),GateType.RX)
        noise.add_all_qubit_quantum_error(amplitude_damping_error(SingleGateNoise),GateType.RZ)
        noise.add_all_qubit_quantum_error(depolarizing_error(DoubleGateNoise),GateType.CNOT)

    paraAnsatzs = 3*qubitNum

    theta = getBestTheta(initialAnsatzsNum, qubitNum, A, B, noise, paraAnsatzs, layerAnsatzs)
    iter = 0
    loss = getLoss(qubitNum, theta, A, B, noise, paraAnsatzs, layerAnsatzs)
    if (isPrintScreen):
        print("iter = ", iter)
        print(loss)

    iterlist = []
    iterlist.append(iter)

    losslist = []
    while (MaxIterStep == 0 or iter < MaxIterStep):
        iter +=1
        tmp_loss = loss
        [bestrate,theta] = multitrain(DifferentStep, qubitNum, theta, loss, LearnRateVector, A, B, noise, paraAnsatzs, layerAnsatzs, usePSR)
        bestrate = clip_rate(bestrate)
        LearnRateVector = compute_asymmetric_log_grid(bestrate, LearnRateTrainSize, LearnRateTrainOffset)
        loss = getLoss(qubitNum, theta, A, B, noise, paraAnsatzs, layerAnsatzs)
        if (isPrintScreen):
            print("iter = ",iter)
            print("bestrate = ",bestrate)
            print(loss)

        iterlist.append(iter)
        losslist.append(loss)
        if (bestrate < 1e-10):
            break
        if (loss < LossMonitor):
            break
        if (loss == tmp_loss):
            break

    W_AnsatzsCircuit = QCircuit()
    V_AnsatzsCircuit = QCircuit()
    for lay in range(layerAnsatzs):
        start_index = lay * 2 * paraAnsatzs
        end_index = (lay + 1) * 2 * paraAnsatzs
        sub_arr = theta[start_index:end_index]
        tmpW, tmpV = create_WV_Circuit(qubitNum, sub_arr)
        W_AnsatzsCircuit << tmpW
        V_AnsatzsCircuit << tmpV
    machine = CPUQVM()
    x_Vec, y_Vec = runWVCircuit(machine, W_AnsatzsCircuit, V_AnsatzsCircuit, qubitNum, noise)


    W = np.array(x_Vec).T
    V = np.array(y_Vec).T
    T = np.dot(np.dot(W.conj().T,A),V)
    S = np.dot(np.dot(W.conj().T,B),V)
    LAMBDA_vec = []
    for i in range(2**qubitNum):   
        tii = T[i,i]
        sii = S[i,i]
        lambda_ii = tii/sii
        LAMBDA_vec.append(lambda_ii)


    # Use sorted + lambda to sort by real part (descending order)
    sorted_LAMBDA_vec = sorted(LAMBDA_vec, key=lambda x: x.real, reverse=True)
    sorted_eigvals = sorted(eigvals, key=lambda x: x.real, reverse=True)

    if (isPrintScreen):
        print("Experimental generalized eigenvalues are:")
        print(sorted_LAMBDA_vec)
        print("Theoretical generalized eigenvalues are:")
        print(sorted_eigvals)

    if(isOutputLoss):
        with open('loss.txt', 'w') as f:
            for loss in losslist:
                f.write(f"{loss:.6e}\n")

    return LAMBDA_vec

def normalize_matrices(A, B, min_threshold_ratio=1e-12):
    """
    Normalize the numerical range of matrices A and B to ensure the maximum value magnitude is not too large while preserving the minimum values.
    Returns normalized A, B, and the scaling factor.
    """
    all_data = np.concatenate([A.flatten(), B.flatten()])
    abs_data = np.abs(all_data)
    
    max_val = np.max(abs_data)
    min_val = np.min(abs_data[abs_data > 0])  # Avoid division by zero

    if min_val == 0 or max_val == 0:
        scale = 1.0  # Avoid all-zero exceptional cases
    else:
        # Ratio and order of magnitude range
        ratio = max_val / min_val
        log_range = np.log10(ratio)

        # Select a scaling factor aiming to compress the maximum value to the range ~1e0~1e3
        scale = max_val if max_val > 1e3 else 1.0

    A_scaled = A / scale
    B_scaled = B / scale
    return A_scaled, B_scaled

def expand_to_power_of_two(A):
    """
    If the dimension of matrix A is not an integer power of 2, expand it to the smallest 2^N size that is larger than its current dimension.
    The expanded parts are filled with zeros, except the diagonal elements which are set to 1 (identity matrix form).
    """
    M, N = A.shape
    assert M == N, "A must be a square matrix"

    # Check if it is a power of 2
    def is_power_of_two(x):
        return (x & (x - 1) == 0) and x != 0

    if is_power_of_two(N):
        return A, N  # No expansion needed

    # Take the ceiling of log2 and expand to the new dimension
    new_size = 2 ** int(np.ceil(np.log2(N)))
    A_expanded = np.eye(new_size, dtype=A.dtype)
    A_expanded[:N, :N] = A
    return A_expanded, new_size



def VQGESpreprocess(A, B):
    A_scale, B_scale = normalize_matrices(A, B)
    AA, N = expand_to_power_of_two(A_scale)
    BB, N = expand_to_power_of_two(B_scale)
    return AA, BB, N

def remove_closest_to_one(array, N):
    array = np.asarray(array).flatten()  # Ensure it is a 1-dimensional array
    distances = np.abs(array - (1 + 0j))
    indices_to_remove = np.argsort(distances)[:N]
    mask = np.ones(len(array), dtype=bool)
    mask[indices_to_remove] = False
    return array[mask]

def VQGESpostprocess(eigvals, oldN, newN):
    eigvals = np.asarray(eigvals).flatten()
    if (newN == oldN):
        return eigvals
    else:
        N = newN-oldN
        return remove_closest_to_one(eigvals, N)
    


def optimizeAnsatzGES(A, B, W, V, EigVals,\
isOutputOriginIR = True,\
isOutputAnsatzCircuit = True,\
isOutputLoss = True,\
isPrintScreen = True,\
InitialAnsatzOffsetRate = 0.1,\
LearnRateVector = np.array([1.0]),\
DifferentStep = 1e-6,\
LearnRateTrainSize = 5,\
LearnRateTrainOffset = 1.0,\
LossMonitor = 1e-8,\
MaxIterStep = 0, isPrintcfg = False):
    """
    Adopt quantum circuit architecture search ansatz

    Parameters:
    - A, B: Matrix pair (A, B) for generalized eigenvalue problem
    - W, V: Matrix pair (W, V) for generalized eigenvalue problem
    - isOutputOriginIR: Whether to output OriginIR file
    - isOutputAnsatzCircuit: Whether to output circuit file
    - isOutputLoss: Whether to output loss function file
    - isPrintScreen: Whether to output to screen
    - InitialAnsatzOffsetRate: Initial ansatz offset rate (difference from theoretical optimal circuit)
    - LearnRateVector: Initial learning rate parameter list
    - DifferentStep: Finite difference step size
    - LearnRateTrainSize: Number of learning rate parameters during training
    - LearnRateTrainOffset: Downward offset rate for learning rate during training
    - LossMonitor: Exit when loss function falls below this value
    - MaxIterStep: Maximum iteration steps (0 = no upper limit)
    - isPrintcfg: Whether to print input parameter configuration

    Returns:
    - result: List of generalized eigenvalues
    """
    
    if (isPrintcfg):
        print("Function inputs:", locals())
    result = QGeneralizedEigenvalue(A, B, W, V, EigVals, isOutputOriginIR, isOutputAnsatzCircuit, isOutputLoss, isPrintScreen, InitialAnsatzOffsetRate, LearnRateVector, DifferentStep, LearnRateTrainSize, LearnRateTrainOffset, LossMonitor, MaxIterStep)
    
    return result
    


def VQGES(A, B, layerAnsatzs = 6, isPrintScreen = True, isOutputOriginIR = True, isOutputAnsatzCircuit = True, InitialAnsatzOffsetRate = 0.1, DifferentStep = 1e-7, LearnRateVector = [1], \
                     LearnRateTrainSize = 5, LearnRateTrainOffset = 1.0, MaxIterStep = 0, LossMonitor = 1e-7, isOutputLoss = True,\
                     initialAnsatzsNum = 1000, useNoise = False, SingleGateNoise = 1e-6, DoubleGateNoise =1e-8, usePSR = True, isPrintcfg = False, isUseOptimizeMethod = True):
    """
    Generalized Eigenvalue Solver Master Interface

    Parameters:
    - A, B: Matrix pair (A, B) for generalized eigenvalue problem
    - layerAnsatzs: Number of ansatz layers
    - isOutputOriginIR: Whether to output OriginIR file
    - isOutputAnsatzCircuit: Whether to output circuit file
    - isOutputLoss: Whether to output loss function file
    - isPrintScreen: Whether to output to screen
    - InitialAnsatzOffsetRate: Initial ansatz offset rate (difference from theoretical optimal circuit)
    - LearnRateVector: Initial learning rate parameter list
    - DifferentStep: Finite difference step size
    - LearnRateTrainSize: Number of learning rate parameters during training
    - LearnRateTrainOffset: Downward offset rate for learning rate during training
    - LossMonitor: Exit when loss function falls below this value
    - MaxIterStep: Maximum iteration steps (0 = no upper limit)
    - initialAnsatzsNum: Number of initial ansatz attempts
    - useNoise: Whether to use noise simulation
    - SingleGateNoise: Single-qubit gate noise strength
    - DoubleGateNoise: Two-qubit gate noise strength
    - usePSR: Whether to use parameter shift rule
    - isPrintcfg: Whether to print input parameter configuration
    - isUseOptimizeMethod: Whether to enable optimization scheme

    Returns:
    - result: List of generalized eigenvalues
    
    """    

    AA, BB, N = VQGESpreprocess(A, B)
    r,c = A.shape
    oldN = r
    rr, cc = AA.shape
    if (math.log2(rr) > 2 and isUseOptimizeMethod):
        S, T, Q, Z = la.qz(AA, BB, output='complex')
        eigvals, eigvecs = la.eig(AA, BB)

        result = optimizeAnsatzGES(AA, BB, W = Q, V = Z, EigVals = eigvals,\
        isOutputOriginIR = isOutputOriginIR,\
        isOutputAnsatzCircuit = isOutputAnsatzCircuit,\
        isOutputLoss = isOutputLoss,\
        isPrintScreen = isPrintScreen,\
        InitialAnsatzOffsetRate = InitialAnsatzOffsetRate,\
        LearnRateVector = LearnRateVector,\
        DifferentStep = DifferentStep,\
        LearnRateTrainSize = LearnRateTrainSize,\
        LearnRateTrainOffset = LearnRateTrainOffset,\
        LossMonitor = LossMonitor,\
        MaxIterStep = MaxIterStep, isPrintcfg = isPrintcfg)
        return VQGESpostprocess(result,oldN, N)
    else:
        result = GeneralAnsatzGES(AA, BB, layerAnsatzs = layerAnsatzs, isPrintScreen = isPrintScreen, DifferentStep = DifferentStep, LearnRateVector = LearnRateVector,\
                     LearnRateTrainSize = LearnRateTrainSize, LearnRateTrainOffset = LearnRateTrainOffset, MaxIterStep = MaxIterStep, LossMonitor = LossMonitor, isOutputLoss = isOutputLoss,\
                     initialAnsatzsNum = initialAnsatzsNum, useNoise = useNoise, SingleGateNoise = SingleGateNoise, DoubleGateNoise = DoubleGateNoise, usePSR = usePSR, isPrintcfg = isPrintcfg)
        return VQGESpostprocess(result,oldN, N)
        


        


