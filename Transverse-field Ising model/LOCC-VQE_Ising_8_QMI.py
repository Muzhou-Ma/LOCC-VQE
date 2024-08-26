import tensorcircuit as tc
import tensorflow as tf
import tensornetwork as tn
import os
import numpy as np
import time
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from qiskit.quantum_info import SparsePauliOp


tc.set_backend("tensorflow")
tc.set_dtype("complex64")

def ZZ_measurement(theta_1, theta_2):
    ZZ_circuit = tc.Circuit(3)
    
    ZZ_circuit.rx(0, theta = theta_1[0])
    ZZ_circuit.ry(0, theta = theta_1[1])
    ZZ_circuit.rz(0, theta = theta_1[2])
    ZZ_circuit.rx(2, theta = theta_1[0])
    ZZ_circuit.ry(2, theta = theta_1[1])
    ZZ_circuit.rz(2, theta = theta_1[2])
    ZZ_circuit.rxx(0, 2, theta = theta_1[3])
    ZZ_circuit.ryy(0, 2, theta = theta_1[4])
    ZZ_circuit.rzz(0, 2, theta = theta_1[5])
    ZZ_circuit.rx(0, theta = theta_1[0])
    ZZ_circuit.ry(0, theta = theta_1[1])
    ZZ_circuit.rz(0, theta = theta_1[2])
    ZZ_circuit.rx(2, theta = theta_1[0])
    ZZ_circuit.ry(2, theta = theta_1[1])
    ZZ_circuit.rz(2, theta = theta_1[2])
    
    ZZ_circuit.rx(1, theta = theta_2[0])
    ZZ_circuit.ry(1, theta = theta_2[1])
    ZZ_circuit.rz(1, theta = theta_2[2])
    ZZ_circuit.rx(2, theta = theta_2[0])
    ZZ_circuit.ry(2, theta = theta_2[1])
    ZZ_circuit.rz(2, theta = theta_2[2])
    ZZ_circuit.rxx(1, 2, theta = theta_2[3])
    ZZ_circuit.ryy(1, 2, theta = theta_2[4])
    ZZ_circuit.rzz(1, 2, theta = theta_2[5])
    ZZ_circuit.rx(2, theta = theta_2[0])
    ZZ_circuit.ry(2, theta = theta_2[1])
    ZZ_circuit.rz(2, theta = theta_2[2])
    ZZ_circuit.rx(2, theta = theta_2[0])
    ZZ_circuit.ry(2, theta = theta_2[1])
    ZZ_circuit.rz(2, theta = theta_2[2])
    
    return ZZ_circuit

def syndrome_circuit(n, params_syn): #paramas_syn shape [n-1][2][9]
    circuit = tc.Circuit(2*n)
    for i in range(n):
        circuit.h(i)
    for i in range(n-1):
        circuit.append(ZZ_measurement(params_syn[i][0],params_syn[i][1]),[i,i+1,n+i+1])    
    return circuit

def correction_circuit_qsim(n, params_corr_1, params_corr_2, params_corr_3):
    corr_circuit = tc.Circuit(2*n)
    m = int(np.log2(n))
    for i in range(m):
        count = 0
        for j in range(1,n,2**(i+1)):
            for k in range(2**i):
                corr_circuit.ry(n+j,theta=np.pi/2)
                corr_circuit.rxx(n+j,j+k+2**i-1,theta=params_corr_1[i][count][0])
                corr_circuit.rx(j+k+2**i-1,theta=params_corr_1[i][count][1])
                corr_circuit.ry(n+j,theta=-np.pi/2)
            if(j+2**i<n):
                corr_circuit.cnot(n+j+2**i,n+j)
        count = count + 1
    
    for i in range(n):
        for j in range(n):
            corr_circuit.ry(n+i,theta=np.pi/2)
            corr_circuit.rxx(n+i,j,theta=params_corr_2[i][j][0])
            corr_circuit.rx(j,theta=params_corr_2[i][j][1]) # previous mistakenly writen i instead of j here!!!
            corr_circuit.ry(n+i,theta=-np.pi/2)   
    
    for i in range(n):
        corr_circuit.rx(i,theta=params_corr_3[i][0])
        corr_circuit.ry(i,theta=params_corr_3[i][1])
        corr_circuit.rz(i,theta=params_corr_3[i][2])
    return corr_circuit


def Hamiltonian(c: tc.Circuit, n: int, h_1: float = 0, h_2: float = 0):
    e = 0.0
    for i in range(0,n-1):
        e += -1/2 * tf.cast(c.expectation_ps(x=[i, i+1]), tf.float64)    
    for i in range(n):
        e += -h_2/2 * tf.cast(c.expectation_ps(z=[i]), tf.float64)
        
    return tc.backend.real(e)

def QMI(c: tc.Circuit, n: int):
    # correlation = tf.cast(tf.constant([0.0]), tf.float64)
    s = c.state()
    SL = tc.quantum.entanglement_entropy(s, cut=list(np.arange(1,2*n)))
    SR = tc.quantum.entanglement_entropy(s, cut=list(np.arange(0,n-1))+list(np.arange(n,2*n)))
    SRL = tc.quantum.entanglement_entropy(s, cut=list(np.arange(1,n-1))+list(np.arange(n,2*n)))
    qmi = SL+SR-SRL
    return qmi


def vqe(param, n, h_1, h_2):
    circuit = tc.Circuit(2*n)
    paramc_0 = tc.backend.cast(
        tf.reshape(param[0:(n-1)*2*6],((n-1),2,6)), tc.dtypestr
    )
    paramc_1 = tc.backend.cast(
        tf.reshape(param[(n-1)*2*6:(n-1)*2*6+int(np.log2(n))*(n//2)*2],(int(np.log2(n)),(n//2),2)), tc.dtypestr
    )
    paramc_2 = tc.backend.cast(
        tf.reshape(param[(n-1)*2*6+int(np.log2(n))*(n//2)*2:(n-1)*2*6+int(np.log2(n))*(n//2)*2+n*n*2],(n,n,2)), tc.dtypestr
    )
    paramc_3 = tc.backend.cast(
        tf.reshape(param[(n-1)*2*6+n*n*2+int(np.log2(n))*(n//2)*2:(n-1)*2*6+n*n*2+int(np.log2(n))*(n//2)*2+n*3],(n,3)), tc.dtypestr
    )
    circuit.append(syndrome_circuit(n,paramc_0))
    circuit.append(correction_circuit_qsim(n,paramc_1,paramc_2,paramc_3))
    # pdb.set_trace()
    QMI_value = QMI(circuit, n)
    energy = Hamiltonian(circuit, n, h_1, h_2)
    return energy, QMI_value

vqe_vmap = tc.backend.jit(
    tc.backend.vmap(vqe, vectorized_argnums = (0,)), static_argnums=(1,2,3,)
)

def batched_train_step_tf(batch, n, h_1, h_2, rand_seed, maxiter=1000, random_idx=None, load_param=False, input_param=None):
    param = tf.convert_to_tensor(input_param, dtype=getattr(tf, tc.rdtypestr))
    
    energy, QMI_value = vqe_vmap(param, n, h_1, h_2)
    # pdb.set_trace()
    # correlation_value = vqe(param[0], n, h_1, h_2)
    return energy, QMI_value

def GS_QMI(n,h_1,h_2):
    list_1 = []
    list_2 = []

    pauli_list = []
    num = n
    
    for i in range(0,n-1):
        t1 = 'I'*(i) + 'X'*2 + 'I'*(n-i-2)
        list_1.append((t1, -1/2))
    
    for i in range(n):
        t2 = 'I'*(i)+'Z'+(n-i)*'I'
        list_2.append((t2, -h_2/2))
    
    pauli_list = list_1 + list_2
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(H_m)
    GS_state = eigenvectors[:,np.argmin(eigenvalues)] # we need column vector
    state = tc.quantum.QuOperator.from_tensor(GS_state)
    
    SL = tc.quantum.entanglement_entropy(state, cut=list(np.arange(1,n)))
    SR = tc.quantum.entanglement_entropy(state, cut=list(np.arange(0,n-1)))
    SRL = tc.quantum.entanglement_entropy(state, cut=list(np.arange(1,n-1)))
    qmi = SL + SR - SRL
    # pdb.set_trace()
    return qmi



if __name__=="__main__":
    n = 4
    batch = 48
    rand_seed = 2131558
    np.random.seed(rand_seed)
    random_seed_array = np.random.randint(1,1000,(150,4))
    optim_results = []
    GS_Energy_list = []
    gap_list = []
    h_1_range = 0.0
    h_2_range = np.arange(0,3.0,0.1)
    max_iter = 2000
    
    GS_QMI_list = []
    QMI_list = []
    energy_list = []
    min_energy_idx_list = []
    min_error_list = []
    
    h_1 = 0.0
    for i in range(30):
        h_2 = 0.1*i
        qmi_GS = GS_QMI(n,0.0,h_2)
        GS_QMI_list.append(qmi_GS)

        param_path = f'./LOCC-VQE_Ising_8_results/LOCC-VQE_Ising_8_seed=2131558_iter=2000/params/h_1=0.0_h_2={h_2:.1f}.npy'
        input_param = np.load(param_path)
        print('----------------------------------------------------------------------------')
        print(f'load parameter from {param_path}') 
        print(qmi_GS)
        
        energy, qmi = batched_train_step_tf(batch, n, h_1, h_2, random_seed_array, maxiter=1,random_idx=i,load_param=True, input_param=input_param)
        print(qmi)
        energy_list.append(energy)
        QMI_list.append(qmi)

    pdb.set_trace()
    np.save('./LOCC-VQE_Ising_8_results/qmi_GS.npy',np.array(GS_QMI_list))
    np.save('./LOCC-VQE_Ising_8_results/QMI_list.npy',np.array(QMI_list))
    np.save('./LOCC-VQE_Ising_8_results/energy_list.npy',np.array(energy_list))