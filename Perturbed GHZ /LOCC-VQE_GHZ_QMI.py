import tensorcircuit as tc
import tensorflow as tf
import tensornetwork as tn
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp


tc.set_backend("tensorflow")
tc.set_dtype("complex128")

def ZZ_measurement(theta_1, theta_2):
    ZZ_circuit = tc.Circuit(3)
    ZZ_circuit.ry(0,theta=np.pi/2)
    ZZ_circuit.rxx(0,2,theta=theta_1[0])
    ZZ_circuit.rx(0,theta=theta_1[1])
    ZZ_circuit.rx(2,theta=theta_1[2])
    ZZ_circuit.ryy(0,2,theta = theta_1[3])
    ZZ_circuit.ry(0,theta=theta_1[4])
    ZZ_circuit.ry(2,theta=theta_1[5])
    ZZ_circuit.rzz(0,2,theta=theta_1[6])
    ZZ_circuit.rz(0,theta=theta_1[7])
    ZZ_circuit.rz(2,theta=theta_1[8])
    ZZ_circuit.ry(0,theta=-np.pi/2)
    ZZ_circuit.barrier_instruction()
    
    ZZ_circuit.ry(1,theta=np.pi/2)
    ZZ_circuit.rxx(1,2,theta=theta_2[0])
    ZZ_circuit.rx(1,theta=theta_2[1])
    ZZ_circuit.rx(2,theta=theta_2[2])
    ZZ_circuit.ryy(1,2,theta = theta_2[3])
    ZZ_circuit.ry(1,theta=theta_2[4])
    ZZ_circuit.ry(2,theta=theta_2[5])
    ZZ_circuit.rzz(1,2,theta=theta_2[6])
    ZZ_circuit.rz(1,theta=theta_2[7])
    ZZ_circuit.rz(2,theta=theta_2[8])
    ZZ_circuit.ry(1,theta=-np.pi/2)
    ZZ_circuit.barrier_instruction()
    return ZZ_circuit

def syndrome_circuit(n, params_syn): #paramas_syn shape [n-1][2][9]
    circuit = tc.Circuit(2*n)
    for i in range(n):
        circuit.h(i)
    
    for i in range(n-1):
        circuit.append(ZZ_measurement(params_syn[i][0],params_syn[i][1]),[i,i+1,n+i+1])
    return circuit

def correction_circuit_qsim(n, params_corr_1, params_corr_2, params_corr_3):
    # params_corr_1 is the first part of our current correction circuit, in correspondence with the original protocol [log2(n)][n//2][3]
    # params_corr_2 is the second part of our current correction circuit, having n params vector with n params each [n][n][3]
    # params_corr_3 is the third part of our current correction circuit, adding an unitary gate to each qubit [n][3]
    corr_circuit = tc.Circuit(2*n)
    m = int(np.log2(n))
    for i in range(m):
        count = 0
        for j in range(1,n,2**(i+1)):
            for k in range(2**i):
                corr_circuit.ry(n+j,theta=np.pi/2)
                corr_circuit.rxx(n+j,j+k+2**i-1,theta=params_corr_1[i][count][0])
                corr_circuit.rx(n+j,theta=params_corr_1[i][count][1])
                corr_circuit.rx(j+k+2**i-1,theta=params_corr_1[i][count][2])
                corr_circuit.ry(n+j,theta=-np.pi/2)
            if(j+2**i<n):
                corr_circuit.cnot(n+j+2**i,n+j)
        count = count + 1
        corr_circuit.barrier_instruction()


    
    for i in range(n):
        for j in range(n):          
            corr_circuit.ry(n+i,theta=np.pi/2)
            corr_circuit.rxx(n+i,j,theta=params_corr_2[i][j][0])
            corr_circuit.rx(j,theta=params_corr_2[i][j][2]) # previous mistakenly writen i instead of j here!!!
            corr_circuit.ry(n+i,theta=-np.pi/2)
            
    for i in range(n):
        
        
        corr_circuit.rx(i,theta=params_corr_3[i][0])
        corr_circuit.ry(i,theta=params_corr_3[i][1])
        corr_circuit.rz(i,theta=params_corr_3[i][2])
    return corr_circuit



def Hamiltonian(c: tc.Circuit, n: int, kx: float, h: float = 0):
    e = 0.0
    for i in range(0,n-1,2):
        e += (1-h)/n * tf.cast(c.expectation_ps(z=[i, i+1]), tf.float64)
    for i in range(1,n-2,2):
        e += (1-h)/n * tf.cast(c.expectation_ps(z=[i, i+1]), tf.float64)
    e += (kx-h)/n * tf.cast(c.expectation_ps(x=list(range(n))), tf.float64)
    
    for i in range(n):
        e += h/n * tf.cast(c.expectation_ps(z=[i]), tf.float64)
        
    return -tc.backend.real(e)

def QMI(c: tc.Circuit, n: int):
    s = c.state()
    SL = tc.quantum.entanglement_entropy(s, cut=list(np.arange(1,2*n)))
    SR = tc.quantum.entanglement_entropy(s, cut=list(np.arange(0,n-1))+list(np.arange(n,2*n)))
    SRL = tc.quantum.entanglement_entropy(s, cut=list(np.arange(1,n-1))+list(np.arange(n,2*n)))
    qmi = SL+SR-SRL
    return qmi

def vqe(param, n, kx, h):
    # param[0]: [n-1][2][9]
    # param[1]: [log2(n)][n//2][3]
    # param[2]: [n][n][3]
    # param[3]: [n][3]
    
    circuit = tc.Circuit(2*n)
    paramc_0 = tc.backend.cast(
        tf.reshape(param[0:(n-1)*2*9],((n-1),2,9)), tc.dtypestr
    )
    
    paramc_1 = tc.backend.cast(
        tf.reshape(param[(n-1)*2*9:(n-1)*2*9+int(np.log2(n))*(n//2)*3],(int(np.log2(n)),n//2,3)), tc.dtypestr
    )

    paramc_2 = tc.backend.cast(
        tf.reshape(param[(n-1)*2*9+int(np.log2(n))*(n//2)*3:(n-1)*2*9+int(np.log2(n))*(n//2)*3+n*n*3],(n,n,3)), tc.dtypestr
    )
    paramc_3 = tc.backend.cast(
        tf.reshape(param[(n-1)*2*9+int(np.log2(n))*(n//2)*3+n*n*3:(n-1)*2*9+int(np.log2(n))*(n//2)*3+n*n*3+n*3],(n,3)), tc.dtypestr
    )
    circuit.append(syndrome_circuit(n,paramc_0))
    circuit.append(correction_circuit_qsim(n,paramc_1,paramc_2,paramc_3))
    energy = Hamiltonian(circuit, n, kx, h)
    qmi = QMI(circuit, n)
    return energy, qmi

vqe_vmap = tc.backend.jit(
    tc.backend.vmap(vqe, vectorized_argnums = (0,)), static_argnums=(1,2,3)
)

def batched_train_step_tf(batch, n, kx, h, rand_seed, maxiter=1000, random_idx=None, load_param=False, input_param=None):
    param = tf.Variable(initial_value=tf.convert_to_tensor(input_param, dtype=getattr(tf, tc.rdtypestr)))
    
    energy, QMI_value = vqe_vmap(param, n, kx, h)
    
    return energy, QMI_value


def GS_QMI(n,h,kx):
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    pauli_list = []
    num = n
    
    for i in range(0,n-1,2):
        t1 = 'I'*(i) + 'Z'*2 + 'I'*(n-i-2)
        list_1.append((t1, -(1-h)/num))

    for i in range(0,n-3,2):
        t2 = 'I'*(i+1) + 'Z'*2 + 'I'*(n-i-1)
        list_2.append((t2, -(1-h)/num))

    t3 = 'X'*n
    list_3.append((t3, -(kx-h)/num))

    for i in range(n):
        t4 = 'I'*(i)+'Z'+(n-i)*'I'
        list_4.append((t4, -h/num))
    
    pauli_list = list_1 + list_2 + list_3 + list_4
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    eigenvalues, eigenvectors = np.linalg.eigh(H_m)
    GS_state = eigenvectors[:,np.argmin(eigenvalues)] # we need column vector
    state = tc.quantum.QuOperator.from_tensor(GS_state)
    SL = tc.quantum.entanglement_entropy(state, cut=list(np.arange(1,n)))
    SR = tc.quantum.entanglement_entropy(state, cut=list(np.arange(0,n-1)))
    SRL = tc.quantum.entanglement_entropy(state, cut=list(np.arange(1,n-1)))
    qmi = SL + SR - SRL
    return qmi


if __name__=="__main__":
    n = 8
    batch = 48
    kx = 16
    rand_seed = 2131558
    np.random.seed(rand_seed)
    random_seed_array = np.random.randint(1,1000,(11,4))
    
    optim_results = []
    GS_qmi_list = []
    qmi_list = []
    gap_list = []
    h_range = np.arange(0,1.1,0.1)
    max_iter = 2000
    
    for i in range(11):
        h = 0.1*i
        GS_qmi = GS_QMI(n,h,kx)
        GS_qmi_list.append(GS_qmi)
        print(f'GS_QMI for h={h} is : {GS_qmi_list[i]}')
        param_path = f'./GHZ_trail_8_seed={int(rand_seed)}_iter={int(max_iter)}/params_Z/Z_h={h}.npy'
        input_param = np.load(param_path)
        energy, qmi = batched_train_step_tf(batch, n, kx, h, random_seed_array, max_iter,random_idx=i,load_param=True, input_param=input_param)
        optim_results.append(energy)
        qmi_list.append(qmi)
        
        print(f'qmi corresponding to the best batch is:{qmi[np.argmin(energy)]}')
        print(f'energy:\n{energy}')
        print(f'qmi:\n{qmi}')
        
    pdb.set_trace()
    np.save(f'./GHZ_8_Z_GS_QMI.npy', np.array(GS_qmi_list))
    np.save(f'./GHZ_8_Z_results_QMI.npy', np.array(qmi_list))
    np.save(f'./GHZ_8_Z_results_energy.npy', np.array(optim_results))
   
    