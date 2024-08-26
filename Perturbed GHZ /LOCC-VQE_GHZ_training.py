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
    return energy

vqe_vvag = tc.backend.jit(
    tc.backend.vectorized_value_and_grad(vqe, vectorized_argnums = (0,)), static_argnums=(1,2,3)
)

def batched_train_step_tf(batch, n, kx, h, rand_seed, maxiter=1000, random_idx=None, load_param=False, input_param=None):
    if load_param:
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(input_param, dtype=getattr(tf, tc.rdtypestr))
        )
    else:
        param = tf.Variable(
            initial_value=tf.concat(
                [tf.random.normal(shape=[int(batch/4), (n-1)*2*9+int(np.log2(n))*(n//2)*3+n*n*3+n*3], mean=0, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][0]),
                tf.random.normal(shape=[int(batch/4), (n-1)*2*9+int(np.log2(n))*(n//2)*3+n*n*3+n*3], mean=np.pi/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][1]),
                tf.random.normal(shape=[int(batch/4), (n-1)*2*9+int(np.log2(n))*(n//2)*3+n*n*3+n*3], mean=np.pi/2, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][2]),
                tf.random.normal(shape=[int(batch/4), (n-1)*2*9+int(np.log2(n))*(n//2)*3+n*n*3+n*3], mean=np.pi*3/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][3])
                ],0)
        )
    opt = tf.keras.optimizers.legacy.Adam(1e-2)
    energy_lowest = []
    for i in range(maxiter):
        energy, grad = vqe_vvag(param, n, kx, h)
        opt.apply_gradients([(grad, param)])
        energy_lowest.append(np.min(energy))
        if i % 100 == 0:
            start = time.time()
            print(f'iter_{i}:')
            print(energy)
            print(f'lowest:{np.min(energy)}')
            end = time.time()
            print(f'time: {end-start}')
    return energy_lowest, param


def GS_energy(n,h,kx):
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
    # pauli_list = list_3 + list_4
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    eigenvalues, eigenvectors = np.linalg.eig(H_m)
    sorted_eigenvalues = sorted(eigenvalues)
    return np.real(sorted_eigenvalues[0])


if __name__=="__main__":
    n = 8
    batch = 48
    kx = 16
    rand_seed = 2131558
    np.random.seed(rand_seed)
    random_seed_array = np.random.randint(1,1000,(11,4))
    
    optim_results = []
    GS_Energy_list = []
    gap_list = []
    h_range = np.arange(0,1.1,0.1)
    max_iter = 2000
    
    for i in range(11):
        h = 0.1*i
        GS_energy_h = GS_energy(n,h,kx)
        GS_Energy_list.append(GS_energy_h)
        print(f'Ground state energy for h={h} is : {GS_Energy_list[i]}')
        energy_lowest, params = batched_train_step_tf(batch, n, kx, h, random_seed_array, max_iter,random_idx=i, load_param=False)
        np.save(f'./results/GHZ/GHZ_trail_8_seed={int(rand_seed)}_iter={int(max_iter)}/params_Z/Z_h={h}.npy', params)
        
        
        optim_result = np.min(np.array(energy_lowest))
        optim_results.append(optim_result)
        gap_list.append(optim_result-GS_energy_h)
        np.save(f'./results/GHZ/GHZ_trail_8_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.npy',energy_lowest)
        
        plt.figure()
        plt.plot(energy_lowest)
        plt.xlabel('iter')
        plt.ylabel('energy')
        plt.title(f'GHZ_8_Z_h_{h}_GS_{GS_energy_h}')
        plt.savefig(f'./results/GHZ/GHZ_trail_8_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.jpg')
        np.save(f'./results/GHZ/GHZ_trail_8_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/GHZ_4_Z_h_{h}_optimization_process.npy',energy_lowest)
        
        print(f'Ground state energy: {GS_energy_h}')
        print(f'Best optimum result: {optim_result}')
        print(f'gap = {optim_result-GS_energy_h}')
        
    np.save(f'./results/GHZ/GHZ_trail_8_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.npy',np.array(optim_results))
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(h_range, GS_Energy_list, label='GS_energy')
    plt.plot(h_range, optim_results, label='optim_result')
    plt.xlabel('h_Z')
    plt.ylabel('energy')
    plt.title('optim_energy')
    plt.subplot(1,2,2)
    plt.plot(h_range, np.array(optim_results)-np.array(GS_Energy_list))
    plt.xlabel('h_Z')
    plt.ylabel('energy')
    plt.title('error')
    plt.tight_layout()
    plt.savefig(f'./results/GHZ/GHZ_trail_8_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_final.jpg')

    