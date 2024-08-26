import tensorcircuit as tc
import tensorflow as tf
import tensornetwork as tn
import numpy as np
import time
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from qiskit.quantum_info import SparsePauliOp


tc.set_backend("tensorflow")
tc.set_dtype("complex64")

def block_circuit(n, theta_1, theta_2): #paramas_syn shape [n-1][2][9]
    circuit = tc.Circuit(n)
    for i in range(n):
        circuit.h(i)
    
    for i in range(int(n/2)):
        circuit.rx(2*i, theta = theta_1[i][0])
        circuit.ry(2*i, theta = theta_1[i][1])
        circuit.rz(2*i, theta = theta_1[i][2])
        circuit.rx(2*i+1, theta = theta_1[i][0])
        circuit.ry(2*i+1, theta = theta_1[i][1])
        circuit.rz(2*i+1, theta = theta_1[i][2])
        circuit.rxx(2*i, 2*i+1, theta = theta_1[i][3])
        circuit.ryy(2*i, 2*i+1, theta = theta_1[i][4])
        circuit.rzz(2*i, 2*i+1, theta = theta_1[i][5])
        circuit.rx(2*i, theta = theta_1[i][0])
        circuit.ry(2*i, theta = theta_1[i][1])
        circuit.rz(2*i, theta = theta_1[i][2])
        circuit.rx(2*i+1, theta = theta_1[i][0])
        circuit.ry(2*i+1, theta = theta_1[i][1])
        circuit.rz(2*i+1, theta = theta_1[i][2])
    
    for i in range(int(n/2-1)):
        circuit.rx(2*i+1, theta = theta_1[int(i+n/2)][0])
        circuit.ry(2*i+1, theta = theta_1[int(i+n/2)][1])
        circuit.rz(2*i+1, theta = theta_1[int(i+n/2)][2])
        circuit.rx(2*i+2, theta = theta_1[int(i+n/2)][0])
        circuit.ry(2*i+2, theta = theta_1[int(i+n/2)][1])
        circuit.rz(2*i+2, theta = theta_1[int(i+n/2)][2])
        circuit.rxx(2*i+1, 2*i+2, theta = theta_1[int(i+n/2)][3])
        circuit.ryy(2*i+1, 2*i+2, theta = theta_1[int(i+n/2)][4])
        circuit.rzz(2*i+1, 2*i+2, theta = theta_1[int(i+n/2)][5])
        circuit.rx(2*i+1, theta = theta_1[int(i+n/2)][0])
        circuit.ry(2*i+1, theta = theta_1[int(i+n/2)][1])
        circuit.rz(2*i+1, theta = theta_1[int(i+n/2)][2])
        circuit.rx(2*i+2, theta = theta_1[int(i+n/2)][0])
        circuit.ry(2*i+2, theta = theta_1[int(i+n/2)][1])
        circuit.rz(2*i+2, theta = theta_1[int(i+n/2)][2])
        
        
    for i in range(n):
        circuit.rx(i, theta = theta_2[i][0])
        circuit.ry(i, theta = theta_2[i][1])
        circuit.rz(i, theta = theta_2[i][2])
    
    return circuit


def Hamiltonian(c: tc.Circuit, n: int, h_1: float = 0, h_2: float = 0):
    e = 0.0
    for i in range(0,n-1):
        e += -1/2 * tf.cast(c.expectation_ps(x=[i, i+1]), tf.float64)
        e += -h_1/2 * tf.cast(c.expectation_ps(z=[i, i+1]), tf.float64)
    
    for i in range(n):
        e += -h_2/2 * tf.cast(c.expectation_ps(z=[i]), tf.float64)
        
    return tc.backend.real(e)


def vqe(param, n, h_1, h_2):

    
    circuit = tc.Circuit(n)
    paramc_1 = tc.backend.cast(
        tf.reshape(param[0:(n-1)*6],(n-1,6)), tc.dtypestr
    )
    
    paramc_2 = tc.backend.cast(
        tf.reshape(param[(n-1)*6:(n-1)*6+n*3],(n,3)), tc.dtypestr
    )

    circuit = block_circuit(n, paramc_1, paramc_2)
    energy = Hamiltonian(circuit, n, h_1, h_2)
    return energy

vqe_vvag = tc.backend.jit(
    tc.backend.vectorized_value_and_grad(vqe, vectorized_argnums = (0,)), static_argnums=(1,2,3)
)

def batched_train_step_tf(batch, n, h_1, h_2, GS_energy_h, rand_seed, maxiter=1000, random_idx=None, load_param=False, input_param=None):
    if load_param:
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(input_param, dtype=getattr(tf, tc.rdtypestr))
        )
    else:
        param = tf.Variable(
            initial_value=tf.concat(
                [tf.random.normal(shape=[int(batch/4), (n-1)*6+n*3], mean=0, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][0]),
                tf.random.normal(shape=[int(batch/4), (n-1)*6+n*3], mean=np.pi/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][1]),
                tf.random.normal(shape=[int(batch/4), (n-1)*6+n*3], mean=np.pi/2, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][2]),
                tf.random.normal(shape=[int(batch/4), (n-1)*6+n*3], mean=np.pi*3/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][3])
                ],0)
        )
    # pdb.set_trace()
    # pdb.set_trace()
    opt = tf.keras.optimizers.legacy.Adam(1e-2)
    energy_lowest = []
    for i in range(maxiter):
        start = time.time()
        energy, grad = vqe_vvag(param, n, h_1, h_2)
        opt.apply_gradients([(grad, param)])
        energy_lowest.append(np.min(energy))
        end = time.time()
        
        if i % 100 == 0:
            
            print(f'iter_{i}:')
            print(energy)
            print(f'lowest:{np.min(energy)}')
            print(f'time: {end-start}')
            if i > 0 and i % 1000 == 0:
                np.save(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed=2131558_iter={int(maxiter)}/params/h_1={h_1}_h_2={h_2}_iter={i}.npy',param)
                optim_result = np.min(np.array(energy_lowest))
                optim_results.append(optim_result)
                np.save(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed=2131558_iter={int(maxiter)}/results/h_1={h_1}_h_2={h_2}_iter={i}.npy',energy_lowest)
                
                plt.figure()
                plt.plot(energy_lowest)
                plt.xlabel('iter')
                plt.ylabel('energy')
                plt.title(f'unitary_XZ_Spin_Chain_8_h_1={h_1}_h_2={h_2}_GS_{GS_energy_h}')
                # plt.title(f'Z_{h}_GS_{GS_energy_h}')
                plt.savefig(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed=2131558_iter={int(maxiter)}/results/h_1={h_1}_h_2={h_2}_iter={i}.jpg')
            
    return energy_lowest, param


def GS_energy(n,h_1,h_2):
    list_1 = []
    list_2 = []
    list_3 = []

    pauli_list = []
    num = n
    
    for i in range(0,n-1):
        t1 = 'I'*(i) + 'X'*2 + 'I'*(n-i-2)
        t2 = 'I'*(i) + 'Z'*2 + 'I'*(n-i-2)
        list_1.append((t1, -1/2))
        list_2.append((t2, -h_1/2))

    for i in range(n):
        t3 = 'I'*(i)+'Z'+(n-i)*'I'
        list_3.append((t3, -h_2/2))
    
    pauli_list = list_1 + list_2 + list_3
    # pauli_list = list_3 + list_4
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    eigenvalues, eigenvectors = np.linalg.eig(H_m)
    sorted_eigenvalues = sorted(eigenvalues)
    return np.real(sorted_eigenvalues[0])


if __name__=="__main__":
    n = 8
    batch = 16
    rand_seed = 2131558
    np.random.seed(rand_seed)
    random_seed_array = np.random.randint(1,1000,(150,4))
    
    optim_results = []
    GS_Energy_list = []
    gap_list = []

    h_2_range = np.arange(0,10.0,0.1)
    max_iter = 2000
    
    h_1 = 0.0

    for i in range(100):
        h_2 = 0.1*i
        GS_energy_h = GS_energy(n,h_1,h_2)
        GS_Energy_list.append(GS_energy_h)
        print(f'Ground state energy for h_1={h_1}, h_2={h_2} is : {GS_Energy_list[i]}')
        energy_lowest, params = batched_train_step_tf(batch, n, h_1, h_2, GS_energy_h, random_seed_array, max_iter,random_idx=i, load_param=False)
        np.save(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed={int(rand_seed)}_iter={int(max_iter)}/params/h_1={h_1}_h_2={h_2}.npy', params)
        
        
        optim_result = np.min(np.array(energy_lowest))
        optim_results.append(optim_result)
        gap_list.append(optim_result-GS_energy_h)
        np.save(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed={int(rand_seed)}_iter={int(max_iter)}/results/h_1={h_1}_h_2={h_2}.npy',energy_lowest)
        
        plt.figure()
        plt.plot(energy_lowest)
        plt.xlabel('iter')
        plt.ylabel('energy')
        plt.title(f'unitary_XZ_Spin_Chain_8_h_1={h_1}_h_2={h_2}_GS_{GS_energy_h}')
        plt.savefig(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed={int(rand_seed)}_iter={int(max_iter)}/results/h_1={h_1}_h_2={h_2}.jpg')
        np.save(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed={int(rand_seed)}_iter={int(max_iter)}/results/unitary_XZ_Spin_Chain_h_1={h_1}_h_2={h_2}_optimization_process.npy',energy_lowest)
        
        print(f'Ground state energy: {GS_energy_h}')
        print(f'Best optimum result: {optim_result}')
        print(f'gap = {optim_result-GS_energy_h}')
        
    np.save(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed={int(rand_seed)}_iter={int(max_iter)}/results/optim_results.npy',np.array(optim_results))
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(h_2_range,GS_Energy_list,label='GS_energy')
    plt.plot(h_2_range,optim_results,label='optim_result')
    plt.xlabel('h_2')
    plt.ylabel('energy')
    plt.subplot(1,2,2)
    plt.plot(h_2_range,gap_list)
    plt.xlabel('h_2')
    plt.ylabel('energy')
    plt.savefig(f'./Unitary_VQE_Ising_8_results/Unitary-VQE_Ising_8_seed={int(rand_seed)}_iter={int(max_iter)}/results/final.jpg')
