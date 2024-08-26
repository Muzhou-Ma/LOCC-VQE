import tensorcircuit as tc
import tensorflow as tf
import tensornetwork as tn
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp


tc.set_backend("tensorflow")
tc.set_dtype("complex64")

Geo_Z_interaction = [[6,8],[1,8],[0,8],[2,8],
                   [7,9],[0,9],[1,9],[3,9],
                   [2,10],[5,10],[4,10],[6,10],
                   [3,11],[4,11],[5,11],[7,11]]

Geo_X_interaction = [[0,12],[2,12],[3,12],[4,12],
                     [1,13],[3,13],[2,13],[5,13],
                     [4,14],[6,14],[7,14],[0,14],
                     [5,15],[7,15],[6,15],[1,15]]


def Unitary_circuit(param_1,param_2,param_3):
    circuit = tc.Circuit(16)
    for i in range(len(Geo_Z_interaction)):
        circuit.rx(Geo_Z_interaction[i][0], theta = param_1[i][0])
        circuit.ry(Geo_Z_interaction[i][0], theta = param_1[i][1])
        circuit.rz(Geo_Z_interaction[i][0], theta = param_1[i][2])
        circuit.rx(Geo_Z_interaction[i][1], theta = param_1[i][0])
        circuit.ry(Geo_Z_interaction[i][1], theta = param_1[i][1])
        circuit.rz(Geo_Z_interaction[i][1], theta = param_1[i][2])
        circuit.rxx(Geo_Z_interaction[i][0], Geo_Z_interaction[i][1], theta = param_1[i][3])
        circuit.ryy(Geo_Z_interaction[i][0], Geo_Z_interaction[i][1], theta = param_1[i][4])
        circuit.rzz(Geo_Z_interaction[i][0], Geo_Z_interaction[i][1], theta = param_1[i][5])
        circuit.rx(Geo_Z_interaction[i][0], theta = param_1[i][0])
        circuit.ry(Geo_Z_interaction[i][0], theta = param_1[i][1])
        circuit.rz(Geo_Z_interaction[i][0], theta = param_1[i][2])
        circuit.rx(Geo_Z_interaction[i][1], theta = param_1[i][0])
        circuit.ry(Geo_Z_interaction[i][1], theta = param_1[i][1])
        circuit.rz(Geo_Z_interaction[i][1], theta = param_1[i][2])
        
    for i in range(len(Geo_X_interaction)):
        circuit.rx(Geo_X_interaction[i][0], theta = param_2[i][0])
        circuit.ry(Geo_X_interaction[i][0], theta = param_2[i][1])
        circuit.rz(Geo_X_interaction[i][0], theta = param_2[i][2])
        circuit.rx(Geo_X_interaction[i][1], theta = param_2[i][0])
        circuit.ry(Geo_X_interaction[i][1], theta = param_2[i][1])
        circuit.rz(Geo_X_interaction[i][1], theta = param_2[i][2])
        circuit.rxx(Geo_X_interaction[i][0], Geo_X_interaction[i][1], theta = param_2[i][3])
        circuit.ryy(Geo_X_interaction[i][0], Geo_X_interaction[i][1], theta = param_2[i][4])
        circuit.rzz(Geo_X_interaction[i][0], Geo_X_interaction[i][1], theta = param_2[i][5])
        circuit.rx(Geo_X_interaction[i][0], theta = param_2[i][0])
        circuit.ry(Geo_X_interaction[i][0], theta = param_2[i][1])
        circuit.rz(Geo_X_interaction[i][0], theta = param_2[i][2])
        circuit.rx(Geo_X_interaction[i][1], theta = param_2[i][0])
        circuit.ry(Geo_X_interaction[i][1], theta = param_2[i][1])
        circuit.rz(Geo_X_interaction[i][1], theta = param_2[i][2])
        
    for i in range(16):
        circuit.rx(i, theta = param_3[i][0])
        circuit.ry(i, theta = param_3[i][1])
        circuit.rz(i, theta = param_3[i][2])
    
    return circuit
    



def Hamiltonian(c: tc.Circuit, h: float = 0):
    e = 0.0
    e += (1-h) * c.expectation_ps(z=[6, 1, 0, 2])
    e += (1-h) * c.expectation_ps(z=[7, 0, 1, 3])
    e += (1-h) * c.expectation_ps(z=[2, 5, 4, 6])
    e += (1-h) * c.expectation_ps(z=[3, 4, 5, 7])
    
    
    e += (1-h) * c.expectation_ps(x=[0, 2, 3, 4])
    e += (1-h) * c.expectation_ps(x=[1, 3, 2, 5])
    e += (1-h) * c.expectation_ps(x=[4, 6, 7, 0])
    e += (1-h) * c.expectation_ps(x=[5, 7, 6, 1])
    
    for i in range(8):
        e += h * c.expectation_ps(z=[i])
        
    return -tc.backend.real(e)


def vqe(param, h):

    paramc_1 = tc.backend.cast(
        tf.reshape(param[0:16*6],(16,6)), tc.dtypestr
    )
    paramc_2 = tc.backend.cast(
        tf.reshape(param[16*6:16*6+16*6],(16,6)), tc.dtypestr
    )
    paramc_3 = tc.backend.cast(
        tf.reshape(param[16*6+16*6:16*6+16*6+16*3],(16,3)), tc.dtypestr
    )
    
    circuit = Unitary_circuit(paramc_1,paramc_2,paramc_3)
    energy = Hamiltonian(circuit, h)
    return energy


vqe_vvag = tc.backend.jit(tc.backend.vectorized_value_and_grad(vqe, vectorized_argnums = (0,)))



def batched_train_step_tf(batch, h, rand_seed, maxiter=1000, random_idx=None, load_param=False, input_param=None):    
    if load_param:
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(input_param, dtype=getattr(tf, tc.rdtypestr))
        )
    else:
        # pdb.set_trace()
        param = tf.Variable(
            initial_value=tf.concat(
                [
                tf.random.normal(shape=[int(batch/4), 240], mean=0, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][0]),
                tf.random.normal(shape=[int(batch/4), 240], mean=np.pi/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][1]),
                tf.random.normal(shape=[int(batch/4), 240], mean=np.pi/2, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][2]),
                tf.random.normal(shape=[int(batch/4), 240], mean=np.pi*3/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][3])
                ],0)
        )

    
    opt = tf.keras.optimizers.legacy.Adam(1e-2)
    energy_lowest = []
    for i in range(maxiter):
        start = time.time()
        energy, grad = vqe_vvag(param, h)
        
        opt.apply_gradients([(grad, param)])
        energy_lowest.append(np.min(energy))
        if i % 10 == 0:
            print(f'iter_{i}:')
            print(energy)
            print(f'lowest:{np.min(energy)}')
        end = time.time()
        print(f'time: {end-start}')
        
        if i % 200 == 1 and i / 100 > 0:
            np.save(f'./results/unitary_square_toric/unitary_square_toric_trail_1_depth_4_seed=2232119_iter={int(max_iter)}/params_Z/Z_h={h}_iter={i}.npy', param)
            # print(f'i={i},save_path=./results/toric/Rec_toric_trail_1_x_preserving_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/params_Z/Z_h={h}_iter={i}.npy')
            
            optim_result = np.min(np.array(energy_lowest))
            optim_results.append(optim_result)
            np.save(f'./results/unitary_square_toric/unitary_square_toric_trail_1_depth_4_seed=2232119_iter={int(max_iter)}/results_Z/Z_h={h}_iter={i}.npy',energy_lowest)
            plt.figure()
            plt.plot(energy_lowest)
            plt.xlabel('iter')
            plt.ylabel('energy')
            # plt.title(f'Z_{h}_GS_{GS_energy_h}')
            plt.savefig(f'./results/unitary_square_toric/unitary_square_toric_trail_1_depth_4_seed=2232119_iter={int(max_iter)}/results_Z/Z_h={h}_iter={i}.jpg')
    
    return energy_lowest, param


def GS_energy(h):
    pauli_list = []
    
    Z_0 = 'ZZZIIIZI'
    Z_1 = 'ZZIZIIIZ'
    Z_2 = 'IIZIZZZI'
    Z_3 = 'IIIZZZIZ'
    X_0 = 'XIXXXIII'
    X_1 = 'IXXXIXII'
    X_2 = 'XIIIXIXX'
    X_3 = 'IXIIIXXX'
    

    
    pauli_list.append((Z_0, -(1-h)))
    pauli_list.append((Z_1, -(1-h)))
    pauli_list.append((Z_2, -(1-h)))
    pauli_list.append((Z_3, -(1-h)))
    pauli_list.append((X_0, -(1-h)))
    pauli_list.append((X_1, -(1-h)))
    pauli_list.append((X_2, -(1-h)))
    pauli_list.append((X_3, -(1-h)))
    
    for i in range(8):
        Pert = 'I'*(i)+'Z'+(8-i)*'I'
        pauli_list.append((Pert, -h))
        
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    eigenvalues, eigenvectors = np.linalg.eig(H_m)
    sorted_eigenvalues = sorted(eigenvalues)
    return np.real(sorted_eigenvalues[0])


if __name__=="__main__":
    batch = 48
    rand_seed = 2232119
    np.random.seed(rand_seed)
    random_seed_array = np.random.randint(1,100000,(11,4))
    
    optim_results = []
    GS_Energy_list = []
    gap_list = []
    h_range = np.arange(0,1.1,0.1)
    
    
    max_iter = 500
    for i in range(11):
        h = i*0.1
        GS_energy_h = GS_energy(h)
        GS_Energy_list.append(GS_energy_h)
        print(f'Ground state energy for h={h} is : {GS_Energy_list[i]}')
        # loaded_params = np.load(f'./results/unitary_toric/unitary_Rec_toric_trail_1_sym_depth_4_seed=2232119_iter=500/params_Z/opt_params/Z_h={h}.npy')
        # energy_lowest, param = batched_train_step_tf(batch,h,rand_seed,1,random_idx=i, load_param=True,input_param=loaded_params)
        energy_lowest, param = batched_train_step_tf(batch,h,random_seed_array,max_iter,random_idx=i)
        # print(energy_lowest)
        
        np.save(f'./results/unitary_square_toric/unitary_square_toric_trail_1_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/params_Z/Z_h={h}.npy',param)
        optim_result = np.min(np.array(energy_lowest))
        optim_results.append(optim_result)
        gap_list.append(optim_result-GS_energy_h)
        
        np.save(f'./results/unitary_square_toric/unitary_square_toric_trail_1_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.npy',energy_lowest)
        plt.figure()
        plt.plot(energy_lowest)
        plt.xlabel('iter')
        plt.ylabel('energy')
        plt.title(f'Z_{h}_GS_{GS_energy_h}')
        plt.savefig(f'./results/unitary_square_toric/unitary_square_toric_trail_1_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.jpg')
        print(f'Ground state energy: {GS_energy_h}')
        print(f'Best optimum result: {optim_result}')
        print(f'gap = {optim_result-GS_energy_h}')
        
    np.save(f'./results/unitary_square_toric/unitary_square_toric_trail_1_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/optim_results.npy',np.array(optim_results))
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(h_range,GS_Energy_list,label='GS_energy')
    plt.plot(h_range,optim_results,label='optim_result')
    plt.xlabel('h_Z')
    plt.ylabel('energy')
    plt.subplot(1,2,2)
    plt.plot(h_range,gap_list)
    plt.xlabel('h_Z')
    plt.ylabel('energy')
    plt.savefig(f'./results/unitary_square_toric/unitary_square_toric_trail_1_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.jpg')
    
    