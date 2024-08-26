import tensorcircuit as tc
import tensorflow as tf
import tensornetwork as tn
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp


tc.set_backend("tensorflow")
ctype, rtype = tc.set_dtype("complex64")

Brickwise_interaction_layer_1 = [[0,12],[6,13],[1,14],[7,15],[16,8],[17,3],[9,19],[20,10],[21,5],[11,22]]
Brickwise_interaction_layer_2 = [[6,12],[0,13],[7,16],[1,15],[2,14],[9,17],[3,19],[4,18],[11,21],[5,22]]
Brickwise_interaction_layer_3 = [[0,12],[1,13],[2,15],[8,17],[3,18],[4,19],[10,21],[11,22]]
Brickwise_interaction_layer_4 = [[6,12],[7,13],[8,15],[2,17],[9,20],[10,19],[4,21],[5,22]]



def unitary_circuit(param_1, param_2, param_3,param_4,param_5):
    circuit = tc.Circuit(23)
    for i in range(23):
        circuit.h(i)
        
    for i in range(10):
        circuit.rx(Brickwise_interaction_layer_1[i][0], theta = param_1[i][0])
        circuit.ry(Brickwise_interaction_layer_1[i][0], theta = param_1[i][1])
        circuit.rz(Brickwise_interaction_layer_1[i][0], theta = param_1[i][2])
        circuit.rx(Brickwise_interaction_layer_1[i][1], theta = param_1[i][0])
        circuit.ry(Brickwise_interaction_layer_1[i][1], theta = param_1[i][1])
        circuit.rz(Brickwise_interaction_layer_1[i][1], theta = param_1[i][2])
        circuit.rxx(Brickwise_interaction_layer_1[i][0], Brickwise_interaction_layer_1[i][1], theta = param_1[i][3])
        circuit.ryy(Brickwise_interaction_layer_1[i][0], Brickwise_interaction_layer_1[i][1], theta = param_1[i][4])
        circuit.rzz(Brickwise_interaction_layer_1[i][0], Brickwise_interaction_layer_1[i][1], theta = param_1[i][5])
        circuit.rx(Brickwise_interaction_layer_1[i][0], theta = param_1[i][0])
        circuit.ry(Brickwise_interaction_layer_1[i][0], theta = param_1[i][1])
        circuit.rz(Brickwise_interaction_layer_1[i][0], theta = param_1[i][2])
        circuit.rx(Brickwise_interaction_layer_1[i][1], theta = param_1[i][0])
        circuit.ry(Brickwise_interaction_layer_1[i][1], theta = param_1[i][1])
        circuit.rz(Brickwise_interaction_layer_1[i][1], theta = param_1[i][2])
        
    for i in range(10):
        circuit.rx(Brickwise_interaction_layer_2[i][0], theta = param_2[i][0])
        circuit.ry(Brickwise_interaction_layer_2[i][0], theta = param_2[i][1])
        circuit.rz(Brickwise_interaction_layer_2[i][0], theta = param_2[i][2])
        circuit.rx(Brickwise_interaction_layer_2[i][1], theta = param_2[i][0])
        circuit.ry(Brickwise_interaction_layer_2[i][1], theta = param_2[i][1])
        circuit.rz(Brickwise_interaction_layer_2[i][1], theta = param_2[i][2])
        circuit.rxx(Brickwise_interaction_layer_2[i][0], Brickwise_interaction_layer_2[i][1], theta = param_2[i][3])
        circuit.ryy(Brickwise_interaction_layer_2[i][0], Brickwise_interaction_layer_2[i][1], theta = param_2[i][4])
        circuit.rzz(Brickwise_interaction_layer_2[i][0], Brickwise_interaction_layer_2[i][1], theta = param_2[i][5])
        circuit.rx(Brickwise_interaction_layer_2[i][0], theta = param_2[i][0])
        circuit.ry(Brickwise_interaction_layer_2[i][0], theta = param_2[i][1])
        circuit.rz(Brickwise_interaction_layer_2[i][0], theta = param_2[i][2])
        circuit.rx(Brickwise_interaction_layer_2[i][1], theta = param_2[i][0])
        circuit.ry(Brickwise_interaction_layer_2[i][1], theta = param_2[i][1])
        circuit.rz(Brickwise_interaction_layer_2[i][1], theta = param_2[i][2])
    
    for i in range(8):
        circuit.rx(Brickwise_interaction_layer_3[i][0], theta = param_3[i][0])
        circuit.ry(Brickwise_interaction_layer_3[i][0], theta = param_3[i][1])
        circuit.rz(Brickwise_interaction_layer_3[i][0], theta = param_3[i][2])
        circuit.rx(Brickwise_interaction_layer_3[i][1], theta = param_3[i][0])
        circuit.ry(Brickwise_interaction_layer_3[i][1], theta = param_3[i][1])
        circuit.rz(Brickwise_interaction_layer_3[i][1], theta = param_3[i][2])
        circuit.rxx(Brickwise_interaction_layer_3[i][0], Brickwise_interaction_layer_3[i][1], theta = param_3[i][3])
        circuit.ryy(Brickwise_interaction_layer_3[i][0], Brickwise_interaction_layer_3[i][1], theta = param_3[i][4])
        circuit.rzz(Brickwise_interaction_layer_3[i][0], Brickwise_interaction_layer_3[i][1], theta = param_3[i][5])
        circuit.rx(Brickwise_interaction_layer_3[i][0], theta = param_3[i][0])
        circuit.ry(Brickwise_interaction_layer_3[i][0], theta = param_3[i][1])
        circuit.rz(Brickwise_interaction_layer_3[i][0], theta = param_3[i][2])
        circuit.rx(Brickwise_interaction_layer_3[i][1], theta = param_3[i][0])
        circuit.ry(Brickwise_interaction_layer_3[i][1], theta = param_3[i][1])
        circuit.rz(Brickwise_interaction_layer_3[i][1], theta = param_3[i][2])
        
    for i in range(8):
        circuit.rx(Brickwise_interaction_layer_4[i][0], theta = param_4[i][0])
        circuit.ry(Brickwise_interaction_layer_4[i][0], theta = param_4[i][1])
        circuit.rz(Brickwise_interaction_layer_4[i][0], theta = param_4[i][2])
        circuit.rx(Brickwise_interaction_layer_4[i][1], theta = param_4[i][0])
        circuit.ry(Brickwise_interaction_layer_4[i][1], theta = param_4[i][1])
        circuit.rz(Brickwise_interaction_layer_4[i][1], theta = param_4[i][2])
        circuit.rxx(Brickwise_interaction_layer_4[i][0], Brickwise_interaction_layer_4[i][1], theta = param_4[i][3])
        circuit.ryy(Brickwise_interaction_layer_4[i][0], Brickwise_interaction_layer_4[i][1], theta = param_4[i][4])
        circuit.rzz(Brickwise_interaction_layer_4[i][0], Brickwise_interaction_layer_4[i][1], theta = param_4[i][5])
        circuit.rx(Brickwise_interaction_layer_4[i][0], theta = param_4[i][0])
        circuit.ry(Brickwise_interaction_layer_4[i][0], theta = param_4[i][1])
        circuit.rz(Brickwise_interaction_layer_4[i][0], theta = param_4[i][2])
        circuit.rx(Brickwise_interaction_layer_4[i][1], theta = param_4[i][0])
        circuit.ry(Brickwise_interaction_layer_4[i][1], theta = param_4[i][1])
        circuit.rz(Brickwise_interaction_layer_4[i][1], theta = param_4[i][2])
        
    for i in range(23):
        circuit.rx(i, theta = param_5[i][0])
        circuit.ry(i, theta = param_5[i][1])
        circuit.rz(i, theta = param_5[i][2])
        
    return circuit



def Hamiltonian(c: tc.Circuit, h: float = 0):
    e = 0.0
    e += (1-h) * c.expectation_ps(x=[0, 6])
    e += (1-h) * c.expectation_ps(x=[1, 2, 7, 8])
    e += (1-h) * c.expectation_ps(x=[3, 4, 9, 10]) # this
    e += (1-h) * c.expectation_ps(x=[5, 11])
    
    e += (1-h) * c.expectation_ps(z=[0, 1, 6, 7])
    e += (1-h) * c.expectation_ps(z=[1, 2])
    e += (1-h) * c.expectation_ps(z=[7, 8])
    e += (1-h) * c.expectation_ps(z=[2, 3, 8, 9])
    e += (1-h) * c.expectation_ps(z=[3, 4])
    e += (1-h) * c.expectation_ps(z=[9, 10])
    e += (1-h) * c.expectation_ps(z=[4, 5, 10, 11])
    
    for i in range(12):
        e += h * c.expectation_ps(z=[i])
        
    return -tc.backend.real(e)

def vqe(param, h):

    
    param_1 = tc.backend.cast(
        tf.reshape(param[0:10*6],(10,6)), tc.dtypestr
    )
    param_2 = tc.backend.cast(
        tf.reshape(param[10*6:10*6+10*6],(10,6)), tc.dtypestr
    )
    param_3 = tc.backend.cast(
        tf.reshape(param[10*6+10*6:10*6+10*6+8*6],(8,6)), tc.dtypestr
    )
    param_4 = tc.backend.cast(
        tf.reshape(param[10*6+10*6+8*6:10*6+10*6+8*6+8*6],(8,6)), tc.dtypestr
    )
    param_5 = tc.backend.cast(
        tf.reshape(param[10*6+10*6+8*6+8*6:10*6+10*6+8*6+8*6+23*3],(23,3)), tc.dtypestr
    ) 
    circuit = unitary_circuit(param_1, param_2, param_3, param_4, param_5)
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
                tf.random.normal(shape=[int(batch/4), 285], mean=0, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][0]),
                tf.random.normal(shape=[int(batch/4), 285], mean=np.pi/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][1]),
                tf.random.normal(shape=[int(batch/4), 285], mean=np.pi/2, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][2]),
                tf.random.normal(shape=[int(batch/4), 285], mean=np.pi*3/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][3])
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
        
        if i % 100 == 1 and i / 100 > 0:
            np.save(f'Unitary-VQE_RSC_results/unitary_Rec_toric_trail_1_sym_depth_4_seed=2232119_iter={int(max_iter)}/params_Z/Z_h={h}_iter={i}.npy', param)
            # print(f'i={i},save_path=./results/toric/Rec_toric_trail_1_x_preserving_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/params_Z/Z_h={h}_iter={i}.npy')
            
            optim_result = np.min(np.array(energy_lowest))
            optim_results.append(optim_result)
            np.save(f'Unitary-VQE_RSC_results/unitary_Rec_toric_trail_1_sym_depth_4_seed=2232119_iter={int(max_iter)}/results_Z/Z_h={h}_iter={i}.npy',energy_lowest)
            plt.figure()
            plt.plot(energy_lowest)
            plt.xlabel('iter')
            plt.ylabel('energy')
            # plt.title(f'Z_{h}_GS_{GS_energy_h}')
            plt.savefig(f'Unitary-VQE_RSC_results/unitary_Rec_toric_trail_1_sym_depth_4_seed=2232119_iter={int(max_iter)}/results_Z/Z_h={h}_iter={i}.jpg')
    
    return energy_lowest, param


def GS_energy(h):
    pauli_list = []
    
    X_0 = 'XIIIIIXIIIII'
    X_1 = 'IXXIIIIXXIII'
    X_2 = 'IIIXXIIIIXXI'
    X_3 = 'IIIIIXIIIIIX'
    Z_0 = 'ZZIIIIZZIIII'
    Z_1 = 'IZZIIIIIIIII'
    Z_2 = 'IIIIIIIZZIII'
    Z_3 = 'IIZZIIIIZZII'
    Z_4 = 'IIIZZIIIIIII'
    Z_5 = 'IIIIIIIIIZZI'
    Z_6 = 'IIIIZZIIIIZZ'
    

    
    pauli_list.append((X_0, -(1-h)))
    pauli_list.append((X_1, -(1-h)))
    pauli_list.append((X_2, -(1-h)))
    pauli_list.append((X_3, -(1-h)))
    pauli_list.append((Z_0, -(1-h)))
    pauli_list.append((Z_1, -(1-h)))
    pauli_list.append((Z_2, -(1-h)))
    pauli_list.append((Z_3, -(1-h)))
    pauli_list.append((Z_4, -(1-h)))
    pauli_list.append((Z_5, -(1-h)))
    pauli_list.append((Z_6, -(1-h)))

    for i in range(12):
        Pert = 'I'*(i)+'Z'+(11-i)*'I'
        pauli_list.append((Pert, -h))
        
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    eigenvalues, eigenvectors = np.linalg.eig(H_m)
    sorted_eigenvalues = sorted(eigenvalues)
    return np.real(sorted_eigenvalues[0])


if __name__=="__main__":
    batch = 16
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
        energy_lowest, param = batched_train_step_tf(batch,h,random_seed_array,max_iter,random_idx=i)
        
        optim_result = np.min(np.array(energy_lowest))
        optim_results.append(optim_result)
        gap_list.append(optim_result-GS_energy_h)
        np.save(f'Unitary-VQE_RSC_results/unitary_Rec_toric_trail_1_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.npy',energy_lowest)
        plt.figure()
        plt.plot(energy_lowest)
        plt.xlabel('iter')
        plt.ylabel('energy')
        plt.title(f'Z_{h}_GS_{GS_energy_h}')
        plt.savefig(f'Unitary-VQE_RSC_results/unitary_Rec_toric_trail_1_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.jpg')
        print(f'Ground state energy: {GS_energy_h}')
        print(f'Best optimum result: {optim_result}')
        print(f'gap = {optim_result-GS_energy_h}')
        
    np.save(f'Unitary-VQE_RSC_results/unitary_Rec_toric_trail_1_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.npy',np.array(optim_results))
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
    plt.savefig(f'Unitary-VQE_RSC_results/unitary_Rec_toric_trail_1_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.jpg')
    
    