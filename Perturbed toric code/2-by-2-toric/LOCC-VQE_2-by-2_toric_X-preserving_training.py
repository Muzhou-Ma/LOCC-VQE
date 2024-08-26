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

Geo_Z_interaction = [[[6,8],[1,8],[0,8],[2,8]],
                   [[7,9],[0,9],[1,9],[3,9]],
                   [[2,10],[5,10],[4,10],[6,10]],
                   [[3,11],[4,11],[5,11],[7,11]]]

Geo_X_interaction = [[0,12],[2,12],[3,12],[4,12],
                     [1,13],[3,13],[2,13],[5,13],
                     [4,14],[6,14],[7,14],[0,14],
                     [5,15],[7,15],[6,15],[1,15]]

def toric_syndrome_circuit(theta_Z):
    # theta_Z [16][3]
        
    toric_syndrome = tc.Circuit(12)
    for i in range(2*4):
        toric_syndrome.h(i) # initialize all physical qubits to plus state
    for i in range(len(Geo_Z_interaction)):
        for j in range(4):
            toric_syndrome.ry(Geo_Z_interaction[i][j][0],theta=np.pi/2)
            toric_syndrome.rxx(Geo_Z_interaction[i][j][0],Geo_Z_interaction[i][j][1],theta=theta_Z[4*i+j][0])
            toric_syndrome.rx(Geo_Z_interaction[i][j][0],theta=theta_Z[4*i+j][1])
            toric_syndrome.rx(Geo_Z_interaction[i][j][1],theta=theta_Z[4*i+j][2])
            
            toric_syndrome.ryy(Geo_Z_interaction[i][j][0],Geo_Z_interaction[i][j][1],theta=theta_Z[4*i+j][3])
            toric_syndrome.ry(Geo_Z_interaction[i][j][0],theta=theta_Z[4*i+j][4])
            toric_syndrome.ry(Geo_Z_interaction[i][j][1],theta=theta_Z[4*i+j][5])
            toric_syndrome.rzz(Geo_Z_interaction[i][j][0],Geo_Z_interaction[i][j][1],theta=theta_Z[4*i+j][6])
            toric_syndrome.rz(Geo_Z_interaction[i][j][0],theta=theta_Z[4*i+j][7])
            toric_syndrome.rz(Geo_Z_interaction[i][j][1],theta=theta_Z[4*i+j][8])
            toric_syndrome.ry(Geo_Z_interaction[i][j][0],theta=-np.pi/2)
        
        toric_syndrome.barrier_instruction()
    return toric_syndrome
               


def correction_circuit(params_corr_1, params_corr_2, params_corr_3):
    # params_corr_1 [4][8][3]
    # params_corr_2 [7][8][3]
    # params_corr_3 [8][3]
    corr_circuit = tc.Circuit(12)
    
    for i in range(4):
        for j in range(8):
            corr_circuit.ry(8+i,theta=np.pi/2)
            corr_circuit.rxx(8+i,j,theta=params_corr_1[i][j][0])
            corr_circuit.rx(8+i,theta=params_corr_1[i][j][1])
            corr_circuit.rx(j,theta=params_corr_1[i][j][2])
            corr_circuit.ry(8+i,theta=-np.pi/2)
        corr_circuit.barrier_instruction()
    
    
    
    corr_circuit.cx(8,9)
    corr_circuit.cx(8,10)
    corr_circuit.cx(8,11)
    for j in range(8):
        corr_circuit.ry(9,theta=np.pi/2)
        corr_circuit.rxx(9,j,theta=params_corr_2[0][j][0])
        corr_circuit.rx(9,theta=params_corr_2[0][j][1])
        corr_circuit.rx(j,theta=params_corr_2[0][j][2])
        corr_circuit.ry(9,theta=-np.pi/2)
        corr_circuit.barrier_instruction()
    for j in range(8):
        corr_circuit.ry(10,theta=np.pi/2)
        corr_circuit.rxx(10,j,theta=params_corr_2[1][j][0])
        corr_circuit.rx(10,theta=params_corr_2[1][j][1])
        corr_circuit.rx(j,theta=params_corr_2[1][j][2])
        corr_circuit.ry(10,theta=-np.pi/2)
        corr_circuit.barrier_instruction()
    for j in range(8):
        corr_circuit.ry(11,theta=np.pi/2)
        corr_circuit.rxx(11,j,theta=params_corr_2[2][j][0])
        corr_circuit.rx(11,theta=params_corr_2[2][j][1])
        corr_circuit.rx(j,theta=params_corr_2[2][j][2])
        corr_circuit.ry(11,theta=-np.pi/2)
        corr_circuit.barrier_instruction()
    corr_circuit.cx(8,10)
    corr_circuit.cx(8,11)
    
    corr_circuit.cx(9,10)
    corr_circuit.cx(9,11) 
    for j in range(8):
        corr_circuit.ry(10,theta=np.pi/2)
        corr_circuit.rxx(10,j,theta=params_corr_2[3][j][0])
        corr_circuit.rx(10,theta=params_corr_2[3][j][1])
        corr_circuit.rx(j,theta=params_corr_2[3][j][2])
        corr_circuit.ry(10,theta=-np.pi/2)
        corr_circuit.barrier_instruction()
    for j in range(8):
        corr_circuit.ry(11,theta=np.pi/2)
        corr_circuit.rxx(11,j,theta=params_corr_2[4][j][0])
        corr_circuit.rx(11,theta=params_corr_2[4][j][1])
        corr_circuit.rx(j,theta=params_corr_2[4][j][2])
        corr_circuit.ry(11,theta=-np.pi/2)
        corr_circuit.barrier_instruction()
    corr_circuit.cx(9,10)
    
    corr_circuit.cx(10,11)
    for j in range(8):
        corr_circuit.ry(11,theta=np.pi/2)
        corr_circuit.rxx(11,j,theta=params_corr_2[5][j][0])
        corr_circuit.rx(11,theta=params_corr_2[5][j][1])
        corr_circuit.rx(j,theta=params_corr_2[5][j][2])
        corr_circuit.ry(11,theta=-np.pi/2)
        corr_circuit.barrier_instruction()
    corr_circuit.cx(9,11)
    for j in range(8):
        corr_circuit.ry(11,theta=np.pi/2)
        corr_circuit.rxx(11,j,theta=params_corr_2[6][j][0])
        corr_circuit.rx(11,theta=params_corr_2[6][j][1])
        corr_circuit.rx(j,theta=params_corr_2[6][j][2])
        corr_circuit.ry(11,theta=-np.pi/2)
        corr_circuit.barrier_instruction()
    
    
    for i in range(8):
        corr_circuit.rx(i,theta=params_corr_3[i][0])
        corr_circuit.ry(i,theta=params_corr_3[i][1])
        corr_circuit.rz(i,theta=params_corr_3[i][2])
    
    return corr_circuit


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
        e += h * c.expectation_ps(y=[i])
        
    return -tc.backend.real(e)


def vqe(param, h):

    
    circuit = tc.Circuit(12)
    paramc_Z = tc.backend.cast(
        tf.reshape(param[0:16*9],(16,9)), tc.dtypestr
    )
    paramc_1 = tc.backend.cast(
        tf.reshape(param[16*9:16*9+4*8*3],(4,8,3)), tc.dtypestr
    )
    paramc_2 = tc.backend.cast(
        tf.reshape(param[16*9+4*8*3:16*9+4*8*3+7*8*3],(7,8,3)), tc.dtypestr
    )
    paramc_3 = tc.backend.cast(
        tf.reshape(param[16*9+4*8*3+7*8*3:16*9+4*8*3+7*8*3+8*3],(8,3)), tc.dtypestr
    )

    circuit.append(toric_syndrome_circuit(paramc_Z))
    circuit.append(correction_circuit(paramc_1,paramc_2,paramc_3))
    energy = Hamiltonian(circuit, h)
    return energy

vqe_vvag = tc.backend.jit(tc.backend.vectorized_value_and_grad(vqe, vectorized_argnums = (0,)))

def batched_train_step_tf(batch, h, rand_seed, maxiter=1000, random_idx=None, load_param=False, input_param=None):    
    if load_param:
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(input_param, dtype=getattr(tf, tc.rdtypestr))
        )
    else:
        param = tf.Variable(
            initial_value=tf.concat(
                [
                tf.random.normal(shape=[int(batch/4), 432], mean=0, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][0]),
                tf.random.normal(shape=[int(batch/4), 432], mean=np.pi/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][1]),
                tf.random.normal(shape=[int(batch/4), 432], mean=np.pi/2, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][2]),
                tf.random.normal(shape=[int(batch/4), 432], mean=np.pi*3/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][3])
                ],0)
        )

    
    opt = tf.keras.optimizers.legacy.Adam(1e-2)
    energy_lowest = []
    for i in range(maxiter):
        start = time.time()
        energy, grad = vqe_vvag(param, h)
        grad_Syn = grad[:,0:16*9] * (1.0+0.5)
        grad_Corr = grad[:,16*9:432]
        grad = tf.cast(tf.concat([grad_Syn,grad_Corr],axis=1),tf.float64)
        opt.apply_gradients([(grad, param)])
        energy_lowest.append(np.min(energy))
        if i % 5 == 0:
            print(f'iter_{i}:')
            print(energy)
            print(f'lowest:{np.min(energy)}')
        end = time.time()
        print(f'time: {end-start}')
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
        Pert = 'I'*(i)+'Y'+(8-i)*'I'
        pauli_list.append((Pert, -h))
        
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    eigenvalues, eigenvectors = np.linalg.eig(H_m)
    sorted_eigenvalues = sorted(eigenvalues)
    return np.real(sorted_eigenvalues[0])


if __name__=="__main__":
    batch = 64
    rand_seed = 2232119
    np.random.seed(rand_seed)
    random_seed_array = np.random.randint(1,100000,(11,4))
    
    optim_results = []
    GS_Energy_list = []
    gap_list = []
    h_range = np.arange(0,1.1,0.1)
    
    
    max_iter = 10000
    for i in range(11):
        h = i*0.1 + 0.5
        GS_energy_h = GS_energy(h)
        GS_Energy_list.append(GS_energy_h)
        print(f'Ground state energy for h={h} is : {GS_Energy_list[i]}')
        energy_lowest, param = batched_train_step_tf(batch,h,random_seed_array,max_iter,random_idx=i)
        
        np.save(f'./results/toric/toric_trail_1_x_preserving_moremoreSyn_seed={int(rand_seed)}_iter={int(max_iter)}_trained_Feb25/params_Y/Y_h={h}.npy', param)
        print(f'i={i},save_path=./results/toric/toric_trail_1_x_preserving_moremoreSyn_seed={int(rand_seed)}_iter={int(max_iter)}_trained_Feb25/params_Y/Y_h={h}.npy')
        
        optim_result = np.min(np.array(energy_lowest))
        optim_results.append(optim_result)
        gap_list.append(optim_result-GS_energy_h)
        np.save(f'./results/toric/toric_trail_1_x_preserving_moremoreSyn_seed={int(rand_seed)}_iter={int(max_iter)}_trained_Feb25/results_Y/Y_h={h}.npy',energy_lowest)
        plt.figure()
        plt.plot(energy_lowest)
        plt.xlabel('iter')
        plt.ylabel('energy')
        plt.title(f'Y_{h}_GS_{GS_energy_h}')
        plt.savefig(f'./results/toric/toric_trail_1_x_preserving_moremoreSyn_seed={int(rand_seed)}_iter={int(max_iter)}_trained_Feb25/results_Y/Y_h={h}.jpg')
        print(f'Ground state energy: {GS_energy_h}')
        print(f'Best optimum result: {optim_result}')
        print(f'gap = {optim_result-GS_energy_h}')
        
    np.save(f'./results/toric/toric_trail_1_x_preserving_moremoreSyn_seed={int(rand_seed)}_iter={int(max_iter)}_trained_Feb25/results_Y/Y_h={h}.npy',np.array(optim_results))
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(h_range,GS_Energy_list,label='GS_energy')
    plt.plot(h_range,optim_results,label='optim_result')
    plt.xlabel('h_Y')
    plt.ylabel('energy')
    plt.subplot(1,2,2)
    plt.plot(h_range,gap_list)
    plt.xlabel('h_Y')
    plt.ylabel('energy')
    plt.savefig(f'./results/toric/toric_trail_1_x_preserving_moremoreSyn_seed={int(rand_seed)}_iter={int(max_iter)}_trained_Feb25/results_Y/Y_final.jpg')
    
    