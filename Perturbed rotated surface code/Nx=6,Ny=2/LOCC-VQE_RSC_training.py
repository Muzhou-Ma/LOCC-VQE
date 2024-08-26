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

Geo_Z_interaction = [
                     [[1,13],[2,13]],
                     [[7,14],[8,14]],
                     [[4,16],[3,16]],
                     [[10,17],[9,17]],
                     [[0,12],[1,12],[6,12],[7,12]],
                     [[5,18],[4,18],[11,18],[10,18]],
                     [[2,15],[3,15],[8,15],[9,15]]
                     ] # The order here has been specifically designed for symmetry

Geo_X_interaction = [[[1,13],[2,13],[7,13],[8,13]],
                     [[3,17],[4,17],[9,17],[10,17]],
                     [[0,21],[6,21]],
                     [[5,22],[11,22]]
                    ]


def toric_syndrome_circuit(theta_Z):
    # theta_Z [14][6]
        
    toric_syndrome = tc.Circuit(19)

    theta_idx = 0
    
        
    for j in range(len(Geo_Z_interaction[1])):
        
        # Cartan decompostion
        toric_syndrome.rx(Geo_Z_interaction[1][j][0], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[1][j][0], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[1][j][0], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rx(Geo_Z_interaction[1][j][1], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[1][j][1], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[1][j][1], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rxx(Geo_Z_interaction[1][j][0],Geo_Z_interaction[1][j][1],theta = theta_Z[theta_idx+j][3])
        toric_syndrome.ryy(Geo_Z_interaction[1][j][0],Geo_Z_interaction[1][j][1],theta = theta_Z[theta_idx+j][4])
        toric_syndrome.rzz(Geo_Z_interaction[1][j][0],Geo_Z_interaction[1][j][1],theta = theta_Z[theta_idx+j][5])
        toric_syndrome.rx(Geo_Z_interaction[1][j][0], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[1][j][0], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[1][j][0], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rx(Geo_Z_interaction[1][j][1], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[1][j][1], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[1][j][1], theta = theta_Z[theta_idx+j][2])
        
    for j in range(len(Geo_Z_interaction[2])):
        
        # Cartan decompostion
        toric_syndrome.rx(Geo_Z_interaction[2][j][0], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[2][j][0], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[2][j][0], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rx(Geo_Z_interaction[2][j][1], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[2][j][1], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[2][j][1], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rxx(Geo_Z_interaction[2][j][0],Geo_Z_interaction[2][j][1],theta = theta_Z[theta_idx+j][3])
        toric_syndrome.ryy(Geo_Z_interaction[2][j][0],Geo_Z_interaction[2][j][1],theta = theta_Z[theta_idx+j][4])
        toric_syndrome.rzz(Geo_Z_interaction[2][j][0],Geo_Z_interaction[2][j][1],theta = theta_Z[theta_idx+j][5])
        toric_syndrome.rx(Geo_Z_interaction[2][j][0], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[2][j][0], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[2][j][0], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rx(Geo_Z_interaction[2][j][1], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[2][j][1], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[2][j][1], theta = theta_Z[theta_idx+j][2])
            
    theta_idx = theta_idx + len(Geo_Z_interaction[0])
        
    for i in range(2):
        for j in range(len(Geo_Z_interaction[i+4])):
        
            # Cartan decompostion
            toric_syndrome.rx(Geo_Z_interaction[i+4][j][0], theta = theta_Z[theta_idx+j][0])
            toric_syndrome.ry(Geo_Z_interaction[i+4][j][0], theta = theta_Z[theta_idx+j][1])
            toric_syndrome.rz(Geo_Z_interaction[i+4][j][0], theta = theta_Z[theta_idx+j][2])
            toric_syndrome.rx(Geo_Z_interaction[i+4][j][1], theta = theta_Z[theta_idx+j][0])
            toric_syndrome.ry(Geo_Z_interaction[i+4][j][1], theta = theta_Z[theta_idx+j][1])
            toric_syndrome.rz(Geo_Z_interaction[i+4][j][1], theta = theta_Z[theta_idx+j][2])
            toric_syndrome.rxx(Geo_Z_interaction[i+4][j][0],Geo_Z_interaction[i+4][j][1],theta = theta_Z[theta_idx+j][3])
            toric_syndrome.ryy(Geo_Z_interaction[i+4][j][0],Geo_Z_interaction[i+4][j][1],theta = theta_Z[theta_idx+j][4])
            toric_syndrome.rzz(Geo_Z_interaction[i+4][j][0],Geo_Z_interaction[i+4][j][1],theta = theta_Z[theta_idx+j][5])
            toric_syndrome.rx(Geo_Z_interaction[i+4][j][0], theta = theta_Z[theta_idx+j][0])
            toric_syndrome.ry(Geo_Z_interaction[i+4][j][0], theta = theta_Z[theta_idx+j][1])
            toric_syndrome.rz(Geo_Z_interaction[i+4][j][0], theta = theta_Z[theta_idx+j][2])
            toric_syndrome.rx(Geo_Z_interaction[i+4][j][1], theta = theta_Z[theta_idx+j][0])
            toric_syndrome.ry(Geo_Z_interaction[i+4][j][1], theta = theta_Z[theta_idx+j][1])
            toric_syndrome.rz(Geo_Z_interaction[i+4][j][1], theta = theta_Z[theta_idx+j][2])
        
    theta_idx = theta_idx + len(Geo_Z_interaction[4])
        
    for j in range(len(Geo_Z_interaction[i+4])):
        
        # Cartan decompostion
        toric_syndrome.rx(Geo_Z_interaction[6][j][0], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[6][j][0], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[6][j][0], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rx(Geo_Z_interaction[6][j][1], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[6][j][1], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[6][j][1], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rxx(Geo_Z_interaction[6][j][0],Geo_Z_interaction[6][j][1],theta = theta_Z[theta_idx+j][3])
        toric_syndrome.ryy(Geo_Z_interaction[6][j][0],Geo_Z_interaction[6][j][1],theta = theta_Z[theta_idx+j][4])
        toric_syndrome.rzz(Geo_Z_interaction[6][j][0],Geo_Z_interaction[6][j][1],theta = theta_Z[theta_idx+j][5])
        toric_syndrome.rx(Geo_Z_interaction[6][j][0], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[6][j][0], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[6][j][0], theta = theta_Z[theta_idx+j][2])
        toric_syndrome.rx(Geo_Z_interaction[6][j][1], theta = theta_Z[theta_idx+j][0])
        toric_syndrome.ry(Geo_Z_interaction[6][j][1], theta = theta_Z[theta_idx+j][1])
        toric_syndrome.rz(Geo_Z_interaction[6][j][1], theta = theta_Z[theta_idx+j][2])   
    
    theta_idx = theta_idx + len(Geo_Z_interaction[6])
    
    #due to symmetry, sharing the same parameters.
    for i in range(2):
        for j in range(len(Geo_X_interaction[i])):
            toric_syndrome.rx(Geo_X_interaction[i][j][0], theta = theta_Z[theta_idx+j][0]) # although we still call it theta_Z, it is actually parameters for X stabilizers
            toric_syndrome.ry(Geo_X_interaction[i][j][0], theta = theta_Z[theta_idx+j][1])
            toric_syndrome.rz(Geo_X_interaction[i][j][0], theta = theta_Z[theta_idx+j][2])
            toric_syndrome.rx(Geo_X_interaction[i][j][1], theta = theta_Z[theta_idx+j][0])
            toric_syndrome.ry(Geo_X_interaction[i][j][1], theta = theta_Z[theta_idx+j][1])
            toric_syndrome.rz(Geo_X_interaction[i][j][1], theta = theta_Z[theta_idx+j][2])
            toric_syndrome.rxx(Geo_X_interaction[i][j][0],Geo_X_interaction[i][j][1],theta = theta_Z[theta_idx+j][3])
            toric_syndrome.ryy(Geo_X_interaction[i][j][0],Geo_X_interaction[i][j][1],theta = theta_Z[theta_idx+j][4])
            toric_syndrome.rzz(Geo_X_interaction[i][j][0],Geo_X_interaction[i][j][1],theta = theta_Z[theta_idx+j][5])
            toric_syndrome.rx(Geo_X_interaction[i][j][0], theta = theta_Z[theta_idx+j][0])
            toric_syndrome.ry(Geo_X_interaction[i][j][0], theta = theta_Z[theta_idx+j][1])
            toric_syndrome.rz(Geo_X_interaction[i][j][0], theta = theta_Z[theta_idx+j][2])
            toric_syndrome.rx(Geo_X_interaction[i][j][1], theta = theta_Z[theta_idx+j][0])
            toric_syndrome.ry(Geo_X_interaction[i][j][1], theta = theta_Z[theta_idx+j][1])
            toric_syndrome.rz(Geo_X_interaction[i][j][1], theta = theta_Z[theta_idx+j][2])
     
        
    return toric_syndrome
               
def projector(projectors = 2 * np.ones(7, dtype=np.int32)):
    proj_circuit = tc.Circuit(19)
    projector_set = tf.cast(tf.constant([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]], [[1., 0.], [0., 1.]]]), ctype,)
    for i in range(7):
        proj_circuit.any(12+i, unitary = projector_set[projectors[i]])
    return proj_circuit
    

def correction_circuit(params_corr_1, params_corr_2, params_corr_3):
    # params_corr_1 [7][12][2]
    # params_corr_2 [14][12][2]
    # params_corr_3 [12][3]
    corr_circuit = tc.Circuit(19)
    
    

        
    for i in range(7):
        for j in range(12):
            corr_circuit.ry(12+i,theta=np.pi/2)
            corr_circuit.rxx(12+i,j,theta=params_corr_1[i][j][0])
            corr_circuit.rx(j,theta=params_corr_1[i][j][1])
            corr_circuit.ry(12+i,theta=-np.pi/2)
    
    
    
    
    corr_circuit.cx(12,18)
    corr_circuit.cx(13,17)
    corr_circuit.cx(14,16)
    for j in range(12):
        corr_circuit.ry(18,theta=np.pi/2)
        corr_circuit.rxx(18,j,theta=params_corr_2[0][j][0])
        corr_circuit.rx(j,theta=params_corr_2[0][j][1])
        corr_circuit.ry(18,theta=-np.pi/2)
        
    for j in range(12):
        corr_circuit.ry(17,theta=np.pi/2)
        corr_circuit.rxx(17,j,theta=params_corr_2[1][j][0])
        corr_circuit.rx(j,theta=params_corr_2[1][j][1])
        corr_circuit.ry(17,theta=-np.pi/2)
        
    for j in range(12):
        corr_circuit.ry(16,theta=np.pi/2)
        corr_circuit.rxx(16,j,theta=params_corr_2[2][j][0])
        corr_circuit.rx(j,theta=params_corr_2[2][j][1])
        corr_circuit.ry(16,theta=-np.pi/2)
    
    
    corr_circuit.cx(13,14)
    corr_circuit.cx(16,17)
    for j in range(12):
        corr_circuit.ry(14,theta=np.pi/2)
        corr_circuit.rxx(14,j,theta=params_corr_2[3][j][0])
        corr_circuit.rx(j,theta=params_corr_2[3][j][1])
        corr_circuit.ry(14,theta=-np.pi/2)
        
    for j in range(12):
        corr_circuit.ry(17,theta=np.pi/2)
        corr_circuit.rxx(17,j,theta=params_corr_2[4][j][0])
        corr_circuit.rx(j,theta=params_corr_2[4][j][1])
        corr_circuit.ry(17,theta=-np.pi/2)
        
        
    corr_circuit.cx(12,15)
    corr_circuit.cx(18,15)    
    
    for j in range(12):
        corr_circuit.ry(15,theta=np.pi/2)
        corr_circuit.rxx(15,j,theta=params_corr_2[5][j][0])
        corr_circuit.rx(j,theta=params_corr_2[5][j][1])
        corr_circuit.ry(15,theta=-np.pi/2)
    
    corr_circuit.cx(13,15)
    corr_circuit.cx(16,15)
    corr_circuit.cx(17,12)
    corr_circuit.cx(14,12)
    
    
    for j in range(12):
        corr_circuit.ry(15,theta=np.pi/2)
        corr_circuit.rxx(15,j,theta=params_corr_2[6][j][0])
        corr_circuit.rx(j,theta=params_corr_2[6][j][1])
        corr_circuit.ry(15,theta=-np.pi/2)

    
    for j in range(12):
        corr_circuit.ry(12,theta=np.pi/2)
        corr_circuit.rxx(12,j,theta=params_corr_2[7][j][0])
        corr_circuit.rx(j,theta=params_corr_2[7][j][1])
        corr_circuit.ry(12,theta=-np.pi/2) 
        
    corr_circuit.cx(16,18)
    corr_circuit.cx(14,18)
     
     
    for j in range(12):
        corr_circuit.ry(18,theta=np.pi/2)
        corr_circuit.rxx(18,j,theta=params_corr_2[8][j][0])
        corr_circuit.rx(j,theta=params_corr_2[8][j][1])
        corr_circuit.ry(18,theta=-np.pi/2)    
    corr_circuit.cx(16,15)
    corr_circuit.cx(14,15)
    
    for j in range(12):
        corr_circuit.ry(15,theta=np.pi/2)
        corr_circuit.rxx(15,j,theta=params_corr_2[9][j][0])
        corr_circuit.rx(j,theta=params_corr_2[9][j][1])
        corr_circuit.ry(15,theta=-np.pi/2)
        
    for i in range(12):
        corr_circuit.rx(i,theta=params_corr_3[i][0])
        corr_circuit.ry(i,theta=params_corr_3[i][1])
        corr_circuit.rz(i,theta=params_corr_3[i][2])
        
    return corr_circuit


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

    
    circuit = tc.Circuit(19)
    paramc_Z = tc.backend.cast(
        tf.reshape(param[0:14*6],(14,6)), tc.dtypestr
    )
    paramc_1 = tc.backend.cast(
        tf.reshape(param[14*6:14*6+7*12*2],(7,12,2)), tc.dtypestr
    )
    paramc_2 = tc.backend.cast(
        tf.reshape(param[14*6+7*12*2:14*6+7*12*2+10*12*2],(10,12,2)), tc.dtypestr
    )
    paramc_3 = tc.backend.cast(
        tf.reshape(param[14*6+7*12*2+10*12*2:14*6+7*12*2+10*12*2+12*3],(12,3)), tc.dtypestr
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
                tf.random.normal(shape=[int(batch/4), 528], mean=0, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][0]),
                tf.random.normal(shape=[int(batch/4), 528], mean=np.pi/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][1]),
                tf.random.normal(shape=[int(batch/4), 528], mean=np.pi/2, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][2]),
                tf.random.normal(shape=[int(batch/4), 528], mean=np.pi*3/4, stddev=0.2, dtype=getattr(tf, tc.rdtypestr), seed=rand_seed[int(random_idx)][3])
                ],0)
        )

    
    opt = tf.keras.optimizers.legacy.Adam(1e-2)
    energy_lowest = []
    for i in range(maxiter):
        start = time.time()
        energy, grad = vqe_vvag(param, h)

        opt.apply_gradients([(grad, param)])
        energy_lowest.append(np.min(energy))
        if i % 5 == 0:
            print(f'iter_{i}:')
            print(energy)
            print(f'lowest:{np.min(energy)}')
        end = time.time()
        print(f'time: {end-start}')
        
        if i % 50 == 1 and i / 50 > 0:
            np.save(f'./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed=2232119_iter={int(max_iter)}/params_Z/Z_h={h}_iter={i}.npy', param)
            
            optim_result = np.min(np.array(energy_lowest))
            optim_results.append(optim_result)
            np.save(f'./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed=2232119_iter={int(max_iter)}/results_Z/Z_h={h}_iter={i}.npy',energy_lowest)
            plt.figure()
            plt.plot(energy_lowest)
            plt.xlabel('iter')
            plt.ylabel('energy')
            plt.savefig(f'./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed=2232119_iter={int(max_iter)}/results_Z/Z_h={h}_iter={i}.jpg')
    
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

        np.save(f'./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/params_Z/Z_h={h}.npy', param)
        print(f'i={i},save_path=./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/params_Z/Z_h={h}.npy')
        
        optim_result = np.min(np.array(energy_lowest))
        optim_results.append(optim_result)
        gap_list.append(optim_result-GS_energy_h)
        np.save(f'./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.npy',energy_lowest)
        plt.figure()
        plt.plot(energy_lowest)
        plt.xlabel('iter')
        plt.ylabel('energy')
        plt.title(f'Z_{h}_GS_{GS_energy_h}')
        plt.savefig(f'./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.jpg')
        print(f'Ground state energy: {GS_energy_h}')
        print(f'Best optimum result: {optim_result}')
        print(f'gap = {optim_result-GS_energy_h}')
        
    np.save(f'./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.npy',np.array(optim_results))
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
    plt.savefig(f'./LOCC-VQE_RSC_results/Rec_toric_trail_1_borrowZforX_sym_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}/results_Z/Z_h={h}.jpg')
    
    