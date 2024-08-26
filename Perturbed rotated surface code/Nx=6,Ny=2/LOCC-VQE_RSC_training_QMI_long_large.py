import tensorcircuit as tc
import tensorflow as tf
import tensornetwork as tn
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp


import cotengra as ctg

optr = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel=True,
    minimize="flops",
    max_time=120,
    max_repeats=4096,
    progbar=True,
)
tc.set_contractor("custom", optimizer=optr, preprocessing=True)

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
        
    # theta_Z = theta_Z*0.0
    toric_syndrome = tc.Circuit(19)
    # for i in range(4):
    #     toric_syndrome.x(i+8)
    # for i in range(12):
        # toric_syndrome.h(i) # initialize all physical qubits to plus state
    
    
    
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
    # params_corr_3 [15][3]
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
        
    for j in range(12):
        corr_circuit.ry(16,theta=np.pi/2)
        corr_circuit.rxx(16,j,theta=params_corr_2[10][j][0])
        corr_circuit.rx(j,theta=params_corr_2[10][j][1])
        corr_circuit.ry(16,theta=-np.pi/2)
               
    for j in range(12):
        corr_circuit.ry(18,theta=np.pi/2)
        corr_circuit.rxx(18,j,theta=params_corr_2[11][j][0])
        corr_circuit.rx(j,theta=params_corr_2[11][j][1])
        corr_circuit.ry(18,theta=-np.pi/2)
    
       
    for j in range(12):
        corr_circuit.ry(17,theta=np.pi/2)
        corr_circuit.rxx(17,j,theta=params_corr_2[12][j][0])
        corr_circuit.rx(j,theta=params_corr_2[12][j][1])
        corr_circuit.ry(17,theta=-np.pi/2)
    
     
        
        
    corr_circuit.cx(12,17)
    corr_circuit.cx(13,16)
    
        
    for j in range(12):
        corr_circuit.ry(17,theta=np.pi/2)
        corr_circuit.rxx(17,j,theta=params_corr_2[13][j][0])
        corr_circuit.rx(j,theta=params_corr_2[13][j][1])
        corr_circuit.ry(17,theta=-np.pi/2)
        
    for j in range(12):
        corr_circuit.ry(16,theta=np.pi/2)
        corr_circuit.rxx(16,j,theta=params_corr_2[14][j][0])
        corr_circuit.rx(j,theta=params_corr_2[14][j][1])
        corr_circuit.ry(16,theta=-np.pi/2)
    
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
    
    e += 4*(1-h) * c.expectation_ps(x=[0,1,2,3,4,5])
    
    for i in range(12):
        e += h * c.expectation_ps(z=[i])
        
    return -tc.backend.real(e)


def QMI(c: tc.Circuit):
    s = c.state()
    SL = tc.quantum.entanglement_entropy(s, cut=[1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18])
    SR = tc.quantum.entanglement_entropy(s, cut=[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,17,18])
    SRL = tc.quantum.entanglement_entropy(s, cut=[1,2,3,4,7,8,9,10,12,13,14,15,16,17,18])
    qmi = SL+SR-SRL
    return qmi

def vqe(param, h):

    
    circuit = tc.Circuit(19)
    paramc_Z = tc.backend.cast(
        tf.reshape(param[0:14*6],(14,6)), tc.dtypestr
    )
    paramc_1 = tc.backend.cast(
        tf.reshape(param[14*6:14*6+7*12*2],(7,12,2)), tc.dtypestr
    )
    paramc_2 = tc.backend.cast(
        tf.reshape(param[14*6+7*12*2:14*6+7*12*2+15*12*2],(15,12,2)), tc.dtypestr
    )
    paramc_3 = tc.backend.cast(
        tf.reshape(param[14*6+7*12*2+15*12*2:14*6+7*12*2+15*12*2+12*3],(12,3)), tc.dtypestr
    )

    circuit.append(toric_syndrome_circuit(paramc_Z))
    circuit.append(correction_circuit(paramc_1,paramc_2,paramc_3))
    qmi = QMI(circuit)
    energy = Hamiltonian(circuit, h)
    return energy, qmi

vqe_vmap = tc.backend.jit(tc.backend.vmap(vqe, vectorized_argnums = (0,)))



def batched_train_step_tf(batch, h, rand_seed, maxiter=1000, random_idx=None, load_param=False, input_param=None):    
    param = tf.Variable(
            initial_value=tf.convert_to_tensor(input_param, dtype=getattr(tf, tc.rdtypestr))
        )

    energy, qmi = vqe_vmap(param, h)
    
    
    return energy, qmi


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
    pauli_list.append(('XXXXXXIIIIII', -4*(1-h)))

    for i in range(12):
        Pert = 'I'*(i)+'Z'+(12-i)*'I'
        pauli_list.append((Pert, -h))
        
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    eigenvalues, eigenvectors = np.linalg.eig(H_m)
    sorted_eigenvalues = sorted(eigenvalues)
    return np.real(sorted_eigenvalues[0])


if __name__=="__main__":
    batch = 32
    rand_seed = 2232119
    np.random.seed(rand_seed)
    random_seed_array = np.random.randint(1,100000,(11,4))
    
    optim_results = []
    GS_Energy_list = []
    gap_list = []
    h_range = np.arange(0,1.1,0.1)
    
    
    max_iter = 2000
    for i in range(11):
        h = i*0.1
        GS_energy_h = GS_energy(h)
        GS_Energy_list.append(GS_energy_h)
        print(f'Ground state energy for h={h} is : {GS_Energy_list[i]}')
        
        loaded_params = np.load(f'./LOCC-VQE_RSC_results/LOCC-VQE_RSC_depth_4_seed={int(rand_seed)}_iter={int(max_iter)}_Z_longstringX4/params_Z/Z_h={h}.npy')
        energy, qmi = batched_train_step_tf(batch,h,rand_seed,max_iter,random_idx=i, load_param=True,input_param=loaded_params)
        print(qmi)
        pdb.set_trace()
        