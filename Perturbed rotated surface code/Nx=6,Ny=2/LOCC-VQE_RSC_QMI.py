import tensorcircuit as tc
import tensorflow as tf
import tensornetwork as tn
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
import qiskit.quantum_info


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


def correlation(c: tc.Circuit):
    # correlation = tf.cast(tf.constant([0.0]), tf.float64)
    O_1O_2_Z = tf.cast(c.expectation_ps(z=[0, 5, 6, 11]), tf.float64)
    O_1_Z = tf.cast(c.expectation_ps(z=[0, 6]), tf.float64)
    O_2_Z = tf.cast(c.expectation_ps(z=[5, 11]), tf.float64)
    correlationa_value_Z = O_1O_2_Z - O_1_Z * O_2_Z
    
    O_1O_2_X = tf.cast(c.expectation_ps(x=[0, 5, 6, 11]), tf.float64)
    O_1_X = tf.cast(c.expectation_ps(x=[0, 6]), tf.float64)
    O_2_X = tf.cast(c.expectation_ps(x=[5, 11]), tf.float64)
    correlationa_value_X = O_1O_2_X - O_1_X * O_2_X
    return [O_1O_2_Z,O_1_Z,O_2_Z], tc.backend.real(correlationa_value_X)

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
        tf.reshape(param[14*6+7*12*2:14*6+7*12*2+10*12*2],(10,12,2)), tc.dtypestr
    )
    paramc_3 = tc.backend.cast(
        tf.reshape(param[14*6+7*12*2+10*12*2:14*6+7*12*2+10*12*2+12*3],(12,3)), tc.dtypestr
    )

    circuit.append(toric_syndrome_circuit(paramc_Z))
    circuit.append(correction_circuit(paramc_1,paramc_2,paramc_3))
    qmi = QMI(circuit)
    energy = Hamiltonian(circuit, h)
    return energy, qmi

vqe_vmap = tc.backend.jit(
    tc.backend.vmap(vqe, vectorized_argnums = (0,)), static_argnums=(1,2,3,)
    )



def batched_train_step_tf(batch, h, input_param):    
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

    for i in range(12):
        Pert = 'I'*(i)+'Z'+(12-i)*'I'
        pauli_list.append((Pert, -h))
        
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
    eigenvalues, eigenvectors = np.linalg.eig(H_m)
    sorted_eigenvalues = sorted(eigenvalues)
    return np.real(sorted_eigenvalues[0])




def GS_QMI(h,k):
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
        Pert = 'I'*(i)+'X'+(11-i)*'I'
        pauli_list.append((Pert, -h))
        
    H = SparsePauliOp.from_list(pauli_list)
    H_m = H.to_matrix(sparse=False)
  
    eigenvalues, eigenvectors = np.linalg.eigh(H_m)
    GS_state = eigenvectors[:,k] # we need column vector
    state = tc.quantum.QuOperator.from_tensor(GS_state)
    
    SL = tc.quantum.entanglement_entropy(state, cut=[1,2,3,4,5,7,8,9,10,11]).numpy()
    SR = tc.quantum.entanglement_entropy(state, cut=[0,1,2,3,4,6,7,8,9,10]).numpy()
    SRL = tc.quantum.entanglement_entropy(state, cut=[1,2,3,4,7,8,9,10]).numpy()
    
    qmi = SL+SR-SRL
    pdb.set_trace()
    return qmi

if __name__=="__main__":
    batch = 16
    rand_seed = 2232119
    np.random.seed(rand_seed)
    random_seed_array = np.random.randint(1,100000,(11,4))
    
    optim_results = []
    GS_Energy_list = []
    gap_list = []
    h_range = np.arange(0,1.1,0.1)
    
        
    GS_QMI_list = []
    QMI_list = []
    energy_list = []
    min_energy_idx_list = []
    min_error_list = []
    
    max_iter = 2000
    
    for i in range(11):
        h = i*0.1
        param_path = f'./LOCC-VQE_RSC_results/LOCC-VQE_RSC_depth_4_seed=2232119_iter=500/params_Z/opt_params/Z_h={h:.1f}.npy'

    for k in range(2):
        GS_QMI_list.append([])
        
        for i in range(11):
            h = i*0.1
            
            qmi_GS = GS_QMI(h,k)
            GS_QMI_list[k].append(qmi_GS)
            print(qmi_GS)
            
            
    plt.figure()
    plt.plot(h_range,GS_QMI_list[0],label='GS 0')
    plt.plot(h_range,GS_QMI_list[1],label='GS 1')

    
            
    pdb.set_trace()
    np.save('./qmi_GS_single.npy',np.array(GS_QMI_list))
    np.save('./QMI_list_single.npy',np.array(QMI_list))
    np.save('./energy_list_single.npy',np.array(energy_list))