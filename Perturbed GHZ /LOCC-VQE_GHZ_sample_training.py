# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorcircuit as tc
import tensorflow as tf
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
import copy

K = tc.set_backend("tensorflow")
ctype, rtype = tc.set_dtype("complex128")
EPSILON = 1e-10

# # dynamical GPU memory allocation
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config = config)
# tf.compat.v1.keras.backend.set_session(sess) # The name tf.keras.backend.set_session is deprecated. Please use tf.compat.v1.keras.backend.set_session instead.


def ZZ_measurement(theta_1):
    ZZ_circuit = tc.Circuit(3)
    theta_1_1 = theta_1[0:9]
    theta_1_2 = theta_1[9:18]
    
    ZZ_circuit.ry(0,theta=np.pi/2)
    ZZ_circuit.rxx(0,2,theta=theta_1_1[0])
    ZZ_circuit.rx(0,theta=theta_1_1[1])
    ZZ_circuit.rx(2,theta=theta_1_1[2])
    ZZ_circuit.ryy(0,2,theta = theta_1_1[3])
    ZZ_circuit.ry(0,theta=theta_1_1[4])
    ZZ_circuit.ry(2,theta=theta_1_1[5])
    ZZ_circuit.rzz(0,2,theta=theta_1_1[6])
    ZZ_circuit.rz(0,theta=theta_1_1[7])
    ZZ_circuit.rz(2,theta=theta_1_1[8])
    ZZ_circuit.ry(0,theta=-np.pi/2)
    ZZ_circuit.barrier_instruction()
    
    ZZ_circuit.ry(1,theta=np.pi/2)
    ZZ_circuit.rxx(1,2,theta=theta_1_2[0])
    ZZ_circuit.rx(1,theta=theta_1_2[1])
    ZZ_circuit.rx(2,theta=theta_1_2[2])
    ZZ_circuit.ryy(1,2,theta = theta_1_2[3])
    ZZ_circuit.ry(1,theta=theta_1_2[4])
    ZZ_circuit.ry(2,theta=theta_1_2[5])
    ZZ_circuit.rzz(1,2,theta=theta_1_2[6])
    ZZ_circuit.rz(1,theta=theta_1_2[7])
    ZZ_circuit.rz(2,theta=theta_1_2[8]) 
    ZZ_circuit.ry(1,theta=-np.pi/2)
    ZZ_circuit.barrier_instruction()
    return ZZ_circuit


def syndrome_circuit(n, syn_params): #syn_param shape [n-1][18]
    circuit = tc.Circuit(2*n)
    for i in range(n):
        circuit.h(i)
    for i in range(n-1):
        circuit.append(ZZ_measurement(syn_params[i]),[i,i+1,n+i+1])
    return circuit

def correction_circuit(n, corr_params): # corr_params shape [n][3]
    corr_circuit = tc.Circuit(2*n) #make it 2*n is just for simplicity for append (without need to specify indices which need to get numpy from tf)
    for i in range(n):

        corr_circuit.rx(i, theta=corr_params[i][0]+corr_params[i][1])
        corr_circuit.rz(i, theta=corr_params[i][3])
        corr_circuit.ry(i, theta=corr_params[i][4])
        corr_circuit.rz(i, theta=corr_params[i][5])
    return corr_circuit

def U1(n, theta_1): # theta_1 shape [(n-1)*18] 
    theta_1 = tf.reshape(theta_1,[n-1,18]) #use tf functions!
    U1_circuit = syndrome_circuit(n, theta_1)
    return U1_circuit

def U2(n, theta_2): # theta_2 = g shape[n*3] 
    theta_2 = tf.reshape(theta_2, [n,6])
    U2_circuit = correction_circuit(n, theta_2)
    return U2_circuit

def g_func_batch_wo_grad(gamma, projector_v, CNOT_Ops, masks, n):
    input_sign = 2 * (projector_v - 0.5)
    output_sign = tf.concat([tf.ones([n,1], dtype=tf.float32),input_sign[:,tf.newaxis],input_sign[:,tf.newaxis],tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32)],axis=1) # if 0, e^(-jtheta[2]/2) rx(theta[0]+theta[1]) if 0, e^(jtheta[2]/2) rx(theta[0]-theta[1])

    layer_1_weights = gamma[0]
    layer_1_bias = gamma[1]
    layer_2_weights = gamma[2]
    layer_2_bias = gamma[3]
    layer_1_out = tf.matmul(layer_1_weights, tf.cast(input_sign[:, tf.newaxis], tf.float32))+ layer_1_bias
    layer_1_out_act = tf.tanh(layer_1_out)
    layer_2_out = tf.matmul(layer_2_weights, layer_1_out_act) + layer_2_bias
    layer_2_out_act = tf.sigmoid(layer_2_out)
    output = tf.squeeze(layer_2_out_act, axis=1) * 0.2#TODO: we can multiply this with a scalar, giving weights for output getting from different methods
    
    ones = tf.ones([2*n,1], dtype=tf.float32)
    input_encoder = tf.constant([-1,1], dtype=tf.float32)
    input_en = (ones + tf.cast(tf.reshape(input_encoder * input_sign[:, tf.newaxis], [2*n,1]), tf.float32))/2
    theta = tf.reshape(tf.matmul(gamma[4], input_en), [n,6])
    theta_out = tf.reshape(output_sign*masks[0]*theta, [6*n])
    output = output + theta_out
    
    # n == 8, l == 3
    input_1 = tf.math.mod(tf.matmul(CNOT_Ops[0], tf.cast(projector_v[:,tf.newaxis], tf.float32)), tf.constant([2], dtype=tf.float32))
    input_sign = 2 * (input_1 - 0.5)
    output_sign = tf.concat([tf.ones([n,1], dtype=tf.float32),input_sign,input_sign,tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32)],axis=1) # if 0, e^(-jtheta[2]/2) rx(theta[0]+theta[1]) if 0, e^(jtheta[2]/2) rx(theta[0]-theta[1])
    input_encoder = tf.constant([-1,1], dtype=tf.float32)
    input_en = (ones + tf.cast(tf.reshape(input_encoder * input_sign[:, tf.newaxis], [2*n,1]), tf.float32))/2
    theta = tf.reshape(tf.matmul(gamma[5], input_en),[n,6])
    theta_out = output_sign * masks[1] * theta # since mask[0] used before loop, start from 1 here
    theta_out = tf.reshape(theta_out, [6*n])
    output = output + theta_out
    
    
    input_1 = tf.math.mod(tf.matmul(CNOT_Ops[1], tf.cast(projector_v[:,tf.newaxis], tf.float32)), tf.constant([2], dtype=tf.float32))
    input_sign = 2 * (input_1 - 0.5)
    output_sign = tf.concat([tf.ones([n,1], dtype=tf.float32),input_sign,input_sign,tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32)],axis=1) # if 0, e^(-jtheta[2]/2) rx(theta[0]+theta[1]) if 0, e^(jtheta[2]/2) rx(theta[0]-theta[1])
    input_encoder = tf.constant([-1,1], dtype=tf.float32)
    input_en = (ones + tf.cast(tf.reshape(input_encoder * input_sign[:, tf.newaxis], [2*n,1]), tf.float32))/2
    theta = tf.reshape(tf.matmul(gamma[6], input_en),[n,6])
    theta_out = output_sign * masks[2] * theta # since mask[0] used before loop, start from 1 here
    theta_out = tf.reshape(theta_out, [6*n])
    output = output + theta_out
        
    input_sign = 2 * (projector_v - 0.5)
    output_sign = tf.concat([tf.ones([n,1], dtype=tf.float32),input_sign[:,tf.newaxis],input_sign[:,tf.newaxis],tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32)],axis=1) # if 0, e^(-jtheta[2]/2) rx(theta[0]+theta[1]) if 0, e^(jtheta[2]/2) rx(theta[0]-theta[1])
    theta = tf.matmul(gamma[7], input_sign[:, tf.newaxis])
    theta_out = tf.reshape(output_sign*tf.reshape(theta, [n,6]), [6*n])
    output = output + theta_out
    return output
g_func_batch_wo_grad_vmap = tc.backend.vmap(g_func_batch_wo_grad, vectorized_argnums=(0,1,))

def g_func_rounds(gamma, projector_v, CNOT_Ops, masks, n):
    assert projector_v.shape[0]==n
    with tf.GradientTape() as tape:
        tape.watch(gamma)
        input_sign = 2 * (projector_v - 0.5)
        output_sign = tf.concat([tf.ones([n,1], dtype=tf.float32),input_sign[:,tf.newaxis],input_sign[:,tf.newaxis],tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32)],axis=1) # if 0, e^(-jtheta[2]/2) rx(theta[0]+theta[1]) if 0, e^(jtheta[2]/2) rx(theta[0]-theta[1])

        layer_1_weights = gamma[0]
        layer_1_bias = gamma[1]
        layer_2_weights = gamma[2]
        layer_2_bias = gamma[3]
        layer_1_out = tf.matmul(layer_1_weights, tf.cast(input_sign[:, tf.newaxis], tf.float32))+ layer_1_bias
        layer_1_out_act = tf.tanh(layer_1_out)
        layer_2_out = tf.matmul(layer_2_weights, layer_1_out_act) + layer_2_bias
        layer_2_out_act = tf.sigmoid(layer_2_out)
        output = tf.squeeze(layer_2_out_act, axis=1) * 0.2#TODO: we can multiply this with a scalar, giving weights for output getting from different methods
        
        # l = int(tf.math.log(tf.cast(n, tf.float32))/tf.math.log(2.0))
        ones = tf.ones([2*n,1], dtype=tf.float32)
        input_encoder = tf.constant([-1,1], dtype=tf.float32)
        input_en = (ones + tf.cast(tf.reshape(input_encoder * input_sign[:, tf.newaxis], [2*n,1]), tf.float32))/2
        theta = tf.reshape(tf.matmul(gamma[4], input_en), [n,6])
        theta_out = tf.reshape(output_sign*masks[0]*theta, [6*n])
        output = output + theta_out
        
        # n == 8, l == 3
        input_1 = tf.math.mod(tf.matmul(CNOT_Ops[0], tf.cast(projector_v[:,tf.newaxis], tf.float32)), tf.constant([2], dtype=tf.float32))
        input_sign = 2 * (input_1 - 0.5)
        output_sign = tf.concat([tf.ones([n,1], dtype=tf.float32),input_sign,input_sign,tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32)],axis=1) # if 0, e^(-jtheta[2]/2) rx(theta[0]+theta[1]) if 0, e^(jtheta[2]/2) rx(theta[0]-theta[1])
        input_encoder = tf.constant([-1,1], dtype=tf.float32)
        input_en = (ones + tf.cast(tf.reshape(input_encoder * input_sign[:, tf.newaxis], [2*n,1]), tf.float32))/2
        theta = tf.reshape(tf.matmul(gamma[5], input_en),[n,6])
        theta_out = output_sign * masks[1] * theta # since mask[0] used before loop, start from 1 here
        theta_out = tf.reshape(theta_out, [6*n])
        output = output + theta_out
        
        
        input_1 = tf.math.mod(tf.matmul(CNOT_Ops[1], tf.cast(projector_v[:,tf.newaxis], tf.float32)), tf.constant([2], dtype=tf.float32))
        input_sign = 2 * (input_1 - 0.5)
        output_sign = tf.concat([tf.ones([n,1], dtype=tf.float32),input_sign,input_sign,tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32)],axis=1) # if 0, e^(-jtheta[2]/2) rx(theta[0]+theta[1]) if 0, e^(jtheta[2]/2) rx(theta[0]-theta[1])
        input_encoder = tf.constant([-1,1], dtype=tf.float32)
        input_en = (ones + tf.cast(tf.reshape(input_encoder * input_sign[:, tf.newaxis], [2*n,1]), tf.float32))/2
        theta = tf.reshape(tf.matmul(gamma[6], input_en),[n,6])
        theta_out = output_sign * masks[2] * theta # since mask[0] used before loop, start from 1 here
        theta_out = tf.reshape(theta_out, [6*n])
        output = output + theta_out
            
        input_sign = 2 * (projector_v - 0.5)
        output_sign = tf.concat([tf.ones([n,1], dtype=tf.float32),input_sign[:,tf.newaxis],input_sign[:,tf.newaxis],tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32),tf.ones([n,1], dtype=tf.float32)],axis=1) # if 0, e^(-jtheta[2]/2) rx(theta[0]+theta[1]) if 0, e^(jtheta[2]/2) rx(theta[0]-theta[1])
        theta = tf.matmul(gamma[7], input_sign[:, tf.newaxis])
        theta_out = tf.reshape(output_sign*tf.reshape(theta, [n,6]), [6*n])
        output = output + theta_out
        
        gamma_gfunc_grad_tensor_list = tape.jacobian(output, gamma) # squeeze for output theta_2_num, 1, gamma_shape --> theta_2_num, gamma_shape TODO:check gradient
    return output, gamma_gfunc_grad_tensor_list
g_func_rounds_vmap = tc.backend.vmap(g_func_rounds, vectorized_argnums=(1,)) # TODO: really don't understand why can't jit this function


def g_func_batched_round(gamma, projector_batched_round, CNOT_Ops, masks, n): # gamma is a list of tensors
    theta_2_round, gamma_gfunc_grad_tensor_list_round = g_func_rounds_vmap(gamma, projector_batched_round, CNOT_Ops, masks, n)
    return theta_2_round, gamma_gfunc_grad_tensor_list_round
g_func_batched_round_vmap = tc.backend.jit(tc.backend.vmap(g_func_batched_round, vectorized_argnums=(0,1,)))


def Hamiltonian(c: tc.Circuit, n: int, kx: float, h: float = 0):
    # c = c.copy()
    e = 0.0
    for i in range(0,n-1,2):
        e += (1-h)/n * c.expectation_ps(z=[i, i+1])
    for i in range(1,n-2,2):
        e += (1-h)/n * c.expectation_ps(z=[i, i+1])
    e += (kx-h)/n * c.expectation_ps(x=list(range(n)))
    for i in range(n):
        e += h/n * c.expectation_ps(x=[i])  
    return -tc.backend.real(e)


def get_prob(n, theta_1, projector_onehot, prob_idx):
    # projector: 2 for id, 0 for 0_projector, 1 for 1_projector
    # since logical operations are unfriendly for jit, use one hot in replacement of 'if'
    theta_1 = tf.cast(theta_1, ctype,)
    circuit = U1(n, theta_1)
    projector_onehot = tf.cast(projector_onehot, ctype,)
    projector_set = tf.cast(tf.constant([[[1., 0.],[0., 0.]], [[0., 0.], [0., 1.]], [[1., 0.], [0., 1.]]]), ctype,)
    for idx in range(n):
        projector_unitary = tf.reduce_sum(projector_set * projector_onehot[idx][:, tf.newaxis, tf.newaxis], axis=-3 )# batch dimension of projector_onehot should be handled by vmap
        circuit.any(n+idx, unitary = projector_unitary)
    prob_0_uncond = circuit.expectation_ps(z=[n+prob_idx])
    return tc.backend.real(prob_0_uncond)
get_prob_vmap = tc.backend.vmap(get_prob, vectorized_argnums=(1, 2,))


def adaptive_vqe(n, theta_1, theta_2, projector_onehot, kx, h):
    theta_1 = tf.cast(theta_1, ctype,)
    circuit = U1(n, theta_1)
    theta_2 = tf.cast(theta_2, ctype,) # since theta_2 is from classical function, its type is float32, but complex128 is needed for quantum circuit construction
    projector_onehot = tf.cast(projector_onehot, ctype,)
    projector_set = tf.cast(tf.constant([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]], [[1., 0.], [0., 1.]]]), ctype,)
    for idx in range(n):
        projector_unitary = tf.reduce_sum(projector_set * projector_onehot[idx][:, tf.newaxis, tf.newaxis], axis=-3)# batch 
        circuit.any(n+idx, unitary = projector_unitary)
    circuit.append(U2(n, theta_2))
    energy = Hamiltonian(circuit, n, kx, h)
    return energy
adaptive_vqe_vmap = tc.backend.jit(tc.backend.vmap(adaptive_vqe, vectorized_argnums=(1, 2, 3,)))


def sample(n, batch_size, theta_1_batched):
    prob_batched = tf.zeros([batch_size, n, 2])
    projector_batched = 2 * tf.ones([batch_size, n]) # 2 for Identity
    cond_prob_batched = tf.ones(batch_size)
    for idx in range(n):
        if(idx > 0):
            logits = tf.math.log(prob_batched[:,idx-1])
            projector_batched = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(projector_batched), indices=tf.constant([[idx-1]]), updates=tf.cast(tf.reshape(tf.random.categorical(logits, 1),(1, batch_size)), dtype=tf.float32)))
            cond_prob_batched = tf.reduce_sum(tf.reshape(cond_prob_batched, shape=(batch_size, 1)) * (prob_batched[:,idx-1] * tf.one_hot(tf.cast(projector_batched[:, idx-1], dtype=tf.int64), 2, dtype=tf.float32)), axis=-1)
            projector_batched = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(projector_batched), indices=tf.constant([[idx]]), updates=tf.cast(tf.zeros(shape=(1, batch_size)), dtype=tf.float32)))
            
        projector_batched_onehot = tf.one_hot(tf.cast(projector_batched, dtype=tf.int64), 3)
        prob_0_uncond = get_prob_vmap(n, theta_1_batched, projector_batched_onehot, idx)
        prob_0 = tf.cast(prob_0_uncond, tf.float32)/cond_prob_batched
        prob_batched = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(prob_batched, perm=[1,2,0]), indices=tf.constant([[[idx, 0], [idx, 1]]]), updates=tf.concat([prob_0[tf.newaxis, :], (1 - prob_0)[tf.newaxis,:]], axis=0)[tf.newaxis, :]), perm=[2,0,1])
    logits = tf.math.log(prob_batched[:,-1])
    projector_batched = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(projector_batched), indices=tf.constant([[n-1]]), updates=tf.cast(tf.reshape(tf.random.categorical(logits, 1),(1, batch_size)), dtype=tf.float32)))
    return projector_batched


sample_vmap = tc.backend.jit(tc.backend.vmap(sample, vectorized_argnums=(2,))) #vectorize the round dime
    
    
def grad_theta_1_paramshift_sample(n, theta_1_batched_round, gamma_batched,  CNOT_Ops, masks, batch_size, sample_round, kx, h):
    # theta_1_batched_round list of tensors(theta_1_batched) same with theta_2 # need parallel sampling
    # theta_1_batched_round shape: sample_round, batch_size, theta_2_num
    gamma_batched_tiled = []
    for i in range(len(gamma_batched)):
        gamma_batched_tiled.append(tf.tile(gamma_batched[i], [int(18*(n-1)),1,1])) # theta_1_num * batch_size, input_feature, output_feature
        
    theta_1_batched_r_b = tf.cast(tf.tile(theta_1_batched_round[:, :, tf.newaxis, :], [1, 1, int(18*(n-1)), 1]), tf.float32) # shape sample_round, batch_size, param_theta_1_num, param_theta_1_num
    shift_tensor_one_batch = np.pi/2 * tf.eye(int(18*(n-1)))
    shift_tensor = tf.cast(tf.tile(shift_tensor_one_batch[tf.newaxis, tf.newaxis, :, :], [sample_round, batch_size, 1, 1]), tf.float32)
    theta_1_pos_batched_tf = theta_1_batched_r_b + shift_tensor
    theta_1_pos_batched_tf = tf.transpose(theta_1_pos_batched_tf, perm=[0,2,1,3]) # round, theta_1_num, batch, num_theta_1(in each batch, shift same theta_idx)
    theta_1_pos_batched_tf_r_theta1_b_theta1 = tf.transpose(theta_1_pos_batched_tf, perm=[1,0,2,3]) # theta_1_num, round, batch, num_theta_1 (for later adaptive_vqe_vmap use)
    theta_1_pos_batched_tf_r_theta1_b_theta1 = tf.reshape(theta_1_pos_batched_tf_r_theta1_b_theta1, [int(18*(n-1) * sample_round*batch_size), int(18*(n-1))]) # theta_1_num*round*batch, num_theta_1 (for later adaptive_vqe_vmap use)
    theta_1_pos_batched_tf = tf.reshape(theta_1_pos_batched_tf, [int(sample_round * 18*(n-1)), batch_size, int(18*(n-1))]) # round*theta_1_num, batch_size, theta_1_num (though same shape with above, but are different! see the order of multiplication for the first dimension)
    projector_pos = sample_vmap(n, batch_size, theta_1_pos_batched_tf) # round*theta_1_num, batch_size, v
    projector_pos = tf.reshape(projector_pos, [sample_round, int(18*(n-1)), batch_size, n]) # round, theta_1_num, batch_size, v
    projector_pos = tf.transpose(projector_pos, [1,0,2,3]) # theta_1_num, sample_round, batch_size, v
    projector_pos_onehot = tf.one_hot(tf.cast(projector_pos, tf.int64), 3) # theta_1_num, sample_round, batch_size, v, 3
    projector_pos_onehot = tf.reshape(projector_pos_onehot, [int(18*(n-1) * sample_round*batch_size), n, 3]) # theta_1_num*sample_round*batch_size, v, 3
    projector_pos = tf.transpose(projector_pos, [0,2,1,3]) # theta_1_num, batch_size, sample_round, v
    projector_pos = tf.reshape(projector_pos, [int(18*(n-1) * batch_size), sample_round, n]) # theta_1_num*batch_size, sample_round, v
    theta_2_pos1_batched, _ = g_func_batched_round_vmap(gamma_batched_tiled, projector_pos, CNOT_Ops, masks, n) # theta_1_num*batch_size, sample_round, theta_2_num
    theta_2_pos1_b_r = tf.reshape(theta_2_pos1_batched, [int(18*(n-1)), batch_size, sample_round, int(6*n)]) # theta_1_num, batch_size, sample_round, theta_2_num
    theta_2_pos1_r_b = tf.transpose(theta_2_pos1_b_r, [0,2,1,3]) # theta_1_num, sample_round, batch_size, theta_2_num
    theta_2_pos1_r_b = tf.reshape(theta_2_pos1_r_b, [int(18*(n-1) * sample_round*batch_size), int(6*n)]) # theta_1_num*sample_round*batch_size, theta_2_num
    energy_pos = adaptive_vqe_vmap(n, theta_1_pos_batched_tf_r_theta1_b_theta1, theta_2_pos1_r_b, projector_pos_onehot, kx, h) # theta_1_num*sample_round*batch_size
    
    
    theta_1_neg_batched_tf = theta_1_batched_r_b - shift_tensor
    theta_1_neg_batched_tf = tf.transpose(theta_1_neg_batched_tf, perm=[0,2,1,3]) # round, theta_1_num, batch, num_theta_1(in each batch, shift same theta_idx)
    theta_1_neg_batched_tf_r_theta1_b_theta1 = tf.transpose(theta_1_neg_batched_tf, perm=[1,0,2,3]) # theta_1_num, round, batch, num_theta_1 (for later adaptive_vqe_vmap use)
    theta_1_neg_batched_tf_r_theta1_b_theta1 = tf.reshape(theta_1_neg_batched_tf_r_theta1_b_theta1, [int(18*(n-1) * sample_round*batch_size), int(18*(n-1))]) # theta_1_num*round*batch, num_theta_1 (for later adaptive_vqe_vmap use)
    theta_1_neg_batched_tf = tf.reshape(theta_1_neg_batched_tf, [int(sample_round * 18*(n-1)), batch_size, int(18*(n-1))]) # round*theta_1_num, batch_size, theta_1_num (though same shape with above, but are different! see the order of multiplication for the first dimension)
    projector_neg = sample_vmap(n, batch_size, theta_1_neg_batched_tf) # round*theta_1_num, batch_size, v
    projector_neg = tf.reshape(projector_neg, [sample_round, int(18*(n-1)), batch_size, n]) # round, theta_1_num, batch_size, v
    projector_neg = tf.transpose(projector_neg, [1,0,2,3]) # theta_1_num, sample_round, batch_size, v
    projector_neg_onehot = tf.one_hot(tf.cast(projector_neg, tf.int64), 3) # theta_1_num, sample_round, batch_size, v, 3
    projector_neg_onehot = tf.reshape(projector_neg_onehot, [int(18*(n-1) * sample_round*batch_size), n, 3]) # theta_1_num*sample_round*batch_size, v, 3
    projector_neg = tf.transpose(projector_neg, [0,2,1,3]) # theta_1_num, batch_size, sample_round, v
    projector_neg = tf.reshape(projector_neg, [int(18*(n-1) * batch_size), sample_round, n]) # theta_1_num*batch_size, sample_round, v
    theta_2_neg1_batched, _ = g_func_batched_round_vmap(gamma_batched_tiled, projector_neg, CNOT_Ops, masks, n) # theta_1_num*batch_size, sample_round, theta_2_num
    theta_2_neg1_b_r = tf.reshape(theta_2_neg1_batched, [int(18*(n-1)), batch_size, sample_round, int(6*n)]) # theta_1_num, batch_size, sample_round, theta_2_num
    theta_2_neg1_r_b = tf.transpose(theta_2_neg1_b_r, [0,2,1,3]) # theta_1_num, sample_round, batch_size, theta_2_num
    theta_2_neg1_r_b = tf.reshape(theta_2_neg1_r_b, [int(18*(n-1) * sample_round*batch_size), int(6*n)]) # theta_1_num*sample_round*batch_size, theta_2_num
    energy_neg = adaptive_vqe_vmap(n, theta_1_neg_batched_tf_r_theta1_b_theta1, theta_2_neg1_r_b, projector_neg_onehot, kx, h) # theta_1_num*sample_round*batch_size
    
    grad_theta_1_round = 0.5 * (energy_pos - energy_neg) # theta_1_num*sample_round*batch_size
    grad_theta_1_round = tf.cast(tf.reshape(grad_theta_1_round, [int(18*(n-1)), sample_round, batch_size]), tf.float32) # theta_1_num, sample_round, batch_size
    grad_theta_1 = tf.reduce_mean(grad_theta_1_round, axis=1) # theta_1_num, batch_size
    grad_theta_1 = tf.transpose(grad_theta_1, perm=[1,0]) # batch_size, theta_1_num
    return grad_theta_1
grad_theta_1_paramshift_sample_jit = tc.backend.jit(grad_theta_1_paramshift_sample)
    
    
def grad_theta_2_paramshift(n, theta_1_batched, theta_2_batched_round, projector_v_onehot_round, batch_size, sample_round, kx, h):
    # projector_v_onehot_round_tf # sample_round, batch_size, v, 3
    # theta_2_batched_round sample_round, batch_size, theta_2_num
    projector_v_onehot_round_tf = tf.tile(projector_v_onehot_round[tf.newaxis, :, :, :, :], [int(6*n), 1, 1, 1, 1]) # theta_2_num, sample_round, batch_size, v, 3
    projector_v_onehot_round_tf = tf.transpose(projector_v_onehot_round_tf, [1,0,2,3,4]) # sample_round, theta_2_num, batch_size, v, 3
    projector_v_onehot_tf = tf.reshape(projector_v_onehot_round_tf, [int(sample_round * 6*n * batch_size), n, 3])
    
    theta_1_batched_tf = tf.tile(theta_1_batched[tf.newaxis, tf.newaxis, :], [sample_round, int(6*n), 1, 1]) # sample_round, theta_2_num, batch_size, theta_1_num
    theta_1_batched_tf = tf.reshape(theta_1_batched_tf, [int(sample_round * 6*n * batch_size), int(18*(n-1))])
    
    # theta_2_batched_round shape: sample_round, batch_size, theta_2_num
    theta_2_batched_r_b = tf.cast(tf.tile(theta_2_batched_round[:, :, tf.newaxis, :], [1, 1, int(6*n), 1]), tf.float32) # shape sample_round, batch_size, param_theta_2_num, param_theta_2_num
    shift_tensor_one_batch = np.pi/2 * tf.eye(int(6*n))
    shift_tensor = tf.cast(tf.tile(shift_tensor_one_batch[tf.newaxis, tf.newaxis, :, :], [sample_round, batch_size, 1, 1]), tf.float32)
    theta_2_pos_batched_tf = theta_2_batched_r_b + shift_tensor
    theta_2_pos_batched_tf = tf.transpose(theta_2_pos_batched_tf, perm=[0,2,1,3]) # round, theta_2_num, batch, num_theta_2(in each batch, shift same theta_idx)
    theta_2_pos_batched_tf = tf.reshape(theta_2_pos_batched_tf, [int(sample_round * 6*n * batch_size), int(6*n)])
    energy_pos = adaptive_vqe_vmap(n, theta_1_batched_tf, theta_2_pos_batched_tf, projector_v_onehot_tf, kx, h)
    
    theta_2_neg_batched_tf = theta_2_batched_r_b - shift_tensor
    theta_2_neg_batched_tf = tf.transpose(theta_2_neg_batched_tf, perm=[0,2,1,3]) # round, theta_2_num, batch, num_theta_2(in each batch, shift same theta_idx)
    theta_2_neg_batched_tf = tf.reshape(theta_2_neg_batched_tf, [int(sample_round * 6*n * batch_size), int(6*n)])
    energy_neg = adaptive_vqe_vmap(n, theta_1_batched_tf, theta_2_neg_batched_tf, projector_v_onehot_tf, kx, h)
    
    theta_2_grad = 0.5 * (energy_pos - energy_neg) # sample_round*theta_2_num*batch_size
    theta_2_grad = tf.reshape(theta_2_grad, [sample_round, int(6*n), batch_size]) # sample_round, theta_2_num, batch_size
    return theta_2_grad
grad_theta_2_paramshift_jit = tc.backend.jit(grad_theta_2_paramshift)
    
    
    

def train(n: int, init_theta_1_batched: tf.Variable, init_gamma_batched: tf.Variable, CNOT_Ops: list, masks: list,  batch_size: int, kx: float, h: float,  optimizer: tf.keras.optimizers, sample_round: int, max_iter: int, record: int):
    theta_1_batched = init_theta_1_batched # should be tf.Variable
    gamma_batched = init_gamma_batched # should be tf.Variable
    energy_batched_record = []
    energy_min_record = []
    theta_1_batched_record = []
    gamma_batched_record = []
    for iter in range(max_iter):
        start = time.time()
        theta_1_batched_round = tf.tile(theta_1_batched[tf.newaxis, :, :], [sample_round, 1, 1]) # sample_round, batch, theta_1_num
        projector_v_batched_round_r_b = sample_vmap(n, batch_size, theta_1_batched_round) # sample_round, batch, v
        projector_v_batched_round_b_r = tf.transpose(projector_v_batched_round_r_b, [1,0,2])
        projector_v_onehot_batched_round_tf = tf.one_hot(tf.cast(projector_v_batched_round_r_b, tf.int64), 3) # sample_round, batch, v, 3 ### remeber delete stack operation in grad_theta_2
        theta_2_batched_b_r, gamma_gfunc_grad_batched_list = g_func_batched_round_vmap(gamma_batched, projector_v_batched_round_b_r, CNOT_Ops, masks, n)# gamma_gfunc_grad_batched should be list of tensors, each tensor: batch_size, round, theta_2_num, layer_input_feature, layer_output_feature
        theta_2_batched_r_b = tf.transpose(theta_2_batched_b_r, perm=[1,0,2])
        grad_theta_1_batched = grad_theta_1_paramshift_sample_jit(n, theta_1_batched_round, gamma_batched,  CNOT_Ops, masks, batch_size, sample_round, kx, h)
        grad_theta_2_round_batched = grad_theta_2_paramshift_jit(n, theta_1_batched, theta_2_batched_r_b, projector_v_onehot_batched_round_tf, batch_size, sample_round, kx, h)
        # shape round, batch, theta_2_num
        
        grad_gamma_batched = []
        for i in range(len(gamma_gfunc_grad_batched_list)):
            gamma_gfunc_grad_batched_list_i = tf.transpose(gamma_gfunc_grad_batched_list[i], perm=[1,0,2,3,4]) #  round, batch_size, theta_2_num, layer_input_feature, layer_output_feature
            gamma_gfunc_grad_batched_list_i = tf.transpose(gamma_gfunc_grad_batched_list_i, perm=[0,2,1,3,4]) #  round, theta_2_num, batch_size, layer_input_feature, layer_output_feature
            grad_gamma_i_batched_round = tf.reduce_sum(gamma_gfunc_grad_batched_list_i * tf.cast(grad_theta_2_round_batched[:, :, :, tf.newaxis, tf.newaxis], tf.float32), axis=1) # round, batch_size, layer_input_feature, layer_output_feature
            grad_gamma_i_batched = tf.cast(tf.reduce_mean(grad_gamma_i_batched_round, axis=0), tf.float32)
            grad_gamma_batched.append(grad_gamma_i_batched)
            
        if (iter%record)==0:
            theta_1_batched_before_updata = tf.constant(theta_1_batched)
            theta_2_batched = g_func_batch_wo_grad_vmap(gamma_batched, projector_v_batched_round_r_b[0,:], CNOT_Ops, masks, n)
            
        optimizer.apply_gradients(zip(grad_gamma_batched, gamma_batched))
        optimizer.apply_gradients(zip([grad_theta_1_batched], [theta_1_batched])) # need to be list of tensor(s)
        
        stop = time.time()
        print(f'--------- finish_iteration_{iter} time_consumption: {stop-start:.3f} ---------')
        if (iter%record)==0:            
            energy_batched = adaptive_vqe_vmap(n, theta_1_batched_before_updata, theta_2_batched, projector_v_onehot_batched_round_tf[0], kx, h)
            energy_batched_record.append(energy_batched)
            theta_1_batched_record.append(theta_1_batched)
            gamma_batched_record.append(gamma_batched)
            energy_min_record.append(np.min(energy_batched.numpy(), axis=0))
            print(f'current_energy(before update):\n {energy_batched}')
            print(f'lowest_energy: {np.min(energy_batched.numpy(), axis=0)}\n')

    return theta_1_batched_record, gamma_batched_record, energy_batched_record, energy_min_record
        

def he_init(shape):
    fan_in = shape[-2]
    stddev = tf.sqrt(2.0/fan_in)
    return tf.random.normal(shape, stddev=stddev)


def init_param(n, m, batch_size):
    l = int(tf.math.log(tf.cast(m, tf.float32))/tf.math.log(2.0))
    init_theta_1 = tf.Variable(initial_value=tf.concat([
        tf.random.normal(shape=(int(batch_size/4),(n-1)*18), mean=0.0, stddev=0.2),
        tf.random.normal(shape=(int(batch_size/4),(n-1)*18), mean=np.pi/2, stddev=0.2),
        tf.random.normal(shape=(int(batch_size/4),(n-1)*18), mean=np.pi, stddev=0.2),
        tf.random.normal(shape=(int(batch_size/4),(n-1)*18), mean=np.pi*3/2, stddev=0.2)
        ], axis=0))
    init_gamma_layer_1_weights = tf.Variable(initial_value=he_init([batch_size, 16*m, m]))
    init_gamma_layer_1_bias = tf.Variable(initial_value=tf.random.normal(shape=[batch_size, 16*m, 1], mean=0, stddev=0.1)) # the first round of g_func has input v = all zero, thus we need non_zero bias to make non-trivial output theta_2
    init_gamma_layer_2_weights = tf.Variable(initial_value=he_init([batch_size, 6*n, 16*m]))
    init_gamma_layer_2_bias = tf.Variable(initial_value=tf.random.normal(shape=[batch_size, 6*n, 1], mean=0, stddev=0.1))
    init_gamma = [init_gamma_layer_1_weights, init_gamma_layer_1_bias, init_gamma_layer_2_weights, init_gamma_layer_2_bias]
    for i in range(l):
        init_gamma_prior = tf.Variable(initial_value=he_init([batch_size, 6*n, 2*m]))
        init_gamma.append(init_gamma_prior)
    init_gamma_prior_dense = tf.Variable(initial_value=he_init([batch_size, 6*n, m]))
    init_gamma.append(init_gamma_prior_dense)
    return init_theta_1, init_gamma 

def get_CNOT_Ops_and_masks(n):
    l = int(tf.math.log(tf.cast(n, tf.float32))/tf.math.log(2.0))
    CNOT_Ops = []
    masks = []
    mask = tf.cast(tf.reshape(tf.concat([tf.zeros([int(n/2),1]),tf.ones([int(n/2),1])],axis=1), [n,1]), tf.float32)
    masks.append(mask)
    for i in range(l-1):
        CNOT_op = tf.eye(n,n,dtype=tf.float32)
        cnot_list = []
        update_cnot = []
        mask_list = []
        update_mask = []
        for j in range(1,n-1,2**(i+1)):
            cnot_list.append([j,j+2**i]) # CNOT j+2**i-->j
            update_cnot.append(1.)
            mask_list.append([j+1])
            update_mask.append(1.)
        CNOT_op = tf.cast(tf.tensor_scatter_nd_update(CNOT_op,cnot_list,update_cnot), tf.float32)
        mask = tf.cast(tf.reshape(tf.tensor_scatter_nd_update(tf.zeros(n), mask_list, update_mask), [n,1]), tf.float32)
        CNOT_Ops.append(CNOT_op)
        masks.append(mask)
    return CNOT_Ops, masks


if __name__ == '__main__':
    n = 8
    m = n
    kx = 16
    h = 0
    batch_size = 4
    sample_round = 40
    max_iter = 400
    record = 1
    log_path = './log/GHZ...'
    init_theta_1_batched, init_gamma_batched = init_param(n, m, batch_size)
    CNOT_Ops, masks = get_CNOT_Ops_and_masks(n)
    initial_learning_rate = 1e-2
    decay_steps = 100
    decay_rate = 0.9
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
        )
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    best_results = []
    
    for i in range(1):
        h = i*0.1
        theta_1_batched_record, gamma_batched_record, energy_batched_record, energy_min_record = train(n, init_theta_1_batched, init_gamma_batched, CNOT_Ops, masks, batch_size, kx, h, optimizer, sample_round, max_iter, record)
        np.save(f'./results/energy_min_record_8_X_h={h}_0.2_neural_new.npy', energy_min_record)
        best_results.append(np.min(energy_min_record))
        plt.figure()
        plt.plot(energy_min_record)
        # plt.show()
        plt.savefig(f'./results/GHZ_8_6_40_X_h={h}_kx={kx}_0.2_neural_new.jpg')
        
    np.save('./results/best_results_8_X_0.2_neural_new.npy', best_results)
    plt.figure()
    plt.plot(0.1*np.arange(11), best_results)
    plt.xlabel('h')
    plt.ylabel('energy')
    plt.savefig('./results/best_results_8_X_new.jpg')
