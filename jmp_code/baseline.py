"""
Disclaimer:
This code is adapted from the implementation provided by Marlon Azinovic, Luca Gaegauf, and Simon Scheidegger (2022).
The supporting files nn_utils.py and utils.py are included verbatim as provided by the authors.

Reference:
Azinovic M, Gaegauf L, Scheidegger S. "Deep equilibrium nets." International Economic Review. 2022 Nov;63(4):1471-1525.
URL: https://onlinelibrary.wiley.com/doi/epdf/10.1111/iere.12575

Description:
This Python code trains neural networks to approximate the policy functions and pricing functions defined in the paper.
The code supports two main functionalities:

1. **Train the neural networks from scratch:**
   To train the networks from scratch, run:
       > python baseline.py --train_from_scratch
   Results will be saved to: `./output/1st_baseline`.

2. **Simulate the economy and reproduce the results:**
   To load the trained neural networks, simulate the economy, and reproduce the graphs and tables presented in the paper, run:
       > python baseline.py
   Results will be saved to: `./output/restart_baseline`.
"""

import pickle
import os
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from datetime import datetime
from utils import random_mini_batches
import codecs
import json
import numpy.matlib

print('tf version:', tf.__version__)

plt.rcParams.update({'font.size': 12})


def train(path_wd, run_name, num_agents,
          num_episodes, len_episodes, epochs_per_episode,
          batch_size, optimizer_name, lr,
          save_interval, num_hidden_nodes, activations_hidden_nodes,
          train_flag=True, load_flag=True, load_run_name=None,
          load_episode=True, seed=1, save_raw_plot_data = False):

    train_dict = {}
    load_dict = {}
    train_setup_dict = {}
    econ_setup_dict = {}
    net_setup_dict = {}
    result_dict = {}
    params_dict = {}

    train_dict['seed'] = seed
    train_dict['identifier'] = run_name

    save_base_path = os.path.join('./output', run_name)
    log_dir = os.path.join(save_base_path, 'tensorboard')
    plot_dir = os.path.join(save_base_path, 'plots')

    train_setup_dict['num_episodes'] = num_episodes
    train_setup_dict['len_episodes'] = len_episodes
    train_setup_dict['epochs_per_episode'] = epochs_per_episode
    train_setup_dict['optimizer'] = optimizer_name
    train_setup_dict['batch_size'] = batch_size
    train_setup_dict['lr'] = lr

    train_dict['train_setup'] = train_setup_dict

    net_setup_dict['num_hidden_nodes'] = num_hidden_nodes
    net_setup_dict['activations_hidden_nodes'] = activations_hidden_nodes

    train_dict['net_setup'] = net_setup_dict

    load_dict['load_flag'] = load_flag
    load_dict['load_run_name'] = load_run_name
    load_dict['load_episode'] = load_episode

    train_dict['load_info'] = load_dict

    from nn_utils import Neural_Net

    if 'output' not in os.listdir():
        os.mkdir('./output')

    if run_name not in os.listdir('./output/'):
        os.mkdir(save_base_path)
        os.mkdir(os.path.join(save_base_path, 'json'))
        os.mkdir(os.path.join(save_base_path, 'model'))
        os.mkdir(os.path.join(save_base_path, 'plots'))
        os.mkdir(os.path.join(save_base_path, 'tensorboard'))

    if 'tensorboard' in os.listdir(save_base_path):
        for f in os.listdir(log_dir):
            os.remove(os.path.join(log_dir, f))

    # Set the seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Global data parameters ==================================================
    NUM_EX_SHOCKS = 2
    A = num_agents
    assert NUM_EX_SHOCKS == 2, 'Two shocks hardcoded'
    
    # Parameters
    EU = np.array([1.2167, 1.6167, 2.1167, 2.4833, 2.7500, 2.9500, 3.1667, 3.3325, 3.4255, 3.5960, 3.6580, 3.8440, 3.7665, 3.4565, 3.2705, 3.0833, 2.7667, 2.0167, 1.4500, 0.9667])
    ED = np.array([1.2667, 1.6667, 2.1500, 2.4333, 2.6167, 2.7833, 2.9833, 3.1500, 3.1833, 3.2500, 3.2500, 3.2333, 3.0333, 2.8333, 2.7000, 2.3500, 2.0500, 1.5333, 1.1500, 0.7833])
    ALPHA = 0.13                                             
    BETA  = 0.83
    RHO = 4.5                  
    EUBAR = np.sum(EU)
    EDBAR = np.sum(ED)
    
    # Starting point from steady state solutions
    hu = 2 * np.array([
    0.1924, 0.2501, 0.3079, 0.3656, 0.4233, 0.5373, 0.5406, 0.5949,
    0.5985, 0.6021, 0.6058, 0.6095, 0.6132, 0.6170, 0.6207, 0.6245,
    0.6283, 0.6321])

    thetau = 2 * np.array([
    0.1924, 0.2501, 0.3079, 0.3656, 0.4233, 0.5337, 0.5406, 0.5907,
    0.5002, 0.3915, 0.2593, 0.0968, -0.1050, -0.3573, -0.6749,
    -0.9302, -1.0067, -0.9502])

    x0 = np.random.rand(1, 1 + 2 * (A - 2)+ 2 * (A-1) + 2)
    x0[:,0] = 0
    x0[:, 1:A-1] = hu-thetau
    x0[:,A-1:2*A-3] = 0
    x0[:,2*A-3: 3*A -4] = EU[:-1]
    x0[:,3*A-4:3*A-2] = np.array([0.9, 0.1])
    x0[:,3*A-2:] = EU[:-1]*0.85
    x0= x0.reshape([1, -1])
    
    # Transition matrix
    PI = np.array([[0.86, 0.14],
                  [0.80, 0.20]])

    print('PI =', PI)
    print('ALPHA =', ALPHA)
    print('BETA =', BETA)
    print('RHO =', RHO)
    

    
    econ_setup_dict['alpha'] = ALPHA
    econ_setup_dict['beta'] = BETA
    econ_setup_dict['rho'] = RHO
    econ_setup_dict['pi'] = PI.tolist()
    econ_setup_dict['num_ex_shocks'] = NUM_EX_SHOCKS
    
    train_dict['econ_params'] = econ_setup_dict

    for key in econ_setup_dict:
        print('{}: {}'.format(key, econ_setup_dict[key]))

    with tf.name_scope('econ_parameters'):
        eu = tf.constant(EU, dtype=tf.float32, name='eu')
        eubar = tf.constant(EUBAR, dtype=tf.float32, name='eubar')
        ed = tf.constant(ED, dtype=tf.float32, name='ed')
        edbar = tf.constant(EDBAR, dtype=tf.float32, name='edbar')
        pi = tf.constant(PI, dtype=tf.float32, name='pi')
        alpha = tf.constant(ALPHA, dtype=tf.float32, name='alpha')
        beta = tf.constant(BETA, dtype=tf.float32, name='beta')
        rho = tf.constant(RHO, dtype=tf.float32, name='rho')
        

    with tf.name_scope('neural_net'):
        n_input = 1 + 2 * (A-2) + A-1 + 2 + A-1
        n_output = 3 + 5 * (A-1)
        num_nodes = [n_input] + num_hidden_nodes + [n_output]
        activation_list = activations_hidden_nodes + [tf.nn.softplus]
        nn = Neural_Net(num_nodes, activation_list)

    X = tf.placeholder(tf.float32, shape=(None, n_input), name='X')

    with tf.name_scope('compute_cost'):
        eps = 0.00001

        # get number samples
        m = tf.shape(X)[0]

        with tf.name_scope('todays_consumption'):
            with tf.name_scope('decompose_state_vector'):
                # get current state
                with tf.name_scope('exog_shock'):
                    # exogenous shock
                    z = X[:, 0]
                    z = tf.reshape(X[:, 0],[m,1])
                    ez_U = tf.ones_like(z) * eu[:-1]
                    ez_D = tf.ones_like(z) * ed[:-1]
                    prob_from_D = tf.ones_like(z) * [[0.8, 0.20]]
                with tf.name_scope('endog_states'):
                    # use xd to get q_t when z_t = D
                    xd = tf.concat([tf.ones_like(z), 
                                    X[:,1:2*A-3],
                                    ez_D, 
                                    prob_from_D, 
                                    X[:,3*A-2:]], axis=1)
                    bu_today = X[:, 1:A-1]
                    bu19_today = 20 - tf.reduce_sum(bu_today, axis=1, keepdims=True)
                    thetad_today = X[:, A-1:2*A-3]
                    thetad19_today = - tf.reduce_sum(thetad_today, axis=1, keepdims=True)
                with tf.name_scope('endowment'):
                    ev = z * ed + (1 - z ) * eu
                with tf.name_scope('aggregate_endowment'):
                    ebar = z * edbar + (1 - z) * eubar
                with tf.name_scope('transition_probability'):
                    prob_next = X[:, 3*A-4: 3*A-2]
                    
            
                
            with tf.name_scope('get_todays_choice_variables'):
                with tf.name_scope('NN'):
                    endog = nn.predict(X)
                    q = endog[:, 0]
                    q = tf.reshape(q,[m,1])
                    piu = endog[:, 1]
                    piu = tf.reshape(piu,[m,1])
                    LTV_jU = (piu)/q
                    pid = endog[:, 2]
                    pid = tf.reshape(pid,[m,1])
                    LTV_jD = (pid)/q
                    h = endog[:, 3 : A + 2]
                    h_all = tf.concat([h, tf.zeros([m, 1])], axis=1)
                    bu = endog[:, A + 2: 2 * A + 1]
                    thetau = h - bu
                    thetau_all = tf.concat([thetau, tf.zeros([m, 1])], axis=1)
                    bd = endog[:, 2 * A + 1: 3 * A]
                    thetad = h - thetau - bd
                    theta = thetau + thetad
                    thetad_all = tf.concat([thetad, tf.zeros([m, 1])], axis=1)
                    miu = endog[:, 3 * A: 4 * A - 1]
                    miuu = endog[:, 4 * A - 1: 5 * A - 2]
                    # each contract shorted
                    coll_U = tf.maximum(thetau, tf.zeros_like(thetau))
                    coll_D = tf.maximum(thetad, tf.zeros_like(thetad))
                    all_coll_U = tf.reshape(tf.reduce_sum(coll_U, axis=1), [m,1])
                    all_coll_D = tf.reshape(tf.reduce_sum(coll_D, axis=1), [m,1])
                    # all contracts shorted
                    theta = coll_U + coll_D
                    # DLTV^a
                    DLTV = (piu * coll_U + pid * coll_D)/(q * h)
                    # get q_t when z_t = D
                    endogd = nn.predict(xd)
                    qd_today = endogd[:,0]
                    qd_today = tf.reshape(qd_today,[m,1])
                    
                    

            with tf.name_scope('financial_wealth_today_all'):
                fw1 = q * tf.concat([tf.zeros([m, 1]), bu_today, bu19_today], axis=1)
                fw2 = qd_today * tf.concat([tf.zeros([m, 1]), thetad_today, thetad19_today], axis=1)
                fw = fw1 - fw2

            with tf.name_scope('compute_todays_consumption'):
                # budget constraints embedded
                c = ev + fw - q * h_all + piu * thetau_all + pid * thetad_all
                c = tf.maximum(c, tf.ones_like(c) * eps, name='c_today')
                
            with tf.name_scope('compute_marginal_utility_today'):
                uc_ = tf.reshape((c[:,:-1]*h**alpha)**(-rho)*h**alpha,[m,A-1])
                uc_A = tf.reshape(c[:,-1]**(-rho),[m,1])
                uc = tf.concat([uc_, uc_A],axis=1, name = 'uc_today')
                # collateral value 
                cv1 = (miu+miuu)/uc[:,0:-1]
                cv2 = miuu/uc[:,0:-1]
            
            with tf.name_scope('tmorrows_consumption'):
                with tf.name_scope('get_tomorrows_state'):
                    with tf.name_scope('tomorrows_exog_shock'):
                        z_prime_U = tf.zeros_like(z, name='zprime_U')
                        z_prime_D = tf.ones_like(z, name='zprime_D')
                        
                    with tf.name_scope('transition'):
                    # prepare transitions
                        pi_trans_fromU = tf.gather(pi, tf.cast(z_prime_U, tf.int32))
                        pi_trans_toU_fromU = tf.expand_dims(pi_trans_fromU[:,0,0], -1)
                        pi_trans_toD_fromU = tf.expand_dims(pi_trans_fromU[:,0,1], -1)

                        pi_trans_fromD = tf.gather(pi, tf.cast(z_prime_D, tf.int32))
                        pi_trans_toU_fromD = tf.expand_dims(pi_trans_fromD[:,0,0], -1)
                        pi_trans_toD_fromD = tf.expand_dims(pi_trans_fromD[:,0,1], -1)

                    with tf.name_scope('concatenate_for_tomorrows_state'):
                        x_prime_U = tf.concat([z_prime_U, 
                                               bu[:, :-1], 
                                               thetad[:, :-1], 
                                               ez_U, 
                                               pi_trans_toU_fromU, 
                                               pi_trans_toD_fromU, 
                                               c[:,:-1]], axis=1)
                        x_prime_D = tf.concat([z_prime_D, 
                                               bu[:, :-1], 
                                               thetad[:, :-1], 
                                               ez_D, 
                                               pi_trans_toU_fromD, 
                                               pi_trans_toD_fromD, 
                                               c[:,:-1]], axis=1)
                
                with tf.name_scope('get_tomorrows_choice_variables'):
                    with tf.name_scope('NN'):
                        # z_t+1 = U
                        endog_prime_U = nn.predict(x_prime_U)
                        q_prime_U = endog_prime_U[:, 0]
                        q_prime_U = tf.reshape(q_prime_U,[m,1])
                        piu_prime_U = endog_prime_U[:, 1]
                        piu_prime_U = tf.reshape(piu_prime_U,[m,1])
                        pid_prime_U = endog_prime_U[:, 2]
                        pid_prime_U = tf.reshape(pid_prime_U,[m,1])
                        h_prime_U = endog_prime_U[:, 3 : A+2]
                        h_prime_U_all = tf.concat([h_prime_U, tf.zeros([m, 1])], axis=1)
                        bu_prime_U = endog_prime_U[:, A+2 : 2*A+1]
                        thetau_prime_U = h_prime_U - bu_prime_U
                        thetau_prime_U_all = tf.concat([thetau_prime_U, tf.zeros([m, 1])], axis=1)
                        bd_prime_U = endog_prime_U[:, 2*A+1 : 3*A]
                        thetad_prime_U = h_prime_U - thetau_prime_U - bd_prime_U
                        thetad_prime_U_all = tf.concat([thetad_prime_U, tf.zeros([m, 1])], axis=1)
                        
                        # z_t+1 = D                        
                        endog_prime_D = nn.predict(x_prime_D)
                        q_prime_D = endog_prime_D[:, 0]
                        q_prime_D = tf.reshape(q_prime_D,[m,1])
                        piu_prime_D = endog_prime_D[:, 1]
                        piu_prime_D = tf.reshape(piu_prime_D,[m,1])
                        pid_prime_D = endog_prime_D[:, 2]
                        pid_prime_D = tf.reshape(pid_prime_D,[m,1])
                        h_prime_D = endog_prime_D[:, 3 : A+2]
                        h_prime_D_all = tf.concat([h_prime_D, tf.zeros([m, 1])], axis=1)
                        bu_prime_D = endog_prime_D[:, A+2 : 2*A+1]
                        thetau_prime_D = h_prime_D - bu_prime_D
                        thetau_prime_D_all = tf.concat([thetau_prime_D, tf.zeros([m, 1])], axis=1)
                        bd_prime_D = endog_prime_D[:, 2*A+1 : 3*A]
                        thetad_prime_D = h_prime_D - thetau_prime_D - bd_prime_D  
                        thetad_prime_D_all = tf.concat([thetad_prime_D, tf.zeros([m, 1])], axis=1)
                        
                        # interest rates
                        R_jU = q_prime_U/(piu)
                        R_jD = q_prime_D/(pid)

                with tf.name_scope('financial_wealth_tomorrow_all'):
                    fw_prime_U = q_prime_U * tf.concat([tf.zeros([m, 1]), bu], axis=1) - q_prime_D * tf.concat([tf.zeros([m, 1]), thetad], axis=1)
                    fw_prime_D = q_prime_D * tf.concat([tf.zeros([m, 1]), bu-thetad], axis=1)
            

                with tf.name_scope('compute_tomorrows_consumption'):
                    c_prime_U = eu * tf.ones_like(z) + fw_prime_U - q_prime_U * h_prime_U_all + piu_prime_U *thetau_prime_U_all + pid_prime_U *thetad_prime_U_all
                    c_prime_D = ed * tf.ones_like(z) + fw_prime_D - q_prime_D * h_prime_D_all + piu_prime_D *thetau_prime_D_all + pid_prime_D *thetad_prime_D_all
                    c_prime_U = tf.maximum(c_prime_U, tf.ones_like(c_prime_U) * eps, name='cu_tomorrow')
                    c_prime_D = tf.maximum(c_prime_D, tf.ones_like(c_prime_D) * eps, name='cd_tomorrow')
                    
                    
                with tf.name_scope('marginal_utility_tmorrow_U'):
                    ucu_ = tf.reshape((c_prime_U[:,:-1]*h_prime_U**alpha)**(-rho)*h_prime_U**alpha, [m,A-1])
                    ucu_A = tf.reshape(c_prime_U[:,-1]**(-rho), [m,1])
                    ucu = tf.concat([ucu_, ucu_A],axis=1, name = 'ucu_tomorrow')
            
                with tf.name_scope('marginal_utility_tmorrow_D'):
                    ucd_ = tf.reshape((c_prime_D[:,:-1]*h_prime_D**alpha)**(-rho)*h_prime_D**alpha, [m,A-1])
                    ucd_A = tf.reshape(c_prime_D[:,-1]**(-rho), [m,1])
                    ucd = tf.concat([ucd_, ucd_A],axis=1, name = 'ucd_tomorrow')

                
            with tf.name_scope('optimality_conditions'):
                # collateral constraints
                with tf.name_scope('kkt_condition'):
                    kt1 = tf.multiply(h-thetau-thetad,miu/uc[:,0:-1])
                    kt2 = tf.multiply(bu,miuu/uc[:,0:-1])
                    opt_kkt = tf.concat([kt1,kt2],axis=1)

                with tf.name_scope('rel_ee'):
                # euler equations
                    pi_trans_toU = tf.expand_dims(prob_next[:,0],-1)*tf.ones([1,A-1])
                    pi_trans_toD = tf.expand_dims(prob_next[:,1],-1)*tf.ones([1,A-1])
                    opt_euler_ju = -1+(((miu + miuu + beta * (pi_trans_toU * q_prime_U * ucu[:,1:]+pi_trans_toD * q_prime_D*ucd[:,1:]))/(piu))**(-1/rho)*h**((alpha-alpha*rho)/rho))/c[:,0:-1]
                    opt_euler_jd = -1+(((miu + beta*(pi_trans_toU*q_prime_D*ucu[:,1:]+pi_trans_toD*q_prime_D*ucd[:,1:]))/(pid))**(-1/rho)*h**((alpha-alpha*rho)/rho))/c[:,0:-1]
                    opt_euler_h  = -1+(((miu + miuu + alpha*c[:,0:-1]/h*uc[:,0:-1] +beta*(pi_trans_toU*q_prime_U*ucu[:,1:]+pi_trans_toD*q_prime_D*ucd[:,1:]))/q)**(-1/rho)*h**((alpha-alpha*rho)/rho))/c[:,0:-1]        
                    opt_euler = tf.concat([opt_euler_ju, opt_euler_jd, opt_euler_h], axis = 1)

                    
                with tf.name_scope('intertemporal_marginal_utilities'):
                    Mar_U = beta*pi_trans_toU*ucu[:,1:]/uc[:,0:-1]
                    Mar_D = beta*pi_trans_toD*ucd[:,1:]/uc[:,0:-1]
                    
                    # SDF for a = 18
                    M_U_end = Mar_U[:,-1]
                    M_U_end = tf.reshape(M_U_end, [m,1])
                    M_D_end = Mar_D[:,-1]
                    M_D_end = tf.reshape(M_D_end, [m,1])
                    
                    # SDF for a = 1
                    M_U_start = Mar_U[:,0]
                    M_U_start = tf.reshape(M_U_start, [m,1])
                    M_D_start = Mar_D[:,0]
                    M_D_start = tf.reshape(M_D_start, [m,1])
                    
                    # for computing liquidity values
                    M_diff_U1 = (M_U_end - M_U_start)
                    M_diff_D1 = (M_D_end - M_D_start)
                    M_diff_U_all = M_U_end - Mar_U
                    M_diff_D_all = M_D_end - Mar_D
                    

                with tf.name_scope('plot_credit_surfaces'):
                    start_j = tf.zeros([m,1])
                    end_j = q_prime_U + 0.2
                    num_columns = 500
                    j = tf.linspace(0.0, 1.0, num_columns)
                    j_matrix = tf.reshape(j, [1, num_columns])
                    j = j * (end_j - start_j) + start_j 
                    pij_matrix = M_U_end * tf.minimum(j, q_prime_U) + M_D_end * tf.minimum(j, q_prime_D)
                    LTVj_matrix = pij_matrix/q
                    Rj_matrix = j/(pij_matrix)
                    # liquidity values for a = 1
                    LVj1_matrix = M_diff_U1*tf.minimum(j,q_prime_U) + M_diff_D1*tf.minimum(j,q_prime_D)

                with tf.name_scope('market_clearing'):
                # market clearing conditions
                    mc_cons = tf.reduce_sum(c,axis=1,keepdims=True)-ebar
                    
                    mc_thetau = tf.reduce_sum(thetau,axis=1,keepdims=True)
                    mc_thetau_U = tf.reduce_sum(thetau_prime_U,axis=1,keepdims=True)
                    mc_thetau_D = tf.reduce_sum(thetau_prime_D,axis=1,keepdims=True)

                    mc_thetad = tf.reduce_sum(thetad,axis=1,keepdims=True)
                    mc_thetad_U = tf.reduce_sum(thetad_prime_U,axis=1,keepdims=True)
                    mc_thetad_D = tf.reduce_sum(thetad_prime_D,axis=1,keepdims=True)

                    mc_h = tf.reduce_sum(h,axis=1,keepdims=True)-20
                    mc_h_U = tf.reduce_sum(h_prime_U,axis=1,keepdims=True)-20
                    mc_h_D = tf.reduce_sum(h_prime_D,axis=1,keepdims=True)-20

                    opt_mc = tf.concat([mc_thetau,mc_thetau_U,mc_thetau_D,
                                        mc_thetad,mc_thetad_U,mc_thetad_D,
                                        mc_h,mc_h_U,mc_h_D,mc_cons], axis=1)
                
                with tf.name_scope('punishments'):     
                # punish if outside of state space    
                    opt_punish_h = (1.0/eps) * tf.maximum(h-20, tf.zeros_like(h))
                    opt_punish_bu = (1.0/eps) * tf.maximum(bu-20, tf.zeros_like(bu))
                    opt_punish_bd = (1.0/eps) * tf.maximum(bd-20, tf.zeros_like(bd))
                    opt_punish_b = (1.0/eps) * tf.maximum(eps-(h-thetad), tf.zeros_like(h))
                    # guess a = 1 to 6 are constrained borrowers
                    guess_regime = tf.concat([thetad[:,0:6]],axis=1)
                    opt_punish = tf.concat([guess_regime, opt_punish_h, opt_punish_bu, opt_punish_bd, opt_punish_b],axis=1) 

                    
                # Put together
                # more weight to collateral constraints
                combined_opt = [10*opt_kkt, opt_euler, opt_mc, opt_punish]
                opt_predict = tf.concat(combined_opt, axis=1, name='combined_opt_cond')

                with tf.name_scope('compute_cost'):
                    # define the correct output
                    opt_correct = tf.zeros_like(opt_predict, name='target')

                    # define the cost function
                    cost = tf.losses.mean_squared_error(opt_correct, opt_predict)

    with tf.name_scope('train_setup'):
        if optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='adam')
        else:
            raise NotImplementedError

        with tf.name_scope('gradients'):
            gvs = optimizer.compute_gradients(cost)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_step = optimizer.apply_gradients(capped_gvs)

    with tf.name_scope('simulate_episode'):
        def simulate_episodes(sess, x_start, episode_length=10**4, print_flag=True):
            sim_start_time = datetime.now()
            if print_flag:
                print('Start simulating {} periods.'.format(episode_length))
            num_state_var = np.shape(x_start)[1]
            X_episodes = np.zeros([episode_length, num_state_var])
            X_episodes[0, :] = x_start
            X_old = x_start
            # Generate a sequence of random shocks
            rand_num = np.random.rand(episode_length, 1)
            for t in range(1, episode_length):
                if rand_num[t - 1] <= PI[int(X_old[0, 0]), 0]:
                    X_new = sess.run(x_prime_U, feed_dict={X: X_old})
                else:
                    X_new = sess.run(x_prime_D, feed_dict={X: X_old})
                X_episodes[t, :] = X_new
                X_old = X_new.copy()
            sim_end_time = datetime.now()
            sim_duration = sim_end_time - sim_start_time
            if print_flag:
                print('Finished simulation. Time for simulation: {}.'.format(sim_duration))
            return X_episodes

        def simulate_batch_episodes(sess, x_start, episode_length=10**4, print_flag=True):
            sim_start_time = datetime.now()
            num_state_var = np.shape(x_start)[1]
            batch_size = np.shape(x_start)[0]
            if print_flag:
                print('Start simulating {} tracks with {} periods.'.format(batch_size, episode_length))
            X_episodes = np.zeros([batch_size * episode_length, num_state_var])
            X_old = x_start
            rand_num = np.random.rand(batch_size, episode_length)
            for t in range(0, episode_length):
                temp_rand = rand_num[:, t]
                X_new = np.zeros((batch_size, num_state_var))
                trans_probs_toU = X_old[:, 3*A-4]
                trans_probs_toD = X_old[:, 3*A-4+1]
                to_U = temp_rand <= trans_probs_toU
                to_D = temp_rand > trans_probs_toU
                X_new[to_U, :] = sess.run(x_prime_U, feed_dict={X: X_old[to_U, :]})
                X_new[to_D, :] = sess.run(x_prime_D, feed_dict={X: X_old[to_D, :]})
                X_episodes[t * batch_size : (t+1) * batch_size , :] = X_new
                X_old = X_new.copy()
            sim_end_time = datetime.now()
            sim_duration = sim_end_time - sim_start_time
            if print_flag:
                print('Finished simulation. Time for simulation: {}.'.format(sim_duration))
            return X_episodes

        
    # Training
    sess = tf.Session()

    with tf.name_scope('get_starting_point'):
        if not(load_flag):
            X_data_train = np.random.rand(1, n_input)
            X_data_train[:, :] = x0
            X_data_train= X_data_train.reshape([1, -1])
            print('Calculated a valid starting point', X_data_train)
        else:
            load_base_path = os.path.join('./output',  load_run_name)
            load_params_nm = load_run_name + '-episode' + str(load_episode)
            load_params_path = os.path.join(load_base_path, 'model', load_params_nm)
            load_data_path = os.path.join(load_base_path,  'model', load_params_nm + '_LastData.npy')
            X_data_train = np.load(load_data_path)
            print('Loaded initial data from ' + load_data_path)
            print('Starting point: ', X_data_train)

    with tf.name_scope('training'):
        minibatch_size = int(batch_size)
        num_minibatches = int(len_episodes / minibatch_size)
        train_seed = 0

        cost_store = np.zeros(num_episodes)
        mov_ave_cost_store = np.zeros(num_episodes)
        mov_ave_len = 100

        time_store = np.zeros(num_episodes)
        mean_ee_store_h = np.zeros((num_episodes, (num_agents - 1)))
        mean_ee_store_ju = np.zeros((num_episodes, (num_agents - 1)))
        mean_ee_store_jd = np.zeros((num_episodes, (num_agents - 1)))
        max_ee_store_h = np.zeros((num_episodes, (num_agents - 1)))
        max_ee_store_ju = np.zeros((num_episodes, (num_agents - 1)))
        max_ee_store_jd = np.zeros((num_episodes, (num_agents - 1)))

        start_time = datetime.now()
        print('start time: {}'.format(start_time))

        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        init = tf.global_variables_initializer()

        print_interval = 100

        sim_batch_size = 1000
        sim_len = int(len_episodes/sim_batch_size)

        # run the initializer
        sess.run(init)

        if load_flag:
            saver = tf.train.Saver(nn.param_dict)
            saver.restore(sess, load_params_path)
            print('Weights loaded from: ' + load_params_path)

        for ep in range(load_episode, num_episodes + load_episode):
            if ep == load_episode:
                if ep <= 2:
                    X_data_train = np.matlib.repmat(X_data_train, sim_batch_size, 1)

            print_flag = (ep % print_interval == 0) or ep == load_episode

            if print_flag:
                print('Episode {}'.format(ep))
            start_time_learn = datetime.now()

            X_episodes = simulate_batch_episodes(sess, X_data_train, episode_length=sim_len, print_flag=print_flag)
            X_data_train = X_episodes[len_episodes - sim_batch_size : len_episodes, :]

            if print_flag:
                print('Starting learning on episode')

            for epoch in range(epochs_per_episode):
                if print_flag:
                    print('Epoch {} on this episode.'.format(epoch))
                train_seed = train_seed + 1

                minibatches = random_mini_batches(X_episodes, minibatch_size, train_seed)
                minibatch_cost = 0
                
                if epoch == 0:
                    ee_error_contract1 = np.zeros((1, num_agents-1))
                    ee_error_contract2 = np.zeros((1, num_agents-1))
                    ee_error_housing = np.zeros((1, num_agents-1))
                    max_ee_contract1 = np.zeros((1, num_agents-1))
                    max_ee_contract2 = np.zeros((1, num_agents-1))
                    max_ee_housing = np.zeros((1, num_agents-1))

                for minibatch in minibatches:
                    (minibatch_X) = minibatch

                    # Run optimization
                    minibatch_cost += sess.run(cost, feed_dict={X: minibatch_X}) / num_minibatches
                    if epoch == 0:
                        opt_euler_1 = np.abs(sess.run(opt_euler_ju, feed_dict={X: minibatch_X}))
                        opt_euler_2 = np.abs(sess.run(opt_euler_jd, feed_dict={X: minibatch_X}))
                        opt_euler_housing_ = np.abs(sess.run(opt_euler_h, feed_dict={X: minibatch_X}))

                        ee_error_contract1 += np.mean(opt_euler_1, axis=0) / num_minibatches
                        ee_error_contract2 += np.mean(opt_euler_2, axis=0) / num_minibatches
                        ee_error_housing += np.mean(opt_euler_housing_, axis=0) / num_minibatches

                        mb_max_ee_contract1 = np.max(opt_euler_1, axis=0, keepdims=True)
                        mb_max_ee_contract2 = np.max(opt_euler_2, axis=0, keepdims=True)
                        mb_max_ee_housing = np.max(opt_euler_housing_, axis=0, keepdims=True)

                        max_ee_contract1 = np.maximum(max_ee_contract1, mb_max_ee_contract1)
                        max_ee_contract2 = np.maximum(max_ee_contract2, mb_max_ee_contract2)
                        max_ee_housing = np.maximum(max_ee_housing, mb_max_ee_housing)


                if epoch == 0:
                    cost_store[ep-load_episode] = minibatch_cost

                if print_flag:
                    print('Epoch {}, log10(Cost)= {:.4f}'.format(epoch, np.log10(minibatch_cost)))

                if train_flag:
                    for minibatch in minibatches:
                        (minibatch_X) = minibatch

                        # Run train step
                        sess.run(train_step, feed_dict={X: minibatch_X})

            end_time_learn = datetime.now()
            if print_flag:
                print('Finished learning on episode. Time for learning: {}.'.format(end_time_learn - start_time_learn))

            if ep-load_episode > mov_ave_len + 10:
                mov_ave_cost_store[ep-load_episode] = np.mean(cost_store[ep-load_episode-mov_ave_len:ep-load_episode])
            else:
                mov_ave_cost_store[ep-load_episode] = np.mean(cost_store[0:ep-load_episode])

            mean_ee_store_h[ep-load_episode, :] = ee_error_housing
            mean_ee_store_ju[ep-load_episode, :] = ee_error_contract1
            mean_ee_store_jd[ep-load_episode, :] = ee_error_contract2
            max_ee_store_h[ep-load_episode, :] = max_ee_housing
            max_ee_store_ju[ep-load_episode, :] = max_ee_contract1
            max_ee_store_jd[ep-load_episode, :] = max_ee_contract2
            
            cur_time = datetime.now() - start_time
            time_store[ep-load_episode] = cur_time.seconds

            # Calculate cost
            print('\nEpisode {}, log10(Cost)= {:.4f}'.format(ep, np.log10(cost_store[ep-load_episode])))
            print('Time: {}; time since start: {}'.format(datetime.now(), datetime.now() - start_time))

            if ep % save_interval == 0 or ep == 1:
                plot_dict = {}
                plot_epi_length = 500000

                #simulate new episodes to plot
                X_data_train_plot = X_episodes[-1, :].reshape([1, -1])
                X_episodes = simulate_episodes(sess, X_data_train_plot, episode_length=plot_epi_length, print_flag=print_flag)
                plot_period = np.arange(1, plot_epi_length+1)
                len_plot_episodes = plot_epi_length

                plot_age_all = np.arange(1, A+1)
                plot_age_exceptlast = np.arange(1, A)

                plt.rc('font', family='serif')
                plt.rc('xtick', labelsize='small')
                plt.rc('ytick', labelsize='small')

                std_figsize = (4, 4)
                percentiles_dict = {100:{'ls':':', 'label':'max'}} 
                
                shockU_dict = {'label': 'shock U', 'color':'r'}
                shockD_dict = {'label': 'shock D', 'color':'b'}
                shock_dict = {1: shockU_dict, 2: shockD_dict}



                # run stuff
                a_condU = (X_episodes[:plot_epi_length, 0] == 0)
                a_condD = (X_episodes[:plot_epi_length, 0] == 1)                    
                ebar_ = sess.run(ebar, feed_dict={X: X_episodes})
                c_ = sess.run(c, feed_dict={X: X_episodes})
                c__ = c_[:,:-1]
                c_prime_U_ = sess.run(c_prime_U, feed_dict={X: X_episodes})
                c_prime_D_ = sess.run(c_prime_D, feed_dict={X: X_episodes})
                x_prime_U_ = sess.run(x_prime_U, feed_dict={X: X_episodes})
                x_prime_D_ = sess.run(x_prime_D, feed_dict={X: X_episodes})
                q_ = sess.run(q, feed_dict={X:X_episodes})
                h_ = sess.run(h, feed_dict = {X:X_episodes})
                ch_ = c__/h_
                
                Bond_U_price_ = sess.run(piu, feed_dict={X:X_episodes})
                thetau_ = -1*sess.run(thetau, feed_dict = {X:X_episodes})
                LTV_jU_ = sess.run(LTV_jU, feed_dict = {X:X_episodes})
                R_jU_ = sess.run(R_jU, feed_dict = {X:X_episodes})
                
                Bond_D_price_ = sess.run(pid, feed_dict={X:X_episodes})
                thetad_ = -1*sess.run(thetad, feed_dict = {X:X_episodes})
                LTV_jD_ = sess.run(LTV_jD, feed_dict = {X:X_episodes})
                R_jD_ = sess.run(R_jD, feed_dict = {X:X_episodes})
                
                theta_ = sess.run(theta, feed_dict = {X:X_episodes})
                Mar_D_ = sess.run(Mar_D, feed_dict = {X:X_episodes})
                
                mu_ = sess.run(miu, feed_dict={X: X_episodes})
                muu_ = sess.run(miuu, feed_dict={X: X_episodes})
                
                opt_euler_h_ = sess.run(opt_euler_h, feed_dict={X: X_episodes})
                opt_euler_ju_ = sess.run(opt_euler_ju, feed_dict={X: X_episodes})
                opt_euler_jd_ = sess.run(opt_euler_jd, feed_dict={X: X_episodes})
                opt_euler_ = sess.run(opt_euler, feed_dict={X:X_episodes})
                
                mc_cons_ = sess.run(mc_cons, feed_dict={X: X_episodes})
                mc_h_ = sess.run(mc_h, feed_dict={X: X_episodes})
                mc_thetau_ = sess.run(mc_thetau, feed_dict={X: X_episodes})
                mc_thetad_ = sess.run(mc_thetad, feed_dict={X: X_episodes})
                
                M_diff_U_ = sess.run(M_diff_U1, feed_dict={X: X_episodes}) 
                M_diff_D_ = sess.run(M_diff_D1, feed_dict={X: X_episodes}) 
                
                
                kt1_ = sess.run(kt1, feed_dict={X: X_episodes}) 
                kt2_ = sess.run(kt2, feed_dict={X: X_episodes}) 
                
                cv1_ = sess.run(cv1, feed_dict={X: X_episodes}) 
                cv2_ = sess.run(cv2, feed_dict={X: X_episodes}) 
                
                j_ = sess.run(j, feed_dict={X: X_episodes}) 
                LVj1_matrix_ = sess.run(LVj1_matrix, feed_dict={X: X_episodes}) 
                pij_matrix_ = sess.run(pij_matrix, feed_dict={X: X_episodes}) 
                LTVj_matrix_ = sess.run(LTVj_matrix, feed_dict={X: X_episodes}) 
                Rj_matrix_ = sess.run(Rj_matrix, feed_dict={X: X_episodes}) 
                
                DLTV_ = sess.run(DLTV, feed_dict={X: X_episodes}) 
                
                
                q_U = q_[a_condU, 0]
                q_D = q_[a_condD, 0]
                LTV_jU_U = LTV_jU_[a_condU, 0]
                LTV_jU_D = LTV_jU_[a_condD, 0]
                LTV_jD_U = LTV_jD_[a_condU, 0]
                LTV_jD_D = LTV_jD_[a_condD, 0]
                R_jU_U = R_jU_[a_condU, 0]
                R_jU_D = R_jU_[a_condD, 0]
                R_jD_U = R_jD_[a_condU, 0]
                R_jD_D = R_jD_[a_condD, 0]
                
                M_diff_U_all_ = sess.run(M_diff_U_all, feed_dict={X: X_episodes})
                M_diff_D_all_ = sess.run(M_diff_D_all, feed_dict={X: X_episodes})
                
                
                ######### Plots #######
                average_q_U = np.mean(q_U)       
                average_q_D = np.mean(q_D)
                average_LTV_jU_U = np.mean(LTV_jU_U)
                average_LTV_jU_D = np.mean(LTV_jU_D)
                average_LTV_jD_U = np.mean(LTV_jD_U)                
                average_LTV_jD_D = np.mean(LTV_jD_D)
                average_R_jU_U = np.mean(R_jU_U)
                average_R_jU_D = np.mean(R_jU_D)
                average_R_jD_U = np.mean(R_jD_U)
                average_R_jD_D = np.mean(R_jD_D)
                
                
                print('(1) Coefficient of variation of housing price:', np.std(q_)/np.mean(q_))
                print('(2) Average housing price in U/D:', average_q_U, average_q_D)
                print('(3) Average LTV_jU in U/D:', average_LTV_jU_U, average_LTV_jU_D)                
                print('(4) Average LTV_jD in U/D:', average_LTV_jD_U, average_LTV_jD_D)
                print('(5) Average R_jU in U/D:', average_R_jU_U, average_R_jU_D)
                print('(6) Average R_jD in U/D:', average_R_jD_U, average_R_jD_D)
                
                # Liquidity Values for a = 1
                plt.figure(figsize=std_figsize, facecolor='none')  # 
                ax6 = plt.subplot(1,1,1)
                plt.plot(j_[a_condU,:], LVj1_matrix_[a_condU, :], color='red')
                plt.plot(j_[a_condD,:], LVj1_matrix_[a_condD, :], color='blue')
                
                ax6.set_xlabel('j')
                ax6.set_ylabel('Liquidity Value')
                plt.savefig(plot_dir + '/' + run_name + '_Liquidity_Values_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()
                
                
                # Credit Surfaces               
                plt.figure(figsize=std_figsize, facecolor='none')  # 
                ax6 = plt.subplot(1,1,1)
                plt.plot(LTVj_matrix_[a_condU, :], Rj_matrix_[a_condU, :], color='red', alpha=0.011)
                plt.plot(LTVj_matrix_[a_condD, :], Rj_matrix_[a_condD, :], color='blue', alpha=0.011)
                plt.plot(np.mean(LTVj_matrix_[a_condU, :],axis=0), np.mean(Rj_matrix_[a_condU, :],axis=0), color='darkred', alpha=1, label = 'Up')
                plt.plot(np.mean(LTVj_matrix_[a_condD, :],axis=0), np.mean(Rj_matrix_[a_condD, :],axis=0), color='darkblue', alpha=1, label = 'Down')
                
                ax6.set_xlabel('LTV')
                ax6.set_ylabel('R')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(plot_dir + '/' + run_name + '_Credit_surfaces_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()
               
            
                # DLTV
                fig=plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                plt.plot(plot_age_exceptlast, np.mean(DLTV_, axis=0), 'k-', marker='o', label = 'mean')
                ax.set_xlabel('Age')
                ax.set_ylabel('DLTV')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(plot_dir + '/' + run_name + '_DLTV_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()
                
                
                # Check liquidity Value
                fig = plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                lines, labs1 =  [], ['Mdiff_U, max', 'Mdiff_U, min', 'Mdiff_D, max', 'Mdiff_U, min']
                l1, = plt.plot(plot_age_exceptlast, np.max(M_diff_U_all_, axis=0), 'k-', label = 'max, M_U')
                l2, = plt.plot(plot_age_exceptlast, np.min(M_diff_U_all_, axis=0), 'r-', label = 'min, M_U')
                l3, = plt.plot(plot_age_exceptlast, np.max(M_diff_D_all_, axis=0), 'g-', label = 'max, M_D')
                l4, = plt.plot(plot_age_exceptlast, np.min(M_diff_D_all_, axis=0), 'b-', label = 'min, M_D')
                
                lines.append([l1, l2, l3, l4])
                ax.set_xlabel('Age')
                ax.set_ylabel('Liquidity Value Check')
                legend1 = plt.legend(lines[0], labs1, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[0] for l in lines], labs1, bbox_to_anchor=(1.05, 0.8), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name + '_Liquidity_Value_check_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                     
                
                

                # Consumptions_today
                plt.figure(figsize=std_figsize)
                ax1 = plt.subplot(1,1,1)
                for perc_key in percentiles_dict:
                    ax1.plot(plot_age_all, np.percentile(c_, perc_key, axis=0), 'k'+percentiles_dict[perc_key]['ls'], label = percentiles_dict[perc_key]['label']+' percentile')
                ax1.plot(plot_age_all, np.mean(c_, axis=0), 'k-', label = 'mean')
                ax1.set_ylabel('c')
                ax1.set_xlabel('Age')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name + '_Cons_today_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                cons_today_dict = {'x': plot_age_all.tolist(), 'y': c_.tolist()}
                plot_dict['cons_today'] = cons_today_dict
                
                # Housing Distribution: h
                fig=plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                ax.plot(plot_age_exceptlast, np.mean(h_, axis=0), 'b-', label = 'House')
                ax.plot(plot_age_exceptlast, np.mean(c__, axis=0), 'k-', label = 'Consumption')
                ax.plot(plot_age_exceptlast, np.mean(ch_, axis=0), 'r-', label = 'Ratio')
                ax.set_xlabel('Age')
                ax.set_ylabel('Consumption and Housing Distribution')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(plot_dir + '/' + run_name + '_C&House_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                housebought_dict = {'x': plot_age_exceptlast.tolist(), 'y': h_.tolist()}
                plot_dict['housebought'] = housebought_dict


                # Contracts: thetau, thetad, h
                fig=plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                lines, labs1, labs2 =  [], ['mean'], ['thetaU', 'thetaD', 'house','sum']
                l1, = plt.plot(plot_age_exceptlast, np.mean(thetau_, axis=0), 'r-', marker='o', markersize=6)
                l2, = plt.plot(plot_age_exceptlast, np.mean(thetad_, axis=0), 'k-', marker='+', markersize=6)
                l3, = plt.plot(plot_age_exceptlast, np.mean(h_, axis=0), 'g-', marker='x', markersize=6)
                l4, = plt.plot(plot_age_exceptlast, np.mean(theta_, axis=0), 'b-', marker='D', markersize=6)
                lines.append([l1, l2, l3, l4])
                ax.set_xlabel('Age')
                ax.set_ylabel('Contracts traded')
                ax.grid(axis='y')
                legend1 = plt.legend(lines[0], labs2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[1] for l in lines], labs1, bbox_to_anchor=(1.05, 0.6), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name + '_Contracts_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                contracts_dict = {'x1': plot_age_exceptlast.tolist(), 'x2': plot_age_exceptlast.tolist(),'x3': plot_age_exceptlast.tolist(),'y1': thetau_.tolist(), 'y2': thetad_.tolist(), 'y3': h_.tolist(), 'y4': theta_.tolist()}
                plot_dict['Contracts'] = contracts_dict

                
                # Housing price in two shocks
                fig=plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                plt.plot(plot_period[a_condU], q_[a_condU, 0], shock_dict[1]['color'] + '*', label = shock_dict[1]['label'])
                plt.plot(plot_period[a_condD], q_[a_condD, 0], shock_dict[2]['color'] + 'o', label = shock_dict[2]['label'])
                plt.plot(np.ones_like(q_) * np.mean(q_), 'k-', label = 'Mean')
                ax.set_ylabel('Housing price')
                ax.set_xlabel('time period')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name + '_Housing_price_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                housingprice_dict = {'x1': plot_period[a_condU].tolist(), 'x2': plot_period[a_condD].tolist(),
                            'y1': q_[a_condU, 0].tolist(), 'y2': q_[a_condD, 0].tolist()}
                plot_dict['Housingprice'] = housingprice_dict

                
                # LTV_jU in two shocks
                fig=plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                plt.plot(plot_period[a_condU], LTV_jU_[a_condU, 0], shock_dict[1]['color'] + '*', label = shock_dict[1]['label'])
                plt.plot(plot_period[a_condD], LTV_jU_[a_condD, 0], shock_dict[2]['color'] + 'o', label = shock_dict[2]['label'])
                plt.plot(np.ones_like(LTV_jU_) * np.mean(LTV_jU_), 'k-', label = 'Mean')
                ax.set_ylabel('LTV_jU')
                ax.set_xlabel('time period')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name + '_LTV_jU_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                ltvju_dict = {'x1': plot_period[a_condU].tolist(), 'x2': plot_period[a_condD].tolist(),
                            'y1': LTV_jU_[a_condU, 0].tolist(), 'y2': LTV_jU_[a_condD, 0].tolist()}
                plot_dict['LTV_jU'] = ltvju_dict
                
                # LTV_jD in two shocks
                fig=plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                plt.plot(plot_period[a_condU], LTV_jD_[a_condU, 0], shock_dict[1]['color'] + '*', label = shock_dict[1]['label'])
                plt.plot(plot_period[a_condD], LTV_jD_[a_condD, 0], shock_dict[2]['color'] + 'o', label = shock_dict[2]['label'])
                plt.plot(np.ones_like(LTV_jD_) * np.mean(LTV_jD_), 'k-', label = 'Mean')
                ax.set_ylabel('LTV_jD')
                ax.set_xlabel('time period')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name + '_LTV_jD_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                ltvjd_dict = {'x1': plot_period[a_condU].tolist(), 'x2': plot_period[a_condD].tolist(),
                            'y1': LTV_jD_[a_condU, 0].tolist(), 'y2': LTV_jD_[a_condD, 0].tolist()}
                plot_dict['LTV_jD'] = ltvjd_dict
                
                
                
                # R_jU in two shocks
                fig=plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                plt.plot(plot_period[a_condU], R_jU_[a_condU, 0], shock_dict[1]['color'] + '*', label = shock_dict[1]['label'])
                plt.plot(plot_period[a_condD], R_jU_[a_condD, 0], shock_dict[2]['color'] + 'o', label = shock_dict[2]['label'])
                plt.plot(np.ones_like(R_jU_) * np.mean(R_jU_), 'k-', label = 'Mean')
                ax.set_ylabel('R_jU')
                ax.set_xlabel('time period')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name + '_R_jU_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                rju_dict = {'x1': plot_period[a_condU].tolist(), 'x2': plot_period[a_condD].tolist(),
                            'y1': R_jU_[a_condU, 0].tolist(), 'y2': R_jU_[a_condD, 0].tolist()}
                plot_dict['R_jU'] = rju_dict
                
                
                
                # R_jD in two shocks
                fig=plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                plt.plot(plot_period[a_condU], R_jD_[a_condU, 0], shock_dict[1]['color'] + '*', label = shock_dict[1]['label'])
                plt.plot(plot_period[a_condD], R_jD_[a_condD, 0], shock_dict[2]['color'] + 'o', label = shock_dict[2]['label'])
                plt.plot(np.ones_like(R_jD_) * np.mean(R_jD_), 'k-', label = 'Mean')
                ax.set_ylabel('R_jD')
                ax.set_xlabel('time period')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name + '_R_jD_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                rjd_dict = {'x1': plot_period[a_condU].tolist(), 'x2': plot_period[a_condD].tolist(),
                            'y1': R_jD_[a_condU, 0].tolist(), 'y2': R_jD_[a_condD, 0].tolist()}
                plot_dict['R_jD'] = rjd_dict
                
                # Consumptions_tomorrow
                plt.figure(figsize=std_figsize)
                ax1 = plt.subplot(1,1,1)
                for perc_key in percentiles_dict:
                    ax1.plot(plot_age_all, np.percentile(c_prime_U_, perc_key, axis=0), shock_dict[1]['color']+percentiles_dict[perc_key]['ls'])
                    ax1.plot(plot_age_all, np.percentile(c_prime_D_, perc_key, axis=0), shock_dict[2]['color']+percentiles_dict[perc_key]['ls'])
                ax1.plot(plot_age_all, np.mean(c_prime_U_, axis=0), shock_dict[1]['color'], label = shock_dict[1]['label'])
                ax1.plot(plot_age_all, np.mean(c_prime_D_, axis=0), shock_dict[2]['color'], label = shock_dict[2]['label'])
                ax1.set_ylabel("c'")
                ax1.set_xlabel('Age')
                plt.legend()
                plt.savefig(plot_dir + '/' + run_name + '_Cons_tomorrow_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                cons_tomorrow_dict = {'x1': plot_age_all.tolist(), 'x2': plot_age_all.tolist(),'y1': c_prime_U_.tolist(), 'y2': c_prime_D_.tolist()}
                plot_dict['cons_tomorrow'] = cons_tomorrow_dict
               
                
                
                
                # Errors: collateral constraints
                fig = plt.figure(figsize = std_figsize)
                ax = fig.add_subplot(1,1,1)
                lines, labs1, labs2 =  [], ['mean'], ['kt_h', 'kt_U']
                l1, = plt.plot(plot_age_exceptlast, np.log10(np.mean(kt1_, axis=0)), 'k-', label = 'kt_h, mean')
                l2, = plt.plot(plot_age_exceptlast, np.log10(np.mean(kt2_, axis=0)), 'r-', label = 'kt_U, mean')
                lines.append([l1, l2])
                for perc_key in percentiles_dict:
                    l1, = plt.plot(plot_age_exceptlast, np.log10(np.percentile(kt1_, perc_key, axis=0)), 'k'+percentiles_dict[perc_key]['ls'], label ='house, ' + percentiles_dict[perc_key]['label'])
                    l2, = plt.plot(plot_age_exceptlast, np.log10(np.percentile(kt2_, perc_key, axis=0)), 'r'+percentiles_dict[perc_key]['ls'], label ='jU, ' + percentiles_dict[perc_key]['label'])
                    lines.append([l1, l2])
                    labs1.append(percentiles_dict[perc_key]['label'])
                ax.set_xlabel('Age')
                ax.set_ylabel('Errors')
                legend1 = plt.legend(lines[0], labs2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[0] for l in lines], labs1, bbox_to_anchor=(1.05, 0.8), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name + '_Collateral_constraint_errors_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()
                # CSV
                kt_combined = np.concatenate([np.abs(kt1_), np.abs(kt2_)])
                collateral_constraint_aggregate = {
                    'Error Type': ['Combined'],  # Single row for aggregate
                    'Min': [np.log10(np.min(kt_combined))],
                    '25': [np.log10(np.percentile(kt_combined, 25))],
                    'Median': [np.log10(np.median(kt_combined))],
                    '99': [np.log10(np.percentile(kt_combined, 99))],
                    'Mean': [np.log10(np.mean(kt_combined))],
                    'Max': [np.log10(np.max(kt_combined))]}
                collateral_constraint_df = pd.DataFrame(collateral_constraint_aggregate)
                csv_path = os.path.join(plot_dir, f'{run_name}_Collateral_Constraint_Error_Summary_episode{ep}.csv')
                collateral_constraint_df.to_csv(csv_path, index=False)
                print(f"Collateral constraint error summary saved to: {csv_path}")

                

                # Errors: euler equations
                plt.figure(figsize=std_figsize)
                ax4 = plt.subplot(1,1,1)
                lines, labs1, labs2 =  [], ['mean'], ['house', 'jU', 'jD']
                l1, = ax4.plot(plot_age_exceptlast, np.log10(np.mean(np.abs(opt_euler_h_), axis=0)), 'k-', label = 'house, mean')
                l2, = ax4.plot(plot_age_exceptlast, np.log10(np.mean(np.abs(opt_euler_ju_), axis=0)), 'r-', label = 'jU, mean')
                l3, = ax4.plot(plot_age_exceptlast, np.log10(np.mean(np.abs(opt_euler_jd_), axis=0)), 'b-', label = 'jD, mean')
                lines.append([l1, l2, l3])
                for perc_key in percentiles_dict:
                    l1, = ax4.plot(plot_age_exceptlast, np.log10(np.percentile(np.abs(opt_euler_h_), perc_key, axis=0)), 'k'+percentiles_dict[perc_key]['ls'], label = 'house, ' + percentiles_dict[perc_key]['label'])#+' percentile')
                    l2, = ax4.plot(plot_age_exceptlast, np.log10(np.percentile(np.abs(opt_euler_ju_), perc_key, axis=0)), 'r'+percentiles_dict[perc_key]['ls'], label = 'jU, ' + percentiles_dict[perc_key]['label'])#+' percentile')
                    l3, = ax4.plot(plot_age_exceptlast, np.log10(np.percentile(np.abs(opt_euler_jd_), perc_key, axis=0)), 'b'+percentiles_dict[perc_key]['ls'], label = 'jD, ' + percentiles_dict[perc_key]['label'])#+' percentile')
                    lines.append([l1, l2, l3])
                    labs1.append(percentiles_dict[perc_key]['label'])# +' percentile')
                ax4.set_xlabel('Age')
                ax4.set_ylabel('Euler error [log10]')
                ax4.grid(axis='y')
                legend1 = plt.legend(lines[0], labs2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[0] for l in lines], labs1, bbox_to_anchor=(1.05, 0.8), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name + '_Euler_Error_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()
                # Errors: Euler equations
                plt.figure(figsize=std_figsize)
                ax4 = plt.subplot(1, 1, 1)
                lines, labs1, labs2 = [], ['mean', 'max'], ['house', 'jU', 'jD']
                # Calculate and plot mean
                l1, = ax4.plot(plot_age_exceptlast, np.log10(np.mean(np.abs(opt_euler_h_), axis=0)), 'k-', label='house, mean')
                l2, = ax4.plot(plot_age_exceptlast, np.log10(np.mean(np.abs(opt_euler_ju_), axis=0)), 'r-', label='jU, mean')
                l3, = ax4.plot(plot_age_exceptlast, np.log10(np.mean(np.abs(opt_euler_jd_), axis=0)), 'b-', label='jD, mean')
                lines.append([l1, l2, l3])
                # Calculate and plot max
                l1, = ax4.plot(plot_age_exceptlast, np.log10(np.max(np.abs(opt_euler_h_), axis=0)), 'k--', label='house, max')
                l2, = ax4.plot(plot_age_exceptlast, np.log10(np.max(np.abs(opt_euler_ju_), axis=0)), 'r--', label='jU, max')
                l3, = ax4.plot(plot_age_exceptlast, np.log10(np.max(np.abs(opt_euler_jd_), axis=0)), 'b--', label='jD, max')
                lines.append([l1, l2, l3])
                # Finalize plot
                ax4.set_xlabel('Age')
                ax4.set_ylabel('Euler error [log10]')
                ax4.grid(axis='y')
                legend1 = plt.legend(lines[0], labs2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.legend([l[0] for l in lines], labs1, bbox_to_anchor=(1.05, 0.8), loc='upper left')
                plt.gca().add_artist(legend1)
                plt.savefig(plot_dir + '/' + run_name + '_Euler_Error_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                # CSV
                euler_errors_csv = {
                    'Statistic': ['house', 'jU', 'jD'],
                    'Min': [np.log10(np.min(np.abs(opt_euler_h_))),
                        np.log10(np.min(np.abs(opt_euler_ju_))),
                        np.log10(np.min(np.abs(opt_euler_jd_)))],
                    '25': [np.log10(np.percentile(np.abs(opt_euler_h_), 25)),
                        np.log10(np.percentile(np.abs(opt_euler_ju_), 25)),
                        np.log10(np.percentile(np.abs(opt_euler_jd_), 25))],
                    'Median': [np.log10(np.median(np.abs(opt_euler_h_))),
                        np.log10(np.median(np.abs(opt_euler_ju_))),
                        np.log10(np.median(np.abs(opt_euler_jd_)))],
                    '99': [np.log10(np.percentile(np.abs(opt_euler_h_), 99)),
                        np.log10(np.percentile(np.abs(opt_euler_ju_), 99)),
                        np.log10(np.percentile(np.abs(opt_euler_jd_), 99))],
                    'Mean': [np.log10(np.mean(np.abs(opt_euler_h_))),
                        np.log10(np.mean(np.abs(opt_euler_ju_))),
                        np.log10(np.mean(np.abs(opt_euler_jd_)))],
                    'Max': [np.log10(np.max(np.abs(opt_euler_h_))),
                        np.log10(np.max(np.abs(opt_euler_ju_))),
                        np.log10(np.max(np.abs(opt_euler_jd_)))]}
                euler_errors_df = pd.DataFrame(euler_errors_csv)
                csv_path = os.path.join(plot_dir, f'{run_name}_Euler_Error_Summary_episode{ep}.csv')
                euler_errors_df.to_csv(csv_path, index=False)
                print(f"Euler error summary saved to: {csv_path}")



                
                # Errors: market clearing conditions
                market_clearing_errors = {
                    'Error Type': ['MC_h', 'MC_thetau', 'MC_thetad', 'MC_Consumption'],  # Row identifiers
                    'Min': [np.log10(np.min(np.abs(mc_h_) / 20)),
                        np.log10(np.min(np.abs(mc_thetau_) / 20)),
                        np.log10(np.min(np.abs(mc_thetad_) / 20)),
                        np.log10(np.min(np.abs(mc_cons_) / ebar_))],
                    '25': [np.log10(np.percentile(np.abs(mc_h_) / 20, 25)),
                        np.log10(np.percentile(np.abs(mc_thetau_) / 20, 25)),
                        np.log10(np.percentile(np.abs(mc_thetad_) / 20, 25)),
                        np.log10(np.percentile(np.abs(mc_cons_) / ebar_, 25))],
                    'Median': [np.log10(np.median(np.abs(mc_h_) / 20)),
                        np.log10(np.median(np.abs(mc_thetau_) / 20)),
                        np.log10(np.median(np.abs(mc_thetad_) / 20)),
                        np.log10(np.median(np.abs(mc_cons_) / ebar_))],
                    '99': [np.log10(np.percentile(np.abs(mc_h_) / 20, 99)),
                        np.log10(np.percentile(np.abs(mc_thetau_) / 20, 99)),
                        np.log10(np.percentile(np.abs(mc_thetad_) / 20, 99)),
                        np.log10(np.percentile(np.abs(mc_cons_) / ebar_, 99))],
                    'Mean': [np.log10(np.mean(np.abs(mc_h_) / 20)),
                        np.log10(np.mean(np.abs(mc_thetau_) / 20)),
                        np.log10(np.mean(np.abs(mc_thetad_) / 20)),
                        np.log10(np.mean(np.abs(mc_cons_) / ebar_))],
                    'Max': [np.log10(np.max(np.abs(mc_h_) / 20)),
                        np.log10(np.max(np.abs(mc_thetau_) / 20)),
                        np.log10(np.max(np.abs(mc_thetad_) / 20)),
                        np.log10(np.max(np.abs(mc_cons_) / ebar_))]}

                market_clearing_df = pd.DataFrame(market_clearing_errors)
                csv_path = os.path.join(plot_dir, f'{run_name}_Market_Clearing_Error_Summary_episode{ep}.csv')
                market_clearing_df.to_csv(csv_path, index=False)
                print(f"Market clearing error summary saved to: {csv_path}")

                
                # Cost evolution
                plt.figure(figsize=std_figsize)
                ax6 = plt.subplot(1,1,1)
                ax6.plot(np.arange(load_episode, ep+1), np.log10(cost_store[0:ep-load_episode+1]), 'k-', label = 'evolution')
                ax6.plot(np.arange(load_episode, ep+1), np.log10(mov_ave_cost_store[0:ep-load_episode+1]), 'r--', label = 'moving mean')
                ax6.set_xlabel('Episode')
                ax6.set_ylabel('log10(cost)')
                ax6.grid(axis='y')
                plt.savefig(plot_dir + '/' + run_name + '_cost_episode' + str(ep)+'.pdf', bbox_inches='tight')
                plt.close()

                cost_dict = {'x': np.arange(load_episode, ep+1).tolist(), 'y': cost_store[0:ep-load_episode+1].tolist()}
                plot_dict['cost'] = cost_dict
                
                if ep - load_episode > 1100:

                    plt.figure(figsize=std_figsize)
                    ax6 = plt.subplot(1,1,1)
                    ax6.plot(np.arange(ep+1 - 1000, ep+1), np.log10(cost_store[ep-load_episode+1 - 1000:ep-load_episode+1]), 'k-', label = 'evolution')
                    ax6.plot(np.arange(ep+1 - 1000, ep+1), np.log10(mov_ave_cost_store[ep-load_episode+1 - 1000:ep-load_episode+1]), 'r-', label = 'moving mean')
                    ax6.set_xlabel('Episode')
                    ax6.set_ylabel('log10(cost)')
                    ax6.grid(axis='y')
                    plt.legend()
                    plt.savefig(plot_dir + '/' + run_name + '_costLAST_episode' + str(ep)+'.pdf', bbox_inches='tight')
                    plt.close()

                    costLAST_dict = {'x': np.arange(ep+1 - 1000, ep+1).tolist(), 'y': cost_store[ep-load_episode+1 - 1000:ep-load_episode+1].tolist()}
                    plot_dict['costLAST'] = costLAST_dict
                
                

                
                # Save data from simulation
                saver = tf.train.Saver(nn.param_dict)
                save_param_path = save_base_path + '/model/' + run_name + '-episode' + str(ep)
                saver.save(sess, save_param_path)
                print('Model saved in path: %s' % save_param_path)
                save_data_path = save_base_path + '/model/' + run_name + '-episode' + str(ep) + '_LastData.npy'
                np.save(save_data_path, X_data_train)
                print('Last points saved at: %s' % save_data_path)
                save_full_episode_path = save_base_path + '/model/' + 'X_episodes.npy'
                np.save(save_full_episode_path, X_episodes)
                print('Full episodes saved at: %s' % save_full_episode_path)
                save_c_path = save_base_path + '/model/' +'c.npy'
                np.save(save_c_path, c_)
                print('Consumption data saved at: %s' % save_c_path)
                save_h_path = save_base_path + '/model/' + 'h.npy'
                np.save(save_h_path, h_)
                print('Housing data saved at: %s' % save_h_path)
                save_x_prime_U_path = save_base_path + '/model/' + 'x_prime_U.npy'
                np.save(save_x_prime_U_path, x_prime_U_)
                print('x_prime_U data saved at: %s' % save_x_prime_U_path)
                save_x_prime_D_path = save_base_path + '/model/' + 'x_prime_D.npy'
                np.save(save_x_prime_D_path, x_prime_D_)
                print('x_prime_D data saved at: %s' % save_x_prime_D_path)
                mean_LTV_U_economy1 = np.mean(LTVj_matrix_[a_condU, :], axis=0)
                save_LTV_U_path = save_base_path + '/model/' + 'mean_LTV_U_economy1.npy'
                np.save(save_LTV_U_path, mean_LTV_U_economy1)
                print('Mean LTV U data saved at: %s' % save_LTV_U_path)
                mean_R_U_economy1 = np.mean(Rj_matrix_[a_condU, :], axis=0)
                save_R_U_path = save_base_path + '/model/' + 'mean_R_U_economy1.npy'
                np.save(save_R_U_path, mean_R_U_economy1)
                print('Mean R U data saved at: %s' % save_R_U_path)
                mean_LTV_D_economy1 = np.mean(LTVj_matrix_[a_condD, :], axis=0)
                save_LTV_D_path = save_base_path + '/model/' + 'mean_LTV_D_economy1.npy'
                np.save(save_LTV_D_path, mean_LTV_D_economy1)
                print('Mean LTV D data saved at: %s' % save_LTV_D_path)
                mean_R_D_economy1 = np.mean(Rj_matrix_[a_condD, :], axis=0)
                save_R_D_path = save_base_path + '/model/' + 'mean_R_D_economy1.npy'
                np.save(save_R_D_path, mean_R_D_economy1)
                print('Mean R D data saved at: %s' % save_R_D_path)
                mean_thetau = np.mean(thetau_, axis=0)
                save_thetau_path = save_base_path + '/model/' + 'mean_thetau.npy'
                np.save(save_thetau_path, mean_thetau)
                print('Mean thetau data saved at: %s' % save_thetau_path)
                mean_thetad = np.mean(thetad_, axis=0)
                save_thetad_path = save_base_path + '/model/' + 'mean_thetad.npy'
                np.save(save_thetad_path, mean_thetad)
                print('Mean thetad data saved at: %s' % save_thetad_path)
                mean_h = np.mean(h_, axis=0)
                save_h_path = save_base_path + '/model/' + 'mean_h.npy'
                np.save(save_h_path, mean_h)
                print('Mean housing data saved at: %s' % save_h_path)
                mean_theta = np.mean(theta_, axis=0)
                save_theta_path = save_base_path + '/model/' + 'mean_theta.npy'
                np.save(save_theta_path, mean_theta)
                print('Mean theta data saved at: %s' % save_theta_path)
                c_prime_U_ = sess.run(c_prime_U, feed_dict={X: X_episodes})
                save_c_prime_U_path = save_base_path + '/model/' + 'c_prime_U.npy'
                np.save(save_c_prime_U_path, c_prime_U_)
                print('c_prime_U data saved at: %s' % save_c_prime_U_path)
                c_prime_D_ = sess.run(c_prime_D, feed_dict={X: X_episodes})
                save_c_prime_D_path = save_base_path + '/model/' + 'c_prime_D.npy'
                np.save(save_c_prime_D_path, c_prime_D_)
                print('c_prime_D data saved at: %s' % save_c_prime_D_path)

        params_dict = sess.run(nn.param_dict)
        for param_key in params_dict:
            params_dict[param_key] = params_dict[param_key].tolist()
        train_dict['params'] = params_dict
        result_dict['cost'] = cost_store.tolist()
        result_dict['time'] = time_store.tolist()
        train_dict['results'] = result_dict
        end_time = datetime.now()
        print('Optimization Finished!')
        print('end time: {}'.format(end_time))
        print('total training time: {}'.format(end_time - start_time))

        train_writer.close()

        return train_dict

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_from_scratch', dest='load_flag', action='store_false')
    parser.set_defaults(load_flag=True)
    args = parser.parse_args()
    load_flag = args.load_flag

    print('##### input arguments #####')
    seed = 0
    path_wd = '.'
    run_name = 'restart_baseline' if args.load_flag else '1st_baseline'
    num_agents = 20
    num_hidden_nodes = [500, 300]
    activations_hidden_nodes = [tf.nn.relu, tf.nn.relu]
    optimizer = 'adam'
    batch_size = 64
    num_episodes = 1
    len_episodes = 10000
    epochs_per_episode = 1
    save_interval = 100
    lr = 1e-5
    load_run_name = 'final_baseline' if args.load_flag else None
    load_episode = 109100 if args.load_flag else 1

    # For the 2nd training schedule, replace load_episode with desired starting point and uncommend the following:
    #################################################################
    '''batch_size = 1024
    num_episodes = 1
    lr = 1e-6
    run_name = '2nd_baseline'
    load_flag = True
    load_run_name = '1st_baseline'
    load_episode = 60000'''
    #################################################################

    print('seed: {}'.format(seed))
    print('working directory: ' + path_wd)
    print('run_name: {}'.format(run_name))
    print('num_agents: {}'.format(num_agents))
    print('hidden nodes: [500, 300]')
    print('activation hidden nodes: [relu, relu]')

    if args.load_flag:
        train_flag = False
        num_episodes = 1
        print('loading weights from final_baseline')
        print('loading from episode {}'.format(load_episode))
    else:
        train_flag = True
        print('optimizer: {}'.format(optimizer))
        print('batch_size: {}'.format(batch_size))
        print('num_episodes: {}'.format(num_episodes))
        print('len_episodes: {}'.format(len_episodes))
        print('epochs_per_episode: {}'.format(epochs_per_episode))
        print('save_interval: {}'.format(save_interval))
        print('lr: {}'.format(lr))

    print('###########################')

    train_dict = train(path_wd, run_name, num_agents,
                       num_episodes, len_episodes, epochs_per_episode,
                       batch_size, optimizer, lr,
                       save_interval, num_hidden_nodes,
                       activations_hidden_nodes, train_flag=train_flag,
                       load_flag=load_flag, load_run_name=load_run_name,
                       load_episode=load_episode, seed=seed)

    # Save outputs
    train_dict['net_setup']['activations_hidden_nodes'] = ['relu', 'relu']
    save_train_dict_path = os.path.join('.', 'output', run_name, 'json', 'train_dict.json')
    json.dump(train_dict, codecs.open(save_train_dict_path, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)
    print('Saved dictionary to:' + save_train_dict_path)

if __name__ == '__main__':
    main()
