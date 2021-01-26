# Fit standard Plackett-Luce model to triplet sushi data via MM

## Global variables set in utils.py: epsilon, rtol, n_iter

## load data and initialize PLADMM method as in run_methods: init_all_methods_real_data

import numpy as np
import cvxpy as cp
from scipy.sparse import save_npz, load_npz
from utils import *
from admm_log import *

dir = 'sushi_dectet_'

### load data
rankings_train = np.load('../data/' + dir + 'data/' + 'rankings.npy')
X = np.load('../data/' + dir + 'data/' + 'features.npy').astype(float)
mat_Pij = load_npz('../data/' + dir + 'data/' + 'mat_Pij.npz')
endog = rankings_train[:, 0]
exog = rankings_train
### Initialization, start from a feasible point for all parameters
(beta_init, b_init, time_beta_b_init), (pi_init, time_pi_init), (u_init, time_u_init), \
    (theta_init, time_theta_init), (exp_beta_init, time_exp_beta_init) = \
        init_params(X, rankings_train, mat_Pij, method_beta_b_init='QP')
### Log all results
log_dict = dict()
### log admm parameters
log_dict['log_admm'] = ADMM_log(rankings_train, X, method_pi_tilde_init='prev')
log_dict['log_admm_conv'] = False
log_dict['beta_log_admm'] = np.copy(exp_beta_init)
log_dict['pi_log_admm'] = softmax(np.dot(X, log_dict['beta_log_admm']))
log_dict['u_log_admm'] = np.copy(u_init)
log_dict['time_log_admm'] = [time_exp_beta_init + time_u_init]
log_dict['diff_pi_log_admm'] = [np.linalg.norm(log_dict['pi_log_admm'])]
log_dict['diff_beta_log_admm'] = [np.linalg.norm(log_dict['beta_log_admm'])]
log_dict['prim_feas_log_admm'] = [np.linalg.norm(np.dot(X, log_dict['beta_log_admm']) - np.log(log_dict['pi_log_admm'] + epsilon))]
log_dict['dual_feas_log_admm'] = [np.linalg.norm(np.dot(log_dict['log_admm'].X.T, np.log(log_dict['pi_log_admm'] + epsilon)))]
log_dict['obj_log_admm'] = [objective(log_dict['pi_log_admm'], rankings_train)]
log_dict['iter_log_admm'] = 0
    

## Run PLADMMM method and save logged results

import pickle

rho = 1

n = X.shape[0]  # number of items
for iter in range(n_iter):
    # log_admm update
    if not log_dict['log_admm_conv']:
        log_dict['pi_log_admm_prev'] = log_dict['pi_log_admm']
        log_dict['beta_log_admm_prev'] = log_dict['beta_log_admm']
        log_dict['tilde_pi_log_admm_prev'] = softmax(np.dot(X, log_dict['beta_log_admm']))
        log_dict['pi_log_admm'], log_dict['beta_log_admm'], log_dict['u_log_admm'], time_log_admm_iter = \
                log_dict['log_admm'].fit_log(rho, weights=log_dict['pi_log_admm'], beta=log_dict['beta_log_admm'], u=log_dict['u_log_admm'])
        # scores predicted by beta
        log_dict['tilde_pi_log_admm'] = softmax(np.dot(X, log_dict['beta_log_admm']))
        log_dict['time_log_admm'].append(time_log_admm_iter)
        log_dict['diff_pi_log_admm'].append(np.linalg.norm(log_dict['pi_log_admm_prev'] - log_dict['pi_log_admm']))
        log_dict['diff_beta_log_admm'].append(np.linalg.norm(log_dict['beta_log_admm_prev'] - log_dict['beta_log_admm']))
        log_dict['prim_feas_log_admm'].append(np.linalg.norm(np.dot(X, log_dict['beta_log_admm']) - np.log(log_dict['pi_log_admm'] + epsilon)))
        log_dict['dual_feas_log_admm'].append(np.linalg.norm(np.dot(log_dict['log_admm'].X.T, np.log(log_dict['pi_log_admm_prev'] + epsilon) - np.log(log_dict['pi_log_admm'] + epsilon))))
        log_dict['obj_log_admm'].append(objective(log_dict['pi_log_admm'], rankings_train))
        log_dict['iter_log_admm'] += 1
        log_dict['log_admm_conv'] = np.linalg.norm(
                log_dict['pi_log_admm_prev'] - log_dict['pi_log_admm']) < rtol * np.linalg.norm(log_dict['pi_log_admm']) \
                and np.linalg.norm(log_dict['tilde_pi_log_admm_prev'] - log_dict['tilde_pi_log_admm']) < rtol * np.linalg.norm(
                log_dict['tilde_pi_log_admm'])
     # stop if converged
    if log_dict['log_admm_conv']:
        break
        
# Correct time scale
log_dict['time_cont_log_admm'] = [sum(log_dict['time_log_admm'][:ind + 1]) for ind in range(len(log_dict['time_log_admm']))]

# Save results as a csv file
save_name = 'pladmm'
with open('../results/' + dir + 'data/' + '_logs_' + save_name + '.pickle', "wb") as pickle_out:
    pickle.dump(log_dict, pickle_out)
    pickle_out.close()

# Read results and examine

import pandas as  pd

object = pd.read_pickle(r'../results/' + dir + 'data/' + '_logs_' + save_name + '.pickle')
object.keys()
object["log_admm_conv"]
np.round_(object["beta_log_admm"], 3) # coefficients
np.round_(object["pi_log_admm"], 3) # implied worth = worth from standard PL model
