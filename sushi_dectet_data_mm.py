# Fit standard Plackett-Luce model to triplet sushi data via MM

## Global variables set in utils.py: epsilon, rtol, n_iter

## load data and initialize MM method as in run_methods: init_all_methods_real_data

import numpy as np
import cvxpy as cp
from scipy.sparse import save_npz, load_npz
from utils import *

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
### mm parameters
log_dict['mm_conv'] = False
log_dict['pi_mm'] = np.copy(pi_init)
log_dict['diff_pi_mm'] = [np.linalg.norm(log_dict['pi_mm'])]
log_dict['obj_mm'] = [objective(log_dict['pi_mm'], rankings_train)]
log_dict['iter_mm'] = 0

## Run MM method and save logged results

from only_scores import *
import pickle

n = X.shape[0]  # number of items
for iter in range(n_iter):
    # mm update
    if not log_dict['mm_conv']:
        log_dict['pi_mm_prev'] = log_dict['pi_mm']
        log_dict['pi_mm'], time_mm_iter = mm_iter(n, rankings_train, weights=log_dict['pi_mm'])
        if np.any(np.isnan(log_dict['pi_mm'])):
            log_dict['mm_conv'] = True
        else:
            log_dict['diff_pi_mm'].append(np.linalg.norm(log_dict['pi_mm_prev'] - log_dict['pi_mm']))
            log_dict['obj_mm'].append(objective(log_dict['pi_mm'], rankings_train))
            log_dict['iter_mm'] += 1
            log_dict['mm_conv'] = np.linalg.norm(log_dict['pi_mm_prev'] - log_dict['pi_mm']) < rtol * np.linalg.norm(log_dict['pi_mm'])
     # stop if converged
    if log_dict['mm_conv']:
        break

# Save results as a csv file
save_name = 'mm'
with open('../results/' + dir + 'data/' + '_logs_' + save_name + '.pickle', "wb") as pickle_out:
    pickle.dump(log_dict, pickle_out)
    pickle_out.close()

# read data

import pandas as pd

object = pd.read_pickle(r'../results/' + dir + 'data/' + '_logs_' + save_name + '.pickle')
object["mm_conv"]
np.round_(object["pi_mm"], 3)
