# Download source data from https://www.kamishima.net/sushi/ (sushi3-2016.zip)
# and put sushi3.idata and sushi3a.5000.10.order in ../data/sushi_dectet_data

# Process as in preprocessing.py save_sushi_data (but here keep original 
# orderings and don't split into training and test)
import numpy as np
import pandas as pd
from utils import *
from scipy.sparse import save_npz, load_npz

dir = 'sushi_dectet_'

n_sushi = 10
# load the feature matrix
sushi_i = np.genfromtxt('../data/' + dir + 'data/' + 'sushi3.idata')
# read in with item names, find row indices of the 10 items 
# (as listed in README-en.txt in data sushi3-2016.zip) and subset sushi_i to match
items = pd.read_csv('../data/' + dir + 'data/' + 'sushi3.idata', sep="\t", header=None)
idx = items.index[items[1].isin(['ebi', 'anago', 'maguro', 'ika', 'uni', 'ikura', 
                                 'tamago', 'toro', 'tekka_maki', 'kappa_maki'])]
sushi_i = sushi_i[idx,]
# continue to process as in preprocessing.py save_sushi_data()
num_sushi = sushi_i.shape[0]
feature_matrix = np.zeros([num_sushi, 6])
feature_matrix[:, 0:2] = sushi_i[:, 2:4]
feature_matrix[:, 2:6] = sushi_i[:, 5:9]
# skip categorical variable - not enough rankings here to estimate effect
# load the orderings of 10 items
sushi_order_file = np.genfromtxt('../data/' + dir + 'data/' + 'sushi3a.5000.10.order', skip_header=1)
# extract just the rankings
rankings = np.array(sushi_order_file)[:, 2:].astype(int)
np.save('../data/' + dir + 'data/' + 'features', feature_matrix)
np.save('../data/' + dir + 'data/' + 'rankings', rankings)
n = feature_matrix.shape[0]
mat_Pij = est_Pij(n, np.array(rankings))
save_npz('../data/' + dir + 'data/' + 'mat_Pij', mat_Pij)
