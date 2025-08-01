import os
import pickle
import argparse
import time 
import jax
import numpy as np
import netket as nk 
from netket.operator.spin import sigmax, sigmaz
import matplotlib.pylab as plt
from itertools import product

from cost_functions import get_cost_fxn
from dset_helpers import make_data, load_data, load_random_data
from helpers import get_features, get_num_params, try_load_dict
from plot import plot_training_summary
from train import train_on_all_configs, train_w_test

#--
parser = argparse.ArgumentParser(description='Parser for training')
parser.add_argument('--exp_name', type=str, default='test',
                    help='experiment name')
parser.add_argument('--N', type=int, default=12,
                    help='number of spins')
parser.add_argument('--h', type=float, default=-1.0,
                    help='strength of transverse field')
parser.add_argument('--cost', type=str, default='hellinger',
                    help='cost function')
parser.add_argument('--bs_power', type=int, default=6,
                    help='batch size = 2**bs_power')
parser.add_argument('--seed', type=int, default=100,
                    help='seed for reproducibility')
parser.add_argument('--save_weights',type=int,default=0,
                    help='boolean for whether or not to write the weights to memory')
args = parser.parse_args()
#--


exp_name = args.exp_name
N = args.N
h = args.h
batch_size = 2**args.bs_power
cost_fxn_name = args.cost
cost_fxn = get_cost_fxn(cost_fxn_name)
seed = args.seed
lr = 0.001
decay = 0.99
num_epochs = 15000
features = get_features(N,extra_large=True)

print(f"\nTraining {len(features)} NQS with features given by (depth, width): \n{features}")
num_params_dict = {} 
for i, (depth,width) in enumerate(features):
    num_params_dict[features[i]] = get_num_params(N,depth=depth,width=width)
    print(f"Network with shape {(depth,width)} has {num_params_dict[features[i]]} parameters")
    print(f"#params/Hilbert size = {num_params_dict[features[i]]/(2**N)}")

print(f"Training with all configs in the Hilbert space")
print(f"Using {cost_fxn_name} cost function and training for {num_epochs} epochs")

SAVE = True
SAVE_WEIGHTS = (args.save_weights==1)
PLOT = True
TRAIN_PRINT = True
base_path = f'/mnt/home/smoss/ceph/nqsLandscape/Supervised/'
system_path = f'{exp_name}/IsingChain_h{h}_N{N}/'

path = base_path + system_path +  f'FullEnum/{cost_fxn_name}/'
weights_path = base_path + f'weights/' + system_path + f'FullEnum/{cost_fxn_name}/'
if batch_size != 2**N:
    path += f'bs_{batch_size}/'
    weights_path += f'bs_{batch_size}/'
else:
    path += 'not_batched/'
    weights_path += 'not_batched/'

path += f'seed_{seed}/'
weights_path += f'seed_{seed}/'
if not os.path.exists(path):
    os.makedirs(path)
if SAVE_WEIGHTS:
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

# Hamiltonian & Exact Energy
graph = nk.graph.Chain(length=N, pbc=True)
hi = nk.hilbert.Spin(s=0.5,N=N)
ha = sum([h*sigmax(hi,i) for i in range(N)])
ha += sum([-sigmaz(hi,i)*sigmaz(hi,(i+1)%N) for i in range(N)])
E_gs, gs = nk.exact.lanczos_ed(ha,compute_eigenvectors=True)

# Get data
data_key = jax.random.PRNGKey(seed)
x_train, _, y_train, _, all_target_logpsis = make_data(h, gs, data_key, how_to_split='dont')
if cost_fxn_name=='infidelity':
    y_train = all_target_logpsis
    batch_size = 2**N

if not os.path.exists(path + 'train_losses.pkl'): # change this so that save after every NN
    print(f"No files found in {path}")
    train_losses_dict = {}
    fidelities_dict = {} 
else:
    print("Loading previous runs")
    with open(path + 'train_losses.pkl', 'rb') as f:
        train_losses_dict = pickle.load(f)
    with open(path + 'fidelities.pkl', 'rb') as f:
        fidelities_dict = pickle.load(f)

for i, (depth,width) in enumerate(features):
    if features[i] not in train_losses_dict.keys():
        print(f'\nNN size: {depth,width}')
        nn_features = [width]*depth
        train_start = time.time()
        train_losses, fidelities, final_params, opt_state = train_on_all_configs(nn_features, num_epochs, batch_size, 
                                                        cost_fxn, x_train, y_train, all_target_logpsis,
                                                        key=seed, lr=lr, decay=decay,
                                                        PRINT=TRAIN_PRINT)
        print(f"{(time.time()-train_start)/60} minutes of training")
        train_losses_dict[features[i]] = train_losses
        fidelities_dict[features[i]] = fidelities
        if SAVE:
            if SAVE_WEIGHTS:
                with open(weights_path + f"depth{depth}_width{width}.pkl", "wb") as f:
                    pickle.dump(final_params,f)
                with open(weights_path + f"opt_depth{depth}_width{width}.pkl", "wb") as f:
                    pickle.dump(opt_state,f)
            with open(path + "num_params.pkl", "wb") as f:
                pickle.dump(num_params_dict,f)
            with open(path + "train_losses.pkl", "wb") as f:
                pickle.dump(train_losses_dict,f)
            with open(path + "fidelities.pkl", "wb") as f:
                pickle.dump(fidelities_dict,f)
    else:
        print(f"NQS with size {features[i]} already trained!")

if PLOT:
    plot_training_summary(features, cost_fxn_name, path, train_losses_dict, fidelities_dict)
