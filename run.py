import os
import pickle
import argparse
import jax

import numpy as np
import netket as nk 
from netket.operator.spin import sigmax, sigmaz
import matplotlib.pylab as plt

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
parser.add_argument('--weight_decay', type=int, default=0,
                    help='decay weights using l2 normalization during training')
parser.add_argument('--bs_power', type=int, default=6,
                    help='batch size = 2**bs_power')
parser.add_argument('--data_split', type=str, default='random',
                    help='how to split the data set')
parser.add_argument('--load_saved_data', type=int, default=1,
                    help='load pre-saved dataset so that all seeds use the same data')
parser.add_argument('--frac_data_in_test', type=float, default=0.25,
                    help='fraction of data in the test set')
parser.add_argument('--num_high_prob_in_test', type=int, default=0,
                    help='how many high probability samples are in the test set')
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
weight_decay = (args.weight_decay==1)
cost_fxn = get_cost_fxn(cost_fxn_name)
how_to_split_data = args.data_split
frac_in_test = args.frac_data_in_test
load_saved_data = (args.load_saved_data==1)
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

print(f"Using test and train datasets split using {how_to_split_data}")
print(f"Using {cost_fxn_name} cost function and training for {num_epochs} epochs")
if weight_decay:
    print(f"Decaying weights during training")

SAVE = True
SAVE_WEIGHTS = (args.save_weights==1)
PLOT = True
TRAIN_PRINT = True
base_path = f'/mnt/home/smoss/ceph/nqsLandscape/Supervised/'
exp_path = base_path + f'{exp_name}/'
system_path = f'IsingChain_h{h}_N{N}/'

path = exp_path + system_path + f'TestTrain/testSize{frac_in_test}_{how_to_split_data}/{cost_fxn_name}/'
weights_path = base_path + f'weights/{exp_name}/' + system_path + f'TestTrain/testSize{frac_in_test}_{how_to_split_data}/{cost_fxn_name}/'
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

if load_saved_data:
    print("load saved data sets (make new if needed).")
    if how_to_split_data == 'random':
        num_high_prob_samples_in_test = args.num_high_prob_in_test
        data_path = base_path + 'training_data/' + system_path + f'Test{frac_in_test}/{how_to_split_data}_{num_high_prob_samples_in_test}inTest'
    else:
        data_path = base_path + 'training_data/' + system_path + f'Test{frac_in_test}/{how_to_split_data}'
    x_train, x_test, y_train, y_test, all_target_logpsis = load_data(data_path, h, gs, how_to_split=how_to_split_data, test_frac=frac_in_test)
else:
    print("using a different data set for every seed... making new data set.")
    data_PRNG = jax.random.PRNGKey(seed)
    x_train, x_test, y_train, y_test, all_target_logpsis = load_random_data(h, gs, data_PRNG, how_to_split_data, frac_in_test, path, halfHilbert=False)

if how_to_split_data=='mc':
    print(2**N - len(y_test))
    np.save(path+'actual_test_size',len(y_test))

if not os.path.exists(path + 'train_losses.pkl'):
    train_losses_dict = {}
    test_losses_dict = {}
    fidelities_dict = {}
else:
    with open(path + 'train_losses.pkl', 'rb') as f:
        train_losses_dict = pickle.load(f)
    with open(path + 'test_losses.pkl', 'rb') as f:
        test_losses_dict = pickle.load(f)
    with open(path + 'fidelities.pkl', 'rb') as f:
        fidelities_dict = pickle.load(f)

for i, (depth,width) in enumerate(features):
    if features[i] not in train_losses_dict.keys():
        print(f'\nNN size: {depth,width}')
        nn_features = [width]*depth
        train_losses, test_losses, fidelities, final_params, opt_state = train_w_test(nn_features, num_epochs, batch_size, 
                                                            cost_fxn, 
                                                            x_train, y_train, x_test, y_test,
                                                            all_target_logpsis, 
                                                            weight_decay=weight_decay,
                                                            key=seed, lr=lr, decay=decay,
                                                            rescale=False, PRINT=TRAIN_PRINT)
        train_losses_dict[features[i]] = train_losses
        test_losses_dict[features[i]] = test_losses
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
            with open(path + "test_losses.pkl", "wb") as f:
                pickle.dump(test_losses_dict,f)
            with open(path + "fidelities.pkl", "wb") as f:
                pickle.dump(fidelities_dict,f)
    else:
        print(f"NQS with size {features[i]} already trained!")

    if PLOT:
        plot_training_summary(features, cost_fxn_name, path, train_losses_dict, fidelities_dict, test_losses_dict)
    
print("Done!")