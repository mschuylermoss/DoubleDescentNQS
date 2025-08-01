import os
import pickle
import argparse

import numpy as np
import netket as nk 
from netket.operator.spin import sigmax, sigmaz
import matplotlib.pylab as plt
from itertools import product

from cost_functions import get_cost_fxn
from helpers import try_load_dict
from plot import plot_quantities_vs_test_size, plot_min_parameters_for_fidelity_thresh

#--
parser = argparse.ArgumentParser(description='Parser for training')
parser.add_argument('--exp_name', type=str, default='test',
                    help='experiment name')
parser.add_argument('--N', type=int, default=12,
                    help='number of spins')
parser.add_argument('--h', type=float, default=-1.0,
                    help='strength of transverse field')
parser.add_argument('--bs_power', type=int, default=6,
                    help='batch size = 2**bs_power')
parser.add_argument('--cost', type=str, default='hellinger',
                    help='cost function')
parser.add_argument('--data_split', type=str, default='random',
                    help='how to split the data set')
args = parser.parse_args()
#--

exp_name = args.exp_name
N = args.N
h = args.h
batch_size = 2**args.bs_power
cost_fxn_name = args.cost
cost_fxn = get_cost_fxn(cost_fxn_name)
how_to_split_data = args.data_split

fracs = np.arange(0.0,1,0.05)

fidelity_threshold = True
if fidelity_threshold:
    fidelity_threshold_vals = {'-1.0': 10**-4, 
                               '-5.0': 10**-4}
    threshold = fidelity_threshold_vals[f'{h}']
    file = f"min_num_params_for_fid_thresh_{threshold}.npy"
else:
    which_best = 'train_losses'
    file = f"num_params_for_best_{which_best}.npy"

PLOT = True
SAVE = True
base_path = f'./data/{exp_name}/IsingChain_h{h}_N{N}/'

final_train_losses = {}
final_test_losses = {}
final_fidelities = {}

if not os.path.exists(base_path + file):
    for frac in fracs:
        frac = np.round(frac,2)
        if frac ==fracs[0]:
            fracs_found = []
            num_params_for_best = []
            min_num_params_for_fid_thresh = []
            fidelities_for_best = []
            fidelities_var_for_best = []
            train_losses_for_best = []
            train_losses_var_for_best = []
            test_losses_for_best = []
            test_losses_var_for_best = []

        path = base_path + f'testSize{frac}/' + f'TestTrain_{how_to_split_data}/{cost_fxn_name}/bs_{batch_size}/'
        if not os.path.exists(path + '/final_fidelities.npy'):
            print(path + '/final_fidelities.npy')
            print("----- FILE NOT FOUND ------")
            # raise FileNotFoundError("Data does not exist, cannot analyze!")
        else:
            params_Hilbert_ratio = np.load(path+'params_Hilbert_ratio.npy')
            final_train_losses = np.load(path+'final_train_losses.npy')
            final_train_losses_var = np.load(path+'final_train_losses_var.npy')
            final_test_losses = np.load(path+'final_test_losses.npy')
            final_test_losses_var = np.load(path+'final_test_losses_var.npy')
            final_fidelities = np.load(path+'final_fidelities.npy')
            final_fidelities_var = np.load(path+'final_fidelities_var.npy')
            print(f"Loaded data for frac in test = {frac}!")
        
            fracs_found.append(frac)
            if fidelity_threshold: 
                idcs = np.where(final_fidelities > (1-threshold))[0]
                if len(idcs) > 0:
                    min_num_params_for_fid_thresh.append(min(params_Hilbert_ratio[idcs]))
            else:
                if which_best == 'fidelities':
                    best = np.max(final_fidelities)
                    where_best = np.where(final_fidelities == best)
                elif which_best == 'train_losses':
                    best = np.min(final_train_losses)
                    where_best = np.where(final_train_losses == best)
                num_params_for_best.append(params_Hilbert_ratio[where_best])
                fidelities_for_best.append(final_fidelities[where_best])
                fidelities_var_for_best.append(final_fidelities_var[where_best])
                train_losses_for_best.append(final_train_losses[where_best])
                train_losses_var_for_best.append(final_train_losses_var[where_best])
                test_losses_for_best.append(final_test_losses[where_best])
                test_losses_var_for_best.append(final_test_losses_var[where_best])

    if SAVE:
        np.save(base_path + "fracs_found", fracs_found)
        if fidelity_threshold:
            np.save(base_path + f"min_num_params_for_fid_thresh_{threshold}", min_num_params_for_fid_thresh)
        else:
            np.save(base_path + f"num_params_for_best_{which_best}", num_params_for_best)
            np.save(base_path + f"fidelities_for_best_{which_best}", fidelities_for_best)
            np.save(base_path + f"fidelities_var_for_best_{which_best}", fidelities_var_for_best)
            np.save(base_path + f"train_losses_for_best_{which_best}", train_losses_for_best)
            np.save(base_path + f"train_losses_var_for_best_{which_best}", train_losses_var_for_best)
            np.save(base_path + f"test_losses_for_best_{which_best}", test_losses_for_best)
            np.save(base_path + f"test_losses_var_for_best_{which_best}", test_losses_var_for_best)

else:
    fracs_found = np.load(base_path + "fracs_found.npy")
    if fidelity_threshold:
        min_num_params_for_fid_thresh = np.load(base_path + f"min_num_params_for_fid_thresh_{threshold}.npy")
    else:
        num_params_for_best =  np.load(base_path + f"num_params_for_best_{which_best}.npy")
        fidelities_for_best =   np.load(base_path + f"fidelities_for_best_{which_best}.npy")
        fidelities_var_for_best =    np.load(base_path + f"fidelities_var_for_best_{which_best}.npy")
        train_losses_for_best =    np.load(base_path + f"train_losses_for_best_{which_best}.npy")
        train_losses_var_for_best =    np.load(base_path + f"train_losses_var_for_best_{which_best}.npy")
        test_losses_for_best =    np.load(base_path + f"test_losses_for_best_{which_best}.npy")
        test_losses_var_for_best =    np.load(base_path + f"test_losses_var_for_best_{which_best}.npy")

if PLOT:
    if fidelity_threshold:
        plot_min_parameters_for_fidelity_thresh(threshold, base_path, fracs_found, min_num_params_for_fid_thresh)
    else:
        plot_quantities_vs_test_size(which_best, fracs_found,fidelities_for_best,base_path,
                                    train_losses_for_best,test_losses_for_best,
                                    num_params_for_best)    

print("Done!")