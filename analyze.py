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
from plot import plot_double_descent, plot_double_descent_variances

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
parser.add_argument('--frac_data_in_test', type=float, default=0.25,
                    help='fraction of data in the test set')
parser.add_argument('--num_pairs', type=int, default=0,
                    help='number of parity pairs in training dataset')
parser.add_argument('--seed', type=int, default=-1,
                    help='seed if we just want to check one')
args = parser.parse_args()
#--

exp_name = args.exp_name
N = args.N
h = args.h
batch_size = 2**args.bs_power
cost_fxn_name = args.cost
cost_fxn = get_cost_fxn(cost_fxn_name)
how_to_split_data = args.data_split

if args.seed == -1:
    seeds = [10,20,30,40,50,60,70,80,90,100]
else:
    seeds = [args.seed]
PLOT = True
SAVE = True
base_path = f'/mnt/home/smoss/ceph/nqsLandscape/Supervised/{exp_name}/'
system_path = f'IsingChain_h{h}_N{N}/'

final_train_losses = {}
final_test_losses = {}
final_fidelities = {}

if how_to_split_data == 'dont':

    path = base_path + system_path + f'FullEnum/{cost_fxn_name}/'
    # batch_size = 2**N
    if batch_size != 2**N:
        path += f'bs_{batch_size}/'
    else:
        path += f'not_batched/'

    if not os.path.exists(path):
        os.makedirs(path)

    for seed in seeds:
        seed_path = path + f'seed_{seed}/'
        if len(seeds)==1:
            path = seed_path 
            
        if not os.path.exists(seed_path + 'train_losses.pkl'):
            print(seed_path + 'train_losses.pkl')
            print("----- FILE NOT FOUND, KEEP GOING ------")
            # raise FileNotFoundError("Data does not exist, cannot analyze!")
        else:
            num_params_dict = try_load_dict(seed_path + 'num_params.pkl')
            train_losses_dict = try_load_dict(seed_path + 'train_losses.pkl')
            fidelities_dict = try_load_dict(seed_path + 'fidelities.pkl')
            print(f"Loaded data for seed {seed}!")
            if seed ==seeds[0]:
                for feat_dim in list(num_params_dict.keys()):
                    final_train_losses[feat_dim] = []
                    final_fidelities[feat_dim] = []

        x_ = []
        for feat_dim in list(num_params_dict.keys()):
            if feat_dim in list(train_losses_dict.keys()):
                (depth,width) = feat_dim
                num_params = num_params_dict[feat_dim]
                x_.append(num_params/2**N)
                final_train_loss = train_losses_dict[feat_dim][-1]
                final_fidelity = fidelities_dict[feat_dim][-1]
                if (width>432):
                    if (final_train_loss < 0.7):
                        final_train_losses[feat_dim].append(final_train_loss)
                        final_fidelities[feat_dim].append(final_fidelity)
                    else:
                        print(f"{width}: did not train!")
                else:
                    final_train_losses[feat_dim].append(final_train_loss)
                    final_fidelities[feat_dim].append(final_fidelity)

    final_train_losses_avgd = []
    final_train_losses_var = []
    final_fidelities_avgd = []
    final_fidelities_var = []
    for feat_dim in list(num_params_dict.keys()):
        if feat_dim in list(train_losses_dict.keys()):
            final_train_losses_avgd.append(np.nanmean(final_train_losses[feat_dim]))
            final_train_losses_var.append(np.nanvar(final_train_losses[feat_dim]))
            final_fidelities_avgd.append(np.nanmean(final_fidelities[feat_dim]))
            final_fidelities_var.append(np.nanvar(final_fidelities[feat_dim]))

    if SAVE:
        np.save(path + "params_Hilbert_ratio", x_)
        np.save(path + "final_train_losses", final_train_losses_avgd)
        np.save(path + "final_train_losses_var", final_train_losses_var)
        np.save(path + "final_fidelities", final_fidelities_avgd)
        np.save(path + "final_fidelities_var", final_fidelities_var)

    if PLOT:
        plot_double_descent(x_, cost_fxn_name, path, 
                            final_train_losses_avgd, np.sqrt(final_train_losses_var)/np.sqrt(len(seeds)), 
                            final_fidelities=final_fidelities_avgd, final_fidelities_errors=np.sqrt(final_fidelities_var)/np.sqrt(len(seeds)),
                            final_test_losses=None, final_test_losses_errors=None)
        if len(seeds) > 1:
            plot_double_descent_variances(x_, cost_fxn_name, path, 
                                          final_train_losses_var, 
                                          final_fidelities_errors=final_fidelities_var, 
                                          final_test_losses_errors=None)

else:
    if how_to_split_data=='parity':
        num_pairs = args.num_pairs
        path = base_path + system_path + f'TestTrain/Parity_{num_pairs}Pairs/{cost_fxn_name}/'
    else:
        frac_in_test = args.frac_data_in_test
        path = base_path + system_path + f'TestTrain/testSize{frac_in_test}_{how_to_split_data}/{cost_fxn_name}/'
    if batch_size != 2**N:
        path += f'bs_{batch_size}/'
    else:
        path += 'not_batched/'
    if not os.path.exists(path):
        os.makedirs(path)

    for seed in seeds:
        seed_path = path + f'seed_{seed}/'
        if len(seeds)==1:
            path = seed_path 

        if not os.path.exists(seed_path + '/train_losses.pkl'):
            print(seed_path + '/train_losses.pkl')
            print("----- FILE NOT FOUND, KEEP GOING ------")
            # raise FileNotFoundError("Data does not exist, cannot analyze!")
        else:
            num_params_dict = try_load_dict(seed_path + 'num_params.pkl')
            train_losses_dict = try_load_dict(seed_path + 'train_losses.pkl')
            test_losses_dict = try_load_dict(seed_path + 'test_losses.pkl')
            fidelities_dict = try_load_dict(seed_path + 'fidelities.pkl')
            print(f"Loaded data for seed {seed}!")
            if seed ==seeds[0]:
                for feat_dim in list(num_params_dict.keys()):
                    final_train_losses[feat_dim] = []
                    final_test_losses[feat_dim] = []
                    final_fidelities[feat_dim] = []
        
        for feat_dim in list(num_params_dict.keys()):
            if feat_dim in list(train_losses_dict.keys()):
                (depth,width) = feat_dim
                final_train_loss = train_losses_dict[feat_dim][-1]
                final_test_loss = test_losses_dict[feat_dim][-1]
                final_fidelity = fidelities_dict[feat_dim][-1]
                final_train_losses[feat_dim].append(final_train_loss)
                final_test_losses[feat_dim].append(final_test_loss)
                final_fidelities[feat_dim].append(final_fidelity)

    x_ = []
    final_train_losses_avgd = []
    final_train_losses_var = []
    final_test_losses_avgd = []
    final_test_losses_var = []
    final_fidelities_avgd = []
    final_fidelities_var = []
    for feat_dim in list(num_params_dict.keys()):
        num_params = num_params_dict[feat_dim]
        x_.append(num_params/2**N)
        final_train_losses_avgd.append(np.nanmean(final_train_losses[feat_dim]))
        final_train_losses_var.append(np.nanvar(final_train_losses[feat_dim]))
        final_test_losses_avgd.append(np.nanmean(final_test_losses[feat_dim]))
        final_test_losses_var.append(np.nanvar(final_test_losses[feat_dim]))
        final_fidelities_avgd.append(np.nanmean(final_fidelities[feat_dim]))
        final_fidelities_var.append(np.nanvar(final_fidelities[feat_dim]))
    
    if SAVE:
        np.save(path + "params_Hilbert_ratio", x_)
        np.save(path + "final_train_losses", final_train_losses_avgd)
        np.save(path + "final_train_losses_var", final_train_losses_var)
        np.save(path + "final_test_losses", final_test_losses_avgd)
        np.save(path + "final_test_losses_var", final_test_losses_var)
        np.save(path + "final_fidelities", final_fidelities_avgd)
        np.save(path + "final_fidelities_var", final_fidelities_var)
    
    if PLOT:
        plot_double_descent(x_, cost_fxn_name, path, 
                            final_train_losses_avgd, np.sqrt(final_train_losses_var)/np.sqrt(len(seeds)), 
                            final_fidelities=final_fidelities_avgd, final_fidelities_errors=np.sqrt(final_fidelities_var)/np.sqrt(len(seeds)),
                            final_test_losses=final_test_losses_avgd, final_test_losses_errors=np.sqrt(final_test_losses_var)/np.sqrt(len(seeds)))
        if len(seeds) > 1:
            plot_double_descent_variances(x_, cost_fxn_name, path, 
                                          final_train_losses_var, 
                                          final_fidelities_errors=final_fidelities_var, 
                                          final_test_losses_errors=final_test_losses_var)

print("Done!")