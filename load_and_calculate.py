import os
import pickle
import argparse

import jax
import numpy as np
import netket as nk 
from netket.operator.spin import sigmax, sigmaz
import matplotlib.pylab as plt
from itertools import product
from jax import numpy as jnp
from architectures import staticDenseNQS
from flax.core import freeze, unfreeze
from dset_helpers import make_data, load_data, load_random_data

from cost_functions import get_cost_fxn
from helpers import try_load_dict
from plot import plot_double_descent, plot_double_descent_variances
from helpers import get_features, get_num_params, try_load_dict

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
features = get_features(N,extra_large=True)
configs = jnp.array([[int(b)*2 - 1 for b in format(i, f'0{N}b')] for i in range(2**N)])

# Hamiltonian & Exact Energy
graph = nk.graph.Chain(length=N, pbc=True)
hi = nk.hilbert.Spin(s=0.5,N=N)
ha = sum([h*sigmax(hi,i) for i in range(N)])
ha += sum([-sigmaz(hi,i)*sigmaz(hi,(i+1)%N) for i in range(N)])
E_gs, gs = nk.exact.lanczos_ed(ha,compute_eigenvectors=True)

if args.seed == -1:
    seeds = [10,20,30,40,50,60,70,80,90,100]
else:
    seeds = [args.seed]
PLOT = True
SAVE = True
base_path = f'/mnt/home/smoss/ceph/nqsLandscape/Supervised/'
system_path = f'{exp_name}/IsingChain_h{h}_N{N}/'

normalizations = {}
R_sq_vals = {}
log_R_sq_vals = {}
for feat in features:
    normalizations[feat] = []
    R_sq_vals[feat] = []
    log_R_sq_vals[feat] = []

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
    frac_in_test = args.frac_data_in_test
    path = base_path + system_path + f'TestTrain/testSize{frac_in_test}_{how_to_split_data}/{cost_fxn_name}/'
    weights_path = base_path + f'weights/' + system_path + f'TestTrain/testSize{frac_in_test}_{how_to_split_data}/{cost_fxn_name}/'
    if batch_size != 2**N:
        path += f'bs_{batch_size}/'
        weights_path += f'bs_{batch_size}/'
    else:
        path += 'not_batched/'
        weights_path += f'not_batched/'
    if not os.path.exists(path):
        os.makedirs(path)

    for seed in seeds:
        print(normalizations)
        seed_path = path + f'seed_{seed}/'
        weights_seed_path = weights_path + f'seed_{seed}/'
        if len(seeds)==1:
            path = seed_path 

        data_PRNG = jax.random.PRNGKey(seed)
        x_train, x_test, y_train, y_test, all_target_logpsis = load_random_data(h, gs, data_PRNG, how_to_split_data, frac_in_test, path, halfHilbert=False)

        if not os.path.exists(weights_seed_path + '/depth3_width2.pkl'):
            print(weights_seed_path)
            print("----- NO WEIGHTS FOUND ------")
            # raise FileNotFoundError("Data does not exist, cannot analyze!")
        else:
            print(f"Weights found for seed {seed}")
            for feat in features:
                (depth,width) = feat
                wavefxn = staticDenseNQS(features=[width]*depth)
                params = wavefxn.init(jax.random.PRNGKey(2), np.random.choice([-1, 1], size=(N,)))
                with open(weights_seed_path + f"depth{depth}_width{width}.pkl", "rb") as f:
                    data = pickle.load(f)
                print(f"Loaded weights for NN size ({depth},{width})!")

                @jax.jit
                def eval_psi(params, x):
                    log_amp = wavefxn.apply(params, x)
                    psi = jnp.exp(log_amp)
                    return psi
                
                params_unfrozen = unfreeze(params)
                params_unfrozen['params'] = data['params']
                params = freeze(params_unfrozen)

                # R^2
                f_train = eval_psi(params,x_train)**2
                residuals = y_train - f_train
                mean_y = jnp.mean(y_train)
                SS_res = jnp.sum(residuals**2)
                SS_tot = jnp.sum((y_train-mean_y)**2)
                R_sq = 1 - SS_res/SS_tot
                R_sq_vals[feat].append(R_sq)

                log_residuals = jnp.log(y_train) - jnp.log(f_train)
                log_mean_y = jnp.log(mean_y)
                log_SS_res = jnp.sum(log_residuals **2)
                log_SS_tot = jnp.sum((jnp.log(y_train) - log_mean_y)**2)
                log_R_sq = 1 - log_SS_res/log_SS_tot
                log_R_sq_vals[feat].append(log_R_sq)

                # normalization
                psi_learned_unnorm = eval_psi(params, configs)
                normalization = jnp.sqrt(psi_learned_unnorm.T @ psi_learned_unnorm)
                psi_learned_norm = 1/normalization * psi_learned_unnorm
                normalizations[feat].append(normalization)

np.save(path + 'normalizations', normalizations)
np.save(path + 'R_sq_values', R_sq_vals)
np.save(path + 'log_R_sq_values', log_R_sq_vals)

print("done")