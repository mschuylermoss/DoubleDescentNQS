import matplotlib.pylab as plt
import numpy as np

def plot_training_summary(feat_dims, cost_fxn_name, path, 
                          train_losses_dict,fidelities_dict,test_losses_dict=None):
        
        fig, axs = plt.subplots(2,1)
        fig.set_size_inches(10,10)
        blues = plt.get_cmap('Blues')
        reds = plt.get_cmap('Reds')
        greens = plt.get_cmap('Greens')

        for feat_dim_i,feat_dim in enumerate(feat_dims):
            if feat_dim in train_losses_dict.keys():
                train_losses = train_losses_dict[feat_dim]
                x = len(train_losses)
                axs[0].plot(range(1, x + 1), train_losses, label=f"train, {feat_dim}", color=blues((feat_dim_i+1)/(len(feat_dims)+1)))
                if fidelities_dict is not None:
                    fidelities = fidelities_dict[feat_dim]
                    infidelities = abs(1 - np.array(fidelities))
                    axs[1].plot(range(1, x + 1), infidelities, label=f"{feat_dim}", color=greens((feat_dim_i+1)/(len(feat_dims)+1)))
                if test_losses_dict != None:
                    test_losses = test_losses_dict[feat_dim]
                    axs[0].plot(range(1, x + 1), test_losses, label=f"test, {feat_dim}", color=reds((feat_dim_i+1)/(len(feat_dims)+1)))

        axs[0].set_ylabel("Loss")
        axs[0].set_yscale("log")
        # axs[0].legend()

        axs[1].set_yscale('log')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Infidelity")
        # axs[1].legend()
        fig.suptitle(cost_fxn_name)
        plt.savefig(path + 'summary.png', bbox_inches='tight')
        plt.close()

def plot_double_descent(x_, title, path, 
                        final_train_losses, 
                        final_train_losses_errors,
                        final_fidelities=None, final_fidelities_errors=None,
                        final_test_losses=None, final_test_losses_errors=None):
    
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(10,10)
    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    greens = plt.get_cmap('Greens')
    
    min_infid = 0.1
    max_infid = 0.1
    min_test = 100
    max_test = 100
    min_train = min(final_train_losses)
    max_train = max(final_train_losses)
    axs[0].errorbar(x_, final_train_losses, yerr=final_train_losses_errors, 
                    color=blues(0.75), fmt='o', linestyle='none', label=f"train" )
    if final_test_losses is not None:
        min_test = min(final_test_losses)
        max_test = max(final_test_losses)
        axs[0].errorbar(x_, final_test_losses, yerr=final_test_losses_errors, label=f"test", 
                        color=reds(0.75), fmt='o', linestyle='none', )
    if final_fidelities is not None:
        infidelities = abs(1-np.array(final_fidelities))
        min_infid = min(infidelities)
        max_infid = max(infidelities)
        # axs[1].scatter(x_,infidelities,color=greens(0.75))
        axs[1].errorbar(x_, infidelities, yerr=final_fidelities_errors/np.sqrt(10), 
                        color=greens(0.75), fmt='o', linestyle='none', )

    axs[0].vlines(1, min(min_train,min_test),max(max_train,max_test),linestyle='--',color='grey')
    axs[0].set_ylabel("Loss",fontsize=20)
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].legend()

    axs[1].vlines(1, min_infid, max_infid,linestyle='--',color='grey')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_xlabel("#params/Hilbert",fontsize=20)
    axs[1].set_ylabel("Infidelity",fontsize=20)

    fig.suptitle(path,fontsize=15)
    plt.savefig(path + 'doubleDescent.png', bbox_inches='tight')


def plot_double_descent_variances(x_, title, path, 
                        final_train_losses_errors,
                        final_fidelities_errors=None,
                        final_test_losses_errors=None):
    
    fig, axs = plt.subplots(1,1)
    fig.set_size_inches(8,8)
    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    greens = plt.get_cmap('Greens')

    min_fid = 100
    max_fid = 100
    min_test = 100
    max_test = 100
    min_train = min(final_train_losses_errors)
    max_train = max(final_train_losses_errors)
    axs.scatter(x_, final_train_losses_errors, 
                    color=blues(0.75), label=f"train")
    if final_test_losses_errors is not None:
        min_test = min(final_test_losses_errors)
        max_test = max(final_test_losses_errors)
        axs.scatter(x_, final_test_losses_errors, label=f"test", 
                        color=reds(0.75))
    if final_fidelities_errors is not None:
        min_fid = min(final_fidelities_errors)
        max_fid = max(final_fidelities_errors)
        axs.scatter(x_, final_fidelities_errors, label=f"fidelity", 
                        color=greens(0.75))

    axs.vlines(1, min(min_train,min_test,min_fid),max(max_train,max_test,max_fid),linestyle='--',color='grey')
    axs.set_ylabel("Variance",fontsize=20)
    axs.set_xlabel("#params/Hilbert",fontsize=20)
    axs.set_yscale("log")
    axs.set_xscale("log")
    axs.legend()

    fig.suptitle(path,fontsize=15)
    plt.savefig(path + 'doubleDescent_vars.png', bbox_inches='tight')


def plot_quantities_vs_test_size(which_best, fracs,fidelities,
                                 path,
                                 train=None,test=None,params=None):
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(8,8)
    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    greens = plt.get_cmap('Greens')

    axs[0].scatter(fracs,abs(1-np.array(fidelities)), color=greens(0.75),label='best fidelity')
    axs[0].plot(fracs,abs(1-np.array(fidelities)), color=greens(0.7))
    if train is not None:
        axs[0].scatter(fracs,train, color=blues(0.75),label='train loss')
        axs[0].plot(fracs,train, color=blues(0.7))
    if test is not None:
        axs[0].scatter(fracs,test, color=reds(0.75),label='test loss') 
        axs[0].plot(fracs,test, color=reds(0.7)) 

    axs[0].set_yscale('log')

    if params is not None:
        axs[1].scatter(fracs,params,color='k')

    axs[1].set_ylabel('#params/Hilbert')
    plt.xlabel("size of TEST set")
    plt.savefig(path + f'test_size_experiments_{which_best}.png', bbox_inches='tight')

def plot_min_parameters_for_fidelity_thresh(threshold_val, path, fracs, num_params):
    fig, axs = plt.subplots(1,1)
    fig.set_size_inches(8,8)

    num_points = len(num_params)

    plt.scatter(fracs[:num_points],num_params, color='k')

    plt.xlim(-0.05,1.01)
    plt.ylabel('#params/Hilbert')
    plt.xlabel("size of TEST set")
    plt.title(f'infidelity < {threshold_val}')
    plt.savefig(path + f'params_for_fid_thresh_{threshold_val}.png', bbox_inches='tight')
