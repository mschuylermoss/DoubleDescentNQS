import os
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pylab as plt

# jax.config.update("jax_enable_x64", True)

def create_mini_batches(x_data, y_data, batch_size, seed=100):
    np.random.seed(seed)
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    x_shuffled = x_data[indices]
    y_shuffled = y_data[indices]
    
    mini_batches = [(x_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size])
                    for i in range(0, x_data.shape[0], batch_size)]
    return mini_batches

def convert_samples_to_index(samples):
    binary_vectors = (samples + 1) // 2 
    powers_of_two = 2 ** jnp.arange(samples.shape[-1])[::-1]  
    indices = jnp.dot(binary_vectors, powers_of_two) 
    return indices

def plot_test_train_data(x_train,y_train,x_test,y_test,path):
    plt.figure(figsize=(8,5))
    plt.scatter(convert_samples_to_index(x_test), y_test, marker = "*", s=50, color=plt.get_cmap('tab10')(1),label='test')
    plt.scatter(convert_samples_to_index(x_train), y_train, marker = "o", s=10, color=plt.get_cmap('tab10')(0),label='train')
    plt.legend(loc='upper center')
    plt.ylabel("Probability")
    plt.yscale('log')
    plt.savefig(path + '/data_viz.png', bbox_inches='tight')
    plt.close()

def make_data(h, gs_wavefxn, key, how_to_split, test_frac=0.25, save=True, halfHilbert=False, save_train_inds=False):
    sizeHilbert = len(gs_wavefxn)
    N = round(jnp.log2(sizeHilbert))
    if halfHilbert:
        sizeHilbert=int(sizeHilbert/2)
        gs_wavefxn = gs_wavefxn[:sizeHilbert]
        gs_wavefxn /= np.linalg.norm(gs_wavefxn)
    x = np.array([[int(x)*2-1 for x in format(i, '0{}b'.format(N))] for i in range(sizeHilbert)])      ## all configs in Hilbert space, in -1,+1 convention
    y = np.abs(gs_wavefxn)**2                                                                          ## probabilities for each sample 
    target_logprobs = jnp.log(y)                                                                       ## log(prob) for each sample
    target_logpsis = jnp.log(abs(gs_wavefxn))                                                          ## log(psi) for each sample
    
    if how_to_split=='dont':
        test_frac = 0
    num_test_samples = int(sizeHilbert * test_frac)

    if how_to_split=='random':
        shuffled_indices = jax.random.permutation(key,sizeHilbert) ## SEED THIS!
        test_indices = shuffled_indices[:num_test_samples]
        train_indices = shuffled_indices[num_test_samples:]
    if how_to_split=='chunk':
        start = 0
        indices = np.arange(sizeHilbert)
        test_chunk = np.arange(start,start+num_test_samples,1)
        train_chunk = np.setdiff1d(indices,test_chunk)
        test_indices = indices[test_chunk]
        train_indices = indices[train_chunk]
    if how_to_split=='probs':
        testprobs = 'low'
        if testprobs=='high':
            sorted_order = np.argsort(np.squeeze(y))[::-1]
            test_indices = sorted_order[:num_test_samples]
            train_indices = sorted_order[num_test_samples:]
        elif testprobs=='low':
            sorted_order = np.argsort(np.squeeze(y))
            test_indices = sorted_order[:num_test_samples]
            train_indices = sorted_order[num_test_samples:]
        else: # middle
            start = int(sizeHilbert/2)
            indices = np.arange(sizeHilbert)
            sorted_order = np.argsort(np.squeeze(y))
            test_chunk = np.arange(start,start+num_test_samples,1)
            train_chunk = np.setdiff1d(indices,test_chunk)
            test_indices = sorted_order[test_chunk]
            train_indices = sorted_order[train_chunk]
    if how_to_split=='mc':
        num_samples = int((1-test_frac)*sizeHilbert)
        indices = np.arange(sizeHilbert)
        train_indices = jax.random.choice(key,jnp.arange(sizeHilbert),shape=(num_samples,),replace=True,p=jnp.squeeze(y))
        unique_train_indices = jnp.unique(train_indices)
        test_indices = np.setdiff1d(indices,unique_train_indices)

    if how_to_split == 'dont':
        x_train,x_test = x, []
        y_train,y_test = y, []
    else:
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        if len(y_test) == 0:
            num_max_prob_test = 0
        else:
            max_prob = max(max(y_train),max(y_test))
            num_max_prob_train = len(np.where(abs(y_train-max_prob)<1e-10)[0])
            num_max_prob_test = len(np.where(abs(y_test-max_prob)<1e-10)[0])
            print(f"NUM HIGH PROB SAMPLES IN TEST: {num_max_prob_test}")

    if (how_to_split != 'dont') & save:
        path = f'/mnt/home/smoss/ceph/nqsLandscape/Supervised/training_data/IsingChain_h{h}_N{N}/Test{test_frac}/{how_to_split}'
        # path = f'./training_data/IsingChain_h{h}_N{N}/Test{test_frac}/{how_to_split}'
        if (how_to_split == 'random'):
            path += f'_{num_max_prob_test}inTest'
        if not os.path.exists(path):
            os.makedirs(path)

        print(f"saving in {path}")
        np.save(path + '/x_train.npy', x_train)
        np.save(path + '/y_train.npy', y_train)
        np.save(path + '/x_test.npy', x_test)
        np.save(path + '/y_test.npy', y_test)
        np.save(path + '/target_logpsis.npy', target_logpsis)
        plot_test_train_data(x_train,y_train,x_test,y_test,path)
    
    if save_train_inds:
        return x_train, x_test, y_train, y_test, target_logpsis, train_indices
    else:
        return x_train, x_test, y_train, y_test, target_logpsis

def load_data(where_to_look, h, gs, how_to_split, test_frac):
    print(f'looking for data in {where_to_look}')
    while True:
        if not os.path.exists(where_to_look + '/target_logpsis.npy'):
            print('data does not exist... getting data')
            if not os.path.exists(where_to_look):
                os.makedirs(where_to_look)
            key = jax.random.PRNGKey(100)
            x_train, x_test, y_train, y_test, target_logpsis = make_data(h,gs,key,how_to_split,test_frac)
        else:
            print('got data!')
            x_train = np.load(where_to_look + '/x_train.npy')
            y_train = np.load(where_to_look + '/y_train.npy')
            x_test = np.load(where_to_look + '/x_test.npy')
            y_test = np.load(where_to_look + '/y_test.npy')
            target_logpsis = np.load(where_to_look + '/target_logpsis.npy')
            break 

    return x_train, x_test, y_train, y_test, target_logpsis

def load_random_data(h,gs,key,how_to_split,test_frac,path,halfHilbert=False):
    x_train, x_test, y_train, y_test, target_logpsis, train_inds = make_data(h,gs,key,how_to_split,test_frac,save=False,save_train_inds=True,halfHilbert=halfHilbert)
    if path is not None:
        np.save(path + 'train_inds',np.array(train_inds))
        plot_test_train_data(x_train,y_train,x_test,y_test,path)
    return x_train, x_test, y_train, y_test, target_logpsis

################ PARITY TESTING ################

def plot_parity_test_train_data(x_train,y_train,x_test,y_test,
                                pairs_in_train, pairs_in_train_y,
                                path,
                                exception_x=None,exception_y=None):
    plt.figure(figsize=(8,5))
    plt.scatter(convert_samples_to_index(x_test), y_test, marker = "o", s=25, color=plt.get_cmap('tab10')(1),alpha=0.5,label='test')
    plt.scatter(convert_samples_to_index(x_train), y_train, marker = "o", s=10, color=plt.get_cmap('tab10')(0),alpha=0.5,label='train')
    plt.scatter(pairs_in_train, pairs_in_train_y, marker = "o", s=10, color=plt.get_cmap('tab10')(2),label='train pairs')
    if exception_x is not None:
        plt.scatter(convert_samples_to_index(exception_x), exception_y, marker = "*", s=75, color=plt.get_cmap('tab10')(3),label='exception')
    plt.legend(loc='upper center')
    plt.ylabel("Probability")
    plt.yscale("log")
    plt.savefig(path + '/data_viz.png', bbox_inches='tight')
    plt.close()

def make_parity_data(gs_wavefxn, prng_key, path, num_pairs=0, exception=None, save_train_inds=False):

    sizeHilbert = len(gs_wavefxn)
    N = round(jnp.log2(sizeHilbert))
    x = np.array([[int(x)*2-1 for x in format(i, '0{}b'.format(N))] for i in range(sizeHilbert)])      ## all configs in Hilbert space, in -1,+1 convention
    y = np.abs(gs_wavefxn)**2                                                                          ## probabilities for each sample 
    target_logprobs = jnp.log(y)                                                                       ## log(prob) for each sample
    target_logpsis = jnp.log(abs(gs_wavefxn))                                                          ## log(psi) for each sample
    
    # split the hilbert space in half
    indices = np.array(range(sizeHilbert))
    half = int(sizeHilbert/2)
    test_chunk = np.arange(0,half,1)
    train_chunk = np.setdiff1d(indices,test_chunk)
    test_indices = indices[test_chunk]
    train_indices = indices[train_chunk][::-1]
    if exception is not None:
        assert exception < 2**(N-1), "excepted sample must be in S- sector"
        print(f"excepting {exception}")
        exception_x = x[exception]
        exception_y = y[exception]
        test_indices_excepted = np.delete(test_indices,exception)
        train_indices_excepted = np.delete(train_indices,exception)
    else:
        exception_x = None
        exception_y = None
        test_indices_excepted = test_indices
        train_indices_excepted = train_indices

    # scramble test and train
    excepted_inds = np.arange(len(test_indices_excepted))
    inds_to_swap = jax.random.permutation(prng_key, excepted_inds)[:len(train_indices)//2]
    train_indices_to_swap = train_indices_excepted[inds_to_swap]
    test_indices_to_swap = test_indices_excepted[inds_to_swap]
    train_indices_excepted_removed = np.delete(train_indices_excepted,inds_to_swap)
    test_indices_excepted_removed = np.delete(test_indices_excepted,inds_to_swap)
    train_indices_excepted_swapped = np.concatenate((train_indices_excepted_removed,test_indices_to_swap))
    test_indices_excepted_swapped = np.concatenate((test_indices_excepted_removed,train_indices_to_swap))
    assert np.var(train_indices_excepted_swapped+test_indices_excepted_swapped)==0, "pairs not preserved... var > 0"

    if num_pairs > 0:
        pair_choices = np.arange(len(test_indices_excepted))
        which_pairs = jax.random.permutation(prng_key, pair_choices)[:num_pairs]
        test_indices = np.delete(test_indices_excepted_swapped,which_pairs)
        pair_indices_test = test_indices_excepted_swapped[which_pairs]
        pair_indices_train = train_indices_excepted_swapped[which_pairs]
        assert np.var(pair_indices_test+pair_indices_train)==0, "pairs not preserved... var > 0"
        if exception is not None:
            train_indices = np.concatenate((np.atleast_1d(train_indices[exception]),train_indices_excepted_swapped,pair_indices_test))
        else:
            train_indices = np.concatenate((train_indices_excepted_swapped,pair_indices_test))
        pairs_in_train_inds = np.concatenate((pair_indices_train,pair_indices_test))
        pairs_in_train_y = y[pairs_in_train_inds]
    else:
        test_indices = test_indices_excepted_swapped
        if exception is not None:
            train_indices = np.concatenate((np.atleast_1d(train_indices[exception]),train_indices_excepted_swapped))
        else:
            train_indices = train_indices_excepted_swapped

    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    if len(y_test) == 0:
        num_max_prob_test = 0
    else:
        max_prob = max(max(y_train),max(y_test))
        num_max_prob_train = len(np.where(abs(y_train-max_prob)<1e-10)[0])
        num_max_prob_test = len(np.where(abs(y_test-max_prob)<1e-10)[0])
        print(f"NUM HIGH PROB SAMPLES IN TEST: {num_max_prob_test}")

    if num_pairs > 0:
        plot_parity_test_train_data(x_train,y_train,x_test,y_test,
                                    pairs_in_train_inds,pairs_in_train_y,
                                    path,
                                    exception_x,exception_y)
    else:
        pairs_in_train_inds = [np.nan]
        pairs_in_train_y = [np.nan]
        plot_parity_test_train_data(x_train,y_train,x_test,y_test,
                                    pairs_in_train_inds,pairs_in_train_y,
                                    path,
                                    exception_x,exception_y)

            
    if save_train_inds:
        np.save(path + 'pairs_in_train_inds',np.array(pairs_in_train_inds))
        np.save(path + 'train_inds',np.array(train_indices))

    return x_train, x_test, y_train, y_test, target_logpsis

def load_parity_data(where_to_look, gs, num_pairs=0, exception=None):
    print(f'looking for data in {where_to_look}')
    if not os.path.exists(where_to_look + '/target_logpsis.npy'):
        print('data does not exist... getting data')
        if not os.path.exists(where_to_look):
            os.makedirs(where_to_look)
        key = jax.random.PRNGKey(100)
        x_train, x_test, y_train, y_test, target_logpsis = make_parity_data(gs,key,where_to_look,
                                                                            num_pairs=num_pairs,exception=exception)
        np.save(where_to_look + '/x_train.npy',x_train)
        np.save(where_to_look + '/y_train.npy',y_train)
        np.save(where_to_look + '/x_test.npy',x_test)
        np.save(where_to_look + '/y_test.npy',y_test)
        np.save(where_to_look + '/target_logpsis.npy',target_logpsis)
    else:
        print('got data!')
        x_train = np.load(where_to_look + '/x_train.npy')
        y_train = np.load(where_to_look + '/y_train.npy')
        x_test = np.load(where_to_look + '/x_test.npy')
        y_test = np.load(where_to_look + '/y_test.npy')
        target_logpsis = np.load(where_to_look + '/target_logpsis.npy')

    return x_train, x_test, y_train, y_test, target_logpsis

################ AMPLITUDE STRUCTURE TESTING ################\

def plot_test_train_data_exclude_row(x_train,y_train,x_test,y_test,
                                     path,
                                     excluded_row_x=None,excluded_row_y=None):
    plt.figure(figsize=(8,5))
    plt.scatter(convert_samples_to_index(x_test), y_test, marker = "o", s=25, color=plt.get_cmap('tab10')(1),alpha=0.5,label='test')
    plt.scatter(convert_samples_to_index(x_train), y_train, marker = "o", s=10, color=plt.get_cmap('tab10')(0),alpha=0.5,label='train')
    if excluded_row_x is not None:
        plt.scatter(convert_samples_to_index(excluded_row_x), excluded_row_y, marker = "*", s=75, color=plt.get_cmap('tab10')(3),zorder=-1,label='excluded row')
    plt.legend(loc='upper center')
    plt.ylabel("Probability")
    plt.yscale('log')
    plt.savefig(path + '/data_viz.png', bbox_inches='tight')
    plt.close()

def make_data_exclude_row(gs_wavefxn, prng_key, path, 
                          test_frac=0.25,
                          which_row=1, how_many_in_row=0,
                          half_Hilbert=True,
                          save_train_inds=False):

    sizeHilbert = len(gs_wavefxn)
    N = round(jnp.log2(sizeHilbert))
    if half_Hilbert:
        sizeHilbert = int(sizeHilbert/2)
        indices = np.array(range(sizeHilbert))
        num_test_samples = int(sizeHilbert * test_frac)
        gs_wavefxn = gs_wavefxn[:sizeHilbert]
        gs_wavefxn /= np.linalg.norm(gs_wavefxn)
    else:
        indices = np.array(range(sizeHilbert))
        num_test_samples = int(sizeHilbert * test_frac)
    x = np.array([[int(x)*2-1 for x in format(i, '0{}b'.format(N))] for i in range(sizeHilbert)])      ## all configs in Hilbert space, in -1,+1 convention
    y = np.abs(gs_wavefxn)**2                                                                          ## probabilities for each sample 
    target_logprobs = jnp.log(y)                                                                       ## log(prob) for each sample
    target_logpsis = jnp.log(abs(gs_wavefxn))                                                          ## log(psi) for each sample

    squeezed_y = np.squeeze(y)
    y_sorted = np.sort(squeezed_y)[::-1]
    second_row_y = y_sorted[2]
    second_row_indices = np.where(np.isclose(squeezed_y,second_row_y,rtol=1e-7))[0]
    second_row_indices_randomized = jax.random.permutation(prng_key,second_row_indices)
    second_row_indices_train = second_row_indices_randomized[:how_many_in_row]
    second_row_indices_test = np.setdiff1d(second_row_indices,second_row_indices_train)
    num_test_samples -= len(second_row_indices_test)

    indices_row_excluded = np.setdiff1d(indices,second_row_indices)
    shuffled_indices_row_excluded = jax.random.permutation(prng_key,indices_row_excluded)
    test_indices = shuffled_indices_row_excluded[:num_test_samples]
    train_indices = np.setdiff1d(indices_row_excluded, test_indices)
    train_indices = np.concatenate((train_indices,second_row_indices_train))
    test_indices = np.concatenate((second_row_indices_test,test_indices))

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    plot_test_train_data_exclude_row(x_train,y_train,x_test,y_test,
                                     path,
                                     excluded_row_x=x[second_row_indices_test],
                                     excluded_row_y=y[second_row_indices_test])
    if save_train_inds:
        np.save(path + 'train_inds',np.array(train_indices))
        np.save(path + 'test_inds_second_row',np.array(second_row_indices_test))

    return x_train, x_test, y_train, y_test, target_logpsis
