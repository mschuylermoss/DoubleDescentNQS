import numpy as np

def all_to_all_interactions(L):
    
    interactions = {}
    for spin_i in range(L):
        for spin_j in range(spin_i, L):
            interaction = np.array([spin_i,spin_j])
            r_1 = abs(spin_j - spin_i)
            r_2 = L - abs(spin_i - spin_j)
            r = min(r_1,r_2)
            if r not in interactions.keys():
                interactions[r] = []
            interactions[r].append((spin_i,spin_j))
    
    np_interactions = {}
    for r in interactions.keys():
        np_interactions[r] = np.stack(interactions[r])

    return np_interactions


def full_correlations(N, configs, probs):
    assert 2**N == len(configs), "# configs < Hilbert space size"

    interactions = all_to_all_interactions(N)
    probs /= sum(probs)

    correlations = {}
    for r in interactions.keys():
        interactions_r = interactions[r]
        sigma_is = (configs[:,interactions_r[:,0]])
        sigma_js = (configs[:,interactions_r[:,1]])
        SziSzjs = np.mean(np.multiply(sigma_is,sigma_js),axis=1)
        expect_SziSzj = sum(probs*SziSzjs)
        correlations[r] = expect_SziSzj
    
    return correlations

def partial_correlations(N, configs, probs):
    assert 2**N > len(configs), "# configs < Hilbert space size"

    interactions = all_to_all_interactions(N)
    probs /= sum(probs)

    correlations = {}
    for r in interactions.keys():
        interactions_r = interactions[r]
        sigma_is = (configs[:,interactions_r[:,0]])
        sigma_js = (configs[:,interactions_r[:,1]])
        SziSzjs = np.mean(np.multiply(sigma_is,sigma_js),axis=1)
        expect_SziSzj = sum(probs*SziSzjs)
        correlations[r] = expect_SziSzj
    
    return correlations