import numpy as np
import pickle
from itertools import product

def get_features(N,extra_large=False):
    if N==12:
        depths = np.array([3])
        small_widths = np.arange(2,4*N,2)
        large_widths = np.arange(4*N,3*N*N,N)
        # large_widths = np.arange(4*N,int(2.5*N*N+1),N)
        ex_large_widths_1 = np.arange(3*N*N,5*N*N,2*N)
        ex_large_widths_2 = np.arange(5*N*N,N*N*N,N*N)
        all_ex_large_widths = np.concatenate((ex_large_widths_1,ex_large_widths_2),axis=0)
        all_widths = np.concatenate((small_widths,large_widths),axis=0)
        if extra_large:
            all_widths = np.concatenate((all_widths,all_ex_large_widths),axis=0)
    elif N==16:
        depths = np.array([4])
        small_widths = np.arange(N,N**2,int(N/2))
        large_widths = np.arange(N**2,2*N**2,2*N)
        ex_large_widths = np.arange(N**2,4*N**2,4*N)
        all_widths = np.concatenate((small_widths,large_widths),axis=0)
        if extra_large:
            all_widths = np.concatenate((all_widths,ex_large_widths),axis=0)
    else:
        print("Returning generic set of features")
        depths = np.array([3])
        small_widths = np.arange(int(N/2),2*N,2)
        large_widths = np.arange(2*N,N*N+1,2*N)
        all_widths = np.concatenate((small_widths,large_widths),axis=0)

    features = list(product(depths, all_widths))

    return features

def get_num_params(N, depth=3,width=10):
    num_params = N*width + width                 # input layer weights 
    num_params += (depth-1)*(width**2 + width)   # intermediate layer weights and biases
    num_params += width + 1                      # final layer weights and bias
    num_params += width * depth * 2              # params from layer norm
    return num_params

def try_load_dict(file):
    try:
        with open(file, "rb") as f:
            file_loaded = pickle.load(f)
    except EOFError:
        print(f"Error: {file} is empty or corrupted.")
        file_loaded = None
    return file_loaded
