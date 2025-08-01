import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.linen.initializers import xavier_uniform, he_uniform, variance_scaling

# jax.config.update("jax_enable_x64", True)

class staticDenseNQS(nn.Module):
    '''
    Static implementation of denseNN for NQS
    Setup function establishes the structure of the NN.
    __call__ function uses the layers established during setup.
    
    features:   list of integers representing number of nodes in each layer
                num layers given by the length of this list
    '''
    features: list
    activation: callable = nn.gelu  # Default activation function is GELU

    def setup(self):
        self.layers_inner = [nn.Dense(feat, kernel_init=he_uniform()) for feat in self.features]
        # self.layers_inner = [nn.Dense(feat, kernel_init=variance_scaling(1., 'fan_in', 'normal')) for feat in self.features]
        self.norms_inner = [nn.LayerNorm() for _ in self.features]
        self.layer_final = nn.Dense(1, kernel_init=he_uniform())

    def __call__(self, x):
        '''
        input x is a sample with N spins
        outputs logpsi(x)

        note that we study real wavefunctions only!
        '''
        for layer, norm in zip(self.layers_inner, self.norms_inner):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
        x = self.layer_final(x)
        return x
  

class dynamicDenseNQS(nn.Module):
    '''
    Dynamic implementation of denseNN for NQS
    No setup function needed because layers are initialized during __call__.
    
    features:   list of integers representing number of nodes in each layer
    '''
    features: list

    @nn.compact
    def __call__(self, x):
        '''
        input x is a sample with N spins
        outputs logpsi(x)
        '''
        for feature in self.features:
            x = nn.Dense(feature, kernel_init=he_uniform())(x)
            x = nn.relu(x)
        x = nn.Dense(1, kernel_init=he_uniform())(x)
        return x
  