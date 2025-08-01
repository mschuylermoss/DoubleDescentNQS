import jax
from functools import partial
from jax import numpy as jnp

import jax.flatten_util
import optax
import copy 
import numpy as np
from architectures import staticDenseNQS
from dset_helpers import create_mini_batches
from cost_functions import fidelity
from jax.tree_util import tree_map

def single_sample_ravel_fn(single_sample_jacobian):
    flat, _ = jax.flatten_util.ravel_pytree(single_sample_jacobian)
    return flat

@partial(jax.jit,static_argnums=(0,))
def calculate_QGT(apply_fun, params, samples, probs):

    jacobian = jax.jacrev(lambda x: apply_fun(x, samples))(params)

    ## Calculate QGT
    model_probs = jnp.exp(apply_fun(params,samples))**2
    model_probs /= jnp.sum(model_probs)

    jacobian_mat = jax.vmap(single_sample_ravel_fn)(jacobian)
    jacobian_scaled_mat = jacobian_mat * model_probs
    term1 = jacobian_mat.T @ jacobian_scaled_mat
    sum_jac_scaled_vec = jnp.sum(jacobian_scaled_mat,axis=0)
    term2 = jnp.outer(sum_jac_scaled_vec,sum_jac_scaled_vec)
    QGT = term1 - term2

    return QGT

@jax.jit
def apply_QGT(QGT, gradients, diag_shift=10**-1, epsilon=10**-15):

    # diagonalize QGT
    eigenvectors, eigenvalues = jax.lax.linalg.eigh(QGT + diag_shift*jnp.eye(QGT.shape[0]))
    
    # flatten gradients 
    gradients_vec, _ = jax.flatten_util.ravel_pytree(gradients)

    # apply regularized QGT to gradients
    updates_vec = jnp.matmul(eigenvectors.T, gradients_vec)
    updates_vec = jnp.where((eigenvalues/eigenvalues[-1])>epsilon, updates_vec/eigenvalues, 0.)
    updates_vec = jnp.matmul(eigenvectors,updates_vec)

    return updates_vec

def unflatten_updates(updates_vec,gradients):
    _, unravel_fn = jax.flatten_util.ravel_pytree(gradients)
    updates_pytree = unravel_fn(updates_vec)
    return updates_pytree

def train_on_all_configs_QGT(features, num_epochs, batch_size, cost, 
                 _x_train, input_y_train,
                 target_logpsis, 
                 key=100, lr=0.001,
                 rescale=False, PRINT=False):
    
    train_losses = []
    fidelities = []

    diag_decay = optax.exponential_decay(init_value=10**-3, transition_steps=100, decay_rate=0.99, transition_begin=0, staircase=False, end_value=10**-6)

    key = jax.random.PRNGKey(key)
    config = _x_train[0]
    N_spins = len(config)
    all_x = _x_train

    x = np.random.choice([-1, 1], size=(N_spins))
    wavefxn = staticDenseNQS(features=features)
    params = wavefxn.init(key, config)

    if rescale:
        scale = 2**N_spins
    else:
        scale = 1
    _y_train = copy.deepcopy(input_y_train) * scale

    @jax.jit
    def cost_fxn(params,x,y):
        return cost(wavefxn,params,x,y)

    @jax.jit
    def update(params, x_batch, y_batch, diag_shift_i):
        loss_val, grads = jax.value_and_grad(lambda x: cost_fxn(x,x_batch,y_batch))(params)   
        QGT = calculate_QGT(wavefxn.apply,params,x_batch,y_batch)
        updates = apply_QGT(QGT,grads,diag_shift_i)
        return grads, updates, loss_val

    train_loss = cost_fxn(params, _x_train, _y_train)
    initial_fidelity = fidelity(wavefxn, params, all_x, target_logpsis)
    if PRINT:
        print("Initial train loss: ", train_loss/scale**2)
        print("Initial fidelity: ", initial_fidelity)

    for epoch in range(num_epochs+1):
        diag_shift_i = diag_decay(epoch)
        minibatchkey = jax.random.PRNGKey(epoch+1)
        mini_batches = create_mini_batches(_x_train, _y_train, batch_size, seed=minibatchkey)
        for x_batch, y_batch in mini_batches:
            gradients, updates, _ = update(params, x_batch, y_batch, diag_shift_i)
            updates_pytree = unflatten_updates(updates, gradients)
            params = tree_map(lambda x, y: (x - 2*lr*y.real), params, updates_pytree)
        train_loss = np.array(cost_fxn(params, _x_train, _y_train))
        track_fidelity = np.array(fidelity(wavefxn, params, all_x, target_logpsis))
        if not np.isnan(train_loss):
            train_losses.append(train_loss)
            fidelities.append(track_fidelity)
            if ((epoch+1) % 100 == 0) and PRINT:
                print(f"Epoch {epoch + 1}, \nLoss: {train_loss/scale**2}, \nFidelity: {track_fidelity}")
        else:
            print("Breaking training due to NaNs...")
            break

    if PRINT:
        print("final train loss: ", train_losses[-1]/scale**2)
        print("final fidelity: ", fidelities[-1])

    return train_losses, fidelities, params


# def multiply_by_probs(quantities,y):
#     ndims = len(quantities.shape)
#     train_probs = jnp.squeeze(y)
#     train_probs_broadcast = jnp.expand_dims(train_probs,axis=[a for a in range(1,ndims)])
#     scaled_quantities = quantities * train_probs_broadcast
#     return scaled_quantities

# def sum_over_samples(quantities):
#     summed_quantities = jnp.sum(quantities,axis=0)
#     return summed_quantities

# @partial(jax.jit,static_argnums=(0,))
# def calculate_QGT_prev(apply_fun, params, samples, probs):
    
#     N_samples = samples.shape[0]

#     jacobian = jax.jacrev(lambda x: apply_fun(x, samples))(params)

#     ## Calculate QGT
#     model_probs = apply_fun(params,samples)
#     model_probs /= sum(model_probs)
#     jacobian_scaled = jax.tree_util.tree_map(lambda x: multiply_by_probs(x,model_probs),jacobian)

#     # sum jacobian over samples
#     sum_jac_scaled = jax.tree_util.tree_map(sum_over_samples,jacobian_scaled)
#     sum_jac_scaled_flattened, param_map = jax.tree_util.tree_flatten(sum_jac_scaled)
#     sum_jac_scaled_flattened_1d = [x.reshape(-1) for x in sum_jac_scaled_flattened]
#     sum_jac_scaled_vec = jnp.concatenate(sum_jac_scaled_flattened_1d, axis=0)
#     term2 = jnp.outer(sum_jac_scaled_vec,sum_jac_scaled_vec)

#     # flatten jacobian 
#     jacobian_flattened, _ = jax.tree_util.tree_flatten(jacobian)
#     jacobian_flattened_2d = [x.reshape(N_samples, -1) for x in jacobian_flattened]
#     jacobian_mat = jnp.concatenate(jacobian_flattened_2d,axis = 1) # (N_samples, N_params)

#     # flatten scaled jacobian 
#     jacobian_scaled_flattened, _ = jax.tree_util.tree_flatten(jacobian_scaled)
#     jacobian_scaled_flattened_2d = [x.reshape(N_samples, -1) for x in jacobian_scaled_flattened]
#     jacobian_scaled_mat = jnp.concatenate(jacobian_scaled_flattened_2d,axis = 1) # (N_samples, N_params)

#     term1 = jacobian_mat.T @ jacobian_scaled_mat

#     QGT = term1-term2

#     return QGT

# @jax.jit
# def apply_QGT_prev(QGT, gradients, diag_shift=10**-1, epsilon=10**-6):

#     # diagonalize QGT
#     eigenvectors, eigenvalues = jax.lax.linalg.eigh(QGT + diag_shift*jnp.eye(QGT.shape[0]))

#     # flatten gradients 
#     gradients_flattened, gradient_tree_def = jax.tree_util.tree_flatten(gradients)
#     gradients_flattened_1d = [x.reshape(-1) for x in gradients_flattened]
#     gradients_vec = jnp.concatenate(gradients_flattened_1d, axis=0)

#     # apply regularized QGT to gradients
#     updates_vec = jnp.matmul(eigenvectors.T, gradients_vec)
#     updates_vec = jnp.where((eigenvalues/eigenvalues[-1])>epsilon, updates_vec/eigenvalues, 0)
#     updates_vec = jnp.matmul(eigenvectors,updates_vec)

#     return updates_vec

# def unflatten_updates_prev(updates_vec,gradients):

#     # flatten gradients 
#     gradients_flattened, gradient_tree_def = jax.tree_util.tree_flatten(gradients)
#     gradients_flattened_shapes = [x.shape for x in gradients_flattened]
#     gradients_flattened_1d = [x.reshape(-1) for x in gradients_flattened]
#     gradients_vec = jnp.concatenate(gradients_flattened_1d, axis=0)

#     # unflatten updates
#     updates_flattened_1d_sizes = [jnp.prod(jnp.array(shape)) for shape in gradients_flattened_shapes]
#     updates_flattened_1d_indices = jnp.cumsum(jnp.array([0] + updates_flattened_1d_sizes))
#     updates_flattened_1d = [updates_vec[updates_flattened_1d_indices[i]:updates_flattened_1d_indices[i+1]].reshape(gradients_flattened_shapes[i])
#               for i in range(len(gradients_flattened_shapes))]
#     updates_pytree = jax.tree_util.tree_unflatten(gradient_tree_def,updates_flattened_1d) 

#     return updates_pytree
