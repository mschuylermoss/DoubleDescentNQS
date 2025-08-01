import copy
import jax
import numpy as np
import optax

from jax import numpy as jnp
from architectures import staticDenseNQS
from dset_helpers import create_mini_batches
from cost_functions import fidelity, l2_loss_params

# jax.config.update("jax_enable_x64",True)

def train_on_all_configs(features, num_epochs, batch_size, cost, 
                         all_x, all_y, target_logpsis, 
                         key=100, lr=0.001, decay=0.99,
                         PRINT=False):

    train_losses = []
    fidelities = []

    key = jax.random.PRNGKey(key)
    zero_config = all_x[0]
    N_spins = len(zero_config)

    wavefxn = staticDenseNQS(features=features)
    params = wavefxn.init(key, zero_config)

    width = features[0]
    if width < 432: # 432 for probs splitting # 200 for mc splitting
        print('exponential decay') 
        scheduler = optax.exponential_decay(init_value=lr, transition_steps=1000, decay_rate=decay)
    else:
        print('warm up + exp decay')
        scheduler = optax.schedules.warmup_exponential_decay_schedule(init_value=1e-6,peak_value=lr,
                                                                    warmup_steps=10000,
                                                                    transition_steps=1000,decay_rate=decay)
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1.0))
    opt_state = optimizer.init(params)

    @jax.jit
    def cost_fxn(params,x,y):
        return cost(wavefxn,params,x,y)

    @jax.jit
    def update(params, x_batch, y_batch, opt_state):
        loss_val, grads = jax.value_and_grad(cost_fxn)(params, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    train_loss = cost_fxn(params, all_x, all_y)
    initial_fidelity = fidelity(wavefxn, params, all_x, target_logpsis)
    if PRINT:
        print("Initial train loss (evaluated on all x): ", train_loss)
        print("Initial fidelity: ", initial_fidelity)

    for epoch in range(num_epochs):
        mini_batches = create_mini_batches(all_x, all_y, batch_size)
        for x_batch, y_batch in mini_batches:
            params, opt_state, loss_val = update(params, x_batch, y_batch, opt_state)
        train_loss = np.array(cost_fxn(params, all_x, all_y))
        track_fidelity = np.array(fidelity(wavefxn,params,all_x,target_logpsis))
        if not np.isnan(train_loss):
            train_losses.append(train_loss)
            fidelities.append(track_fidelity)
            if ((epoch+1) % 100 == 0) & PRINT:
                print(f"Epoch {epoch + 1}, Loss: {train_loss}")
        else:
            print("Breaking training due to NaNs...")
            break

    if PRINT:
        print("final train loss (evaluated on all x): ", train_losses[-1])
        print("final wavefunction fidelity: ",fidelities[-1])

    return train_losses, fidelities, params, opt_state


def train_w_test(features, num_epochs, batch_size, cost, 
                 _x_train, input_y_train, _x_test, input_y_test, 
                 target_logpsis, 
                 weight_decay=False,
                 key=100, lr=0.001, decay=0.99,
                 rescale=False, PRINT=False):
    
    train_losses = []
    test_losses = []
    fidelities = []

    key = jax.random.PRNGKey(key)
    config = _x_train[0]
    N_spins = len(config)
    if len(input_y_test)==(len(target_logpsis)-len(input_y_train)):
        all_x = jnp.concatenate((_x_train,_x_test),axis=0)
        all_y = jnp.concatenate((input_y_train,input_y_test),axis=0)
        # have to do this so that the order of configs is the same
        target_logpsis = 0.5 * jnp.log(all_y) 
    else:
        # go here for MC sampling since x_train contains repeats
        all_x = jnp.array([[int(x)*2-1 for x in format(i, '0{}b'.format(N_spins))] for i in range(2**(N_spins))]) 
        # now input target logpsis are in the same order as all_x above
        all_y = jnp.exp(target_logpsis)**2

    x = np.random.choice([-1, 1], size=(N_spins))
    wavefxn = staticDenseNQS(features=features)
    params = wavefxn.init(key, config)

    width = features[0]
    if width < 432: # 432 for probs splitting # 200 for mc splitting
        print('exponential decay') 
        scheduler = optax.exponential_decay(init_value=lr, transition_steps=1000, decay_rate=decay)
    else:
        print('warm up + exp decay')
        scheduler = optax.schedules.warmup_exponential_decay_schedule(init_value=1e-6,peak_value=lr,
                                                                    warmup_steps=10000,
                                                                    transition_steps=1000,decay_rate=decay)
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(scheduler), optax.scale(-1.0))
    opt_state = optimizer.init(params)

    if rescale:
        scale = 2**N_spins
    else:
        scale = 1
    _y_train = copy.deepcopy(input_y_train) * scale
    _y_test = copy.deepcopy(input_y_test) * scale

    @jax.jit
    def cost_fxn(params,x,y):
        return cost(wavefxn,params,x,y)

    @jax.jit
    def update(params, x_batch, y_batch, opt_state):
        loss_val, grads = jax.value_and_grad(cost_fxn)(params, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    train_loss = cost_fxn(params, _x_train, _y_train)
    test_loss = cost_fxn(params, _x_test, _y_test)
    initial_fidelity = fidelity(wavefxn, params, all_x, target_logpsis)
    if PRINT:
        print("Initial train loss: ", train_loss/scale**2)
        print("Initial test loss: ", test_loss/scale**2)
        print("Initial fidelity: ", initial_fidelity)

    for epoch in range(num_epochs):
        minibatchkey = jax.random.PRNGKey(epoch+1)
        mini_batches = create_mini_batches(_x_train, _y_train, batch_size, seed=minibatchkey)
        for x_batch, y_batch in mini_batches:
            params, opt_state, loss_val = update(params, x_batch, y_batch, opt_state)
        train_loss = np.array(cost_fxn(params, _x_train, _y_train))
        test_loss = np.array(cost_fxn(params, _x_test, _y_test))
        track_fidelity = np.array(fidelity(wavefxn, params, all_x, target_logpsis))
        if not np.isnan(train_loss):
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            fidelities.append(track_fidelity)
            if ((epoch+1) % 100 == 0) and PRINT:
                print(f"Epoch {epoch + 1}, Loss: {train_loss/scale**2}")
        else:
            print("Breaking training due to NaNs...")
            break

    if PRINT:
        print("final train loss: ", train_losses[-1]/scale**2)
        print("final test loss: ", test_losses[-1]/scale**2)
        print("final fidelity: ", fidelities[-1])

    return train_losses, test_losses, fidelities, params, opt_state