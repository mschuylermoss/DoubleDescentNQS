import jax
from jax import numpy as jnp

# jax.config.update("jax_enable_x64", True)

def get_cost_fxn(name):
  if name=='mse':
    return mse
  elif name=='sse':
    return sse
  elif name=='mae':
    return mae
  elif name=='hellinger':
    return hellinger_distance
  elif name=='infidelity':
    return infidelity
  else:
    raise NotImplementedError(f"{name} not implemented as a valid cost function!")

def l2_loss(x, alpha):
    return alpha * (x ** 2).sum()

def l2_loss_params(model_params,alpha=1e-5):
  return sum(l2_loss(w, alpha=alpha) for w in jax.tree_leaves(model_params["params"]))


# -------- Cost Functions ------------

# O(probs^2)
def mse(model, params, x_batched, y_batched, scale=1):

  def squared_error(x, y):
    logpsi = model.apply(params, x)
    psi = jnp.exp(logpsi)
    prob = psi**2 * scale
    return (prob - y) ** 2
    
  return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched))

# O(probs^2)
def sse(model, params, x_batched, y_batched, scale=1):

  def squared_error(x, y):
    logpsi = model.apply(params, x)
    psi = jnp.exp(logpsi)
    prob = psi**2 * scale
    return (prob - y) ** 2
    
  return jnp.sum(jax.vmap(squared_error)(x_batched,y_batched))


# O(probs)
def mae(model, params, x_batched, y_batched, scale=1):
    
    def absolute_error(x, y):
        logpsi = model.apply(params, x)
        psi = jnp.exp(logpsi)
        prob = psi**2 * scale
        return jnp.abs(prob - y)
    
    return jnp.mean(jax.vmap(absolute_error)(x_batched, y_batched))


# basically MSE between amplitudes - O(amps^2) = O(probs)
def hellinger_distance(model, params, x_batched, y_batched, scale=1):

    def distance(x, y):
        logpsi = model.apply(params, x)
        psi = jnp.exp(logpsi)
        prob = psi**2 * scale
        return (jnp.sqrt(prob) - jnp.sqrt(y)) ** 2
    
    Hellinger_distance = (1/jnp.sqrt(2)) * jnp.sqrt(jnp.sum(jax.vmap(distance)(x_batched,y_batched)))
    
    return Hellinger_distance

# -------- EXACT Fidelity and Infidelity ------------

def infidelity(model, params, all_x, all_y):
  """
  all_x: all configurations in a hilbert space
  all_y: the target LOG AMPLITUDES for all configurations (psi^2)
  """

  logpsi = model.apply(params, all_x)
  psi = jnp.exp(logpsi)
  Norm = jnp.sum(psi**2)
  psi /= jnp.sqrt(Norm)
  target_psis = jnp.exp(all_y)
  overlap = jnp.sum(jnp.multiply(jnp.squeeze(psi),jnp.squeeze(target_psis)))
  
  return 1 - jnp.linalg.norm(overlap)**2

def fidelity(model, params, all_x, all_y):
  """
  all_x: all configurations in a hilbert space
  all_y: the target LOG AMPLITUDES for all configurations (psi^2)
  """

  logpsi = model.apply(params, all_x)
  psi = jnp.exp(logpsi)
  Norm = jnp.sum(psi**2)
  psi /= jnp.sqrt(Norm)
  target_psis = jnp.exp(all_y)
  overlap = jnp.sum(jnp.multiply(jnp.squeeze(psi),jnp.squeeze(target_psis)))
  
  return jnp.linalg.norm(overlap)**2
