from typing import Any, Tuple
import jax, jax.numpy as jnp, jax.random as jr 
import equinox as eqx
import optax
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from maf import MAF

Array = jax.Array
OptState  = optax.OptState 
Optimiser = optax.GradientTransformation


@eqx.filter_jit
def batch_loss_fn(model: MAF, x: Array, y: Array) -> Array:
    model = eqx.tree_inference(model, False)
    loss = jax.vmap(model.loss)(x=x, y=y).mean()
    return loss


@eqx.filter_jit
def make_step(
    model: MAF, 
    x: Array, 
    y: Array, 
    opt_state: OptState,
    opt: Optimiser
) -> Tuple[MAF, OptState, Array]:
    _fn = eqx.filter_value_and_grad(batch_loss_fn)
    L, grads = _fn(model, x, y)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, L 


@eqx.filter_jit
def batch_eval_fn(model: MAF, x: Array, y: Array) -> Array:
    model = eqx.tree_inference(model, True)
    loss = jax.vmap(model.loss)(x=x, y=y).mean()
    return loss


key = jr.PRNGKey(0)
n_data = 2000
n_steps = 10_000
lr = 2e-4
n_samples = 20_000
hidden_dim = 8
n_layers = 5

X, Y = make_moons(n_data, noise=0.05)
X, Y = jnp.asarray(X), jnp.asarray(Y)[:, jnp.newaxis]
scaler = StandardScaler()
X = scaler.fit_transform(X)

_, data_dim = X.shape
_, y_dim = Y.shape

maf = MAF(data_dim, hidden_dim, n_layers, y_dim, key=key)

opt = optax.adam(lr)
opt_state = opt.init(eqx.filter(maf, eqx.is_array))

losses = []
for i in range(n_steps):
    key = jr.fold_in(key, i)
    maf, opt_state, loss = make_step(maf, X, Y, opt_state, opt)
    losses += [loss]
    print(f"\r{i=}, {loss=:.3f}", end="")

keys = jr.split(key, n_samples)
Q = jr.choice(key, jnp.array([0., 1.]), (len(keys),))[:, None]
samples, _ = jax.vmap(maf.sample)(keys, Q)

plt.figure(dpi=200)
plt.hist2d(*samples.T, bins=200, cmap="PuOr")
plt.xlim(-2., 2.)
plt.ylim(-2., 2.)
plt.axis("off")
plt.savefig("samples.png")
plt.close()

plt.figure(dpi=200)
plt.semilogy(losses)
plt.savefig("loss.png")
plt.close()