from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from made import MADE


class Permute(eqx.Module):
    perm: jax.Array
    reverse_perm: jax.Array

    def __init__(self, nvars, perm=None, *, key):
        # If perm is none, chose some random permutation that gets fixed at initialization
        if perm is None:
            perm = jr.permutation(key, jnp.arange(nvars))
        self.perm = perm
        self.reverse_perm = jnp.argsort(perm)

    def forward(self, x, y):
        idx = self.perm
        return x[idx], 0

    def reverse(self, x, y):
        rev_idx = self.reverse_perm
        return x[rev_idx], 0


class MAF(eqx.Module):
    n_inputs: int
    layers: Tuple[eqx.Module]

    def __init__(self, n_inputs, *layers):
        self.n_inputs = n_inputs
        self.layers = tuple(layers)

    def forward(self, x, y=None):
        log_det = jnp.zeros(x.shape)
        for layer in self.layers:
            x, _log_det = layer.forward(x, y)
            log_det += _log_det
        for layer in self.layers[::-1]:
            if isinstance(layer, Permute):
                continue
        return x, log_det


    def reverse(self, z, y=None):
        for layer in self.layers:
            if isinstance(layer, Permute):
                continue
        log_det = jnp.zeros(z.shape)
        for layer in self.layers[::-1]:
            z, _log_det = layer.reverse(z, y)
            log_det += _log_det
        return z, log_det

    def sample(self, key, y=None):
        u = jr.normal(key, (self.n_inputs,))
        samples, log_det = self.reverse(u, y)
        return samples, log_det

    def log_prob(self, x, y=None):
        u, log_jacob = self.forward(x, y=y)
        log_probs = (-0.5 * (u ** 2) - 0.5 * jnp.log(2. * jnp.pi)).sum()
        return (log_probs + log_jacob).sum()

    def loss(self, x, y):
        return -self.log_prob(x=x, y=y)

    def prior_log_prob(self, z, y=None):
        log_probs = (-0.5 * (z ** 2) - 0.5 * jnp.log(2. * jnp.pi)).sum()
        return log_probs.sum()

    def forward_and_log_prob(self, x, y=None):
        u, log_jacob = self.forward(x, y=y)
        log_probs = (-0.5 * (u ** 2) - 0.5 * jnp.log(2. * jnp.pi)).sum()
        return u, (log_probs + log_jacob).sum()