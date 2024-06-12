from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    if mask_type == 'input':
        in_degrees = jnp.arange(in_features) % in_flow_features # Check this gives correct range...
    else:
        in_degrees = jnp.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = jnp.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = jnp.arange(out_features) % (in_flow_features - 1)
    return (out_degrees[..., None] >= in_degrees[None, ...])


class MaskedLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    mask: jax.Array
    cond_linear: Optional[eqx.nn.Linear] = None

    def __init__(
        self,
        in_features,
        out_features,
        mask,
        cond_in_features=None,
        *,
        key
    ):
        key_w, key_c = jr.split(key)
        self.weight = jr.normal(key_w, (out_features, in_features))
        self.bias = jnp.zeros((out_features,))

        if cond_in_features is not None:
            self.cond_linear = eqx.nn.Linear(
                cond_in_features, out_features, use_bias=False, key=key_c
            )
        self.mask = mask

    def __call__(self, inputs, cond_inputs=None, key=None):
        output = (self.weight * self.mask) @ inputs + self.bias
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


class MADE(eqx.Module):
    conditioner: eqx.nn.Linear
    shift_and_scale: eqx.nn.Sequential

    def __init__(
        self,
        input_dim,
        hidden_width,
        y_dim=None,
        activation=jax.nn.tanh,
        *,
        key
    ):
        input_mask = get_mask(
            input_dim, hidden_width, input_dim, mask_type='input'
        )
        hidden_mask = get_mask(
            hidden_width, hidden_width, input_dim
        )
        output_mask = get_mask(
            hidden_width, input_dim * 2, input_dim, mask_type='output'
        )

        keys = jr.split(key, 3)

        self.conditioner = MaskedLinear(
            input_dim, hidden_width, input_mask, y_dim, key=keys[0]
        )

        self.shift_and_scale = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(activation),
                MaskedLinear(hidden_width, hidden_width, hidden_mask, key=keys[1]), 
                eqx.nn.Lambda(activation),
                MaskedLinear(hidden_width, input_dim * 2, output_mask, key=keys[2])
            ]
        )

    def forward(self, inputs, cond_inputs=None):
        h = self.conditioner(inputs, cond_inputs)
        m, a = jnp.split(self.shift_and_scale(h), 2)
        u = (inputs - m) * jnp.exp(-a)
        return u, -a.sum()

    def reverse(self, inputs, cond_inputs=None):
        x = jnp.zeros(inputs.shape)
        for dim in range(x.size):
            h = self.conditioner(x, cond_inputs)
            m, a = jnp.split(self.shift_and_scale(h), 2)
            x = x.at[dim].set(inputs[dim] * jnp.exp(a[dim]) + m[dim])
        return x, -a.sum()