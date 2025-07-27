from flax import linen as nn
import jax.numpy as jnp

class MLP(nn.Module):
    '''
    MLP for MNIST dataset using Flax's linen API.

    Assumptions
        - Input is a flattened 28x28 image.
        - Output is a 10-dimensional vector.
    '''
    hidden_dim: int
    depth: int = 3

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth - 1):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x