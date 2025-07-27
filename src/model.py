from flax import linen as nn
import jax.numpy as jnp

class MLP(nn.Module):
    '''
    MLP in https://www.arxiv.org/pdf/2507.14805

    Assumptions
        - Input is a flattened 28x28 image.
        - Output is a 10-dimensional vector.
    '''
    auxiliary : int
    depth: int = 3

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth - 1):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        # Output 10 + auxiliary dimensions
        x = nn.Dense(10 + self.auxiliary)(x)

        # Softmax main & auxiliary outputs separately
        x = jnp.concatenate([nn.softmax(x[:, :10]), nn.softmax(x[:, 10:])], axis=-1)
        return x