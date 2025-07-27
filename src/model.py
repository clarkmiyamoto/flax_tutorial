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
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        # self.intermediate = x  # Removed - causes Flax error
        x = nn.Dense(10 + self.auxiliary)(x)

        # Return raw logits, not softmax outputs
        return x