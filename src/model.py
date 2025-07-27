from flax import nnx

class MLP(nnx.Module):
    '''
    MLP for MNIST dataset.

    Assumptions
        - Input is a flattened 28x28 image.
        - Output is a 10-dimensional vector.
    '''

    def __init__(self, 
                 hidden_dim: int, 
                 seed: int = 42):
        self.linear1 = nnx.Linear(in_features=28 * 28, out_features=hidden_dim, rngs=nnx.Rngs(seed))
        self.linear2 = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=nnx.Rngs(seed))
        self.linear3 = nnx.Linear(in_features=hidden_dim, out_features=10, rngs=nnx.Rngs(seed))

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        # Remove softmax - let the loss function handle it

        return x