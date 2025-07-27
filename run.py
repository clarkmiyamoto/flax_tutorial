import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from src.data import Dataset, DataLoader
from src.model import MLP
from src.train import Trainer




if __name__ == "__main__":
    seed = 42
   
    # Load Data
    batch_size = 128
    X_train, y_train, X_test, y_test = Dataset.load_MNIST()
    dataloader_train = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    dataloader_test = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # Define model
    hidden_dim = 100
    model = MLP(hidden_dim=hidden_dim, seed=seed)

    # Optimizer
    lr = 1e-3
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    # Metrics
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    # Loss function
    def loss_fn(model: nnx.Module, batch):
        logits = model(batch['image'])
        loss = optax.softmax_cross_entropy(logits, batch['label']).mean()

        return loss, logits

    # Trainer
    epochs = 5
    trainer = Trainer(model, dataloader_train, optimizer, metrics, loss_fn)
    trainer._train_step(dataloader_train[0], dataloader_train[1])





