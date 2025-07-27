import jax
from flax import nnx
from functools import partial

from tqdm import tqdm
from typing import Callable

from .data import DataLoader

class Trainer:
    def __init__(self, 
                 model: nnx.Module, 
                 dataloader: DataLoader,
                 optimizer: nnx.Optimizer, 
                 metrics: nnx.MultiMetric,
                 loss: Callable):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss

    def train(self, epochs: int):
        '''
        Train the model for a given number of epochs.
        '''
        for _ in tqdm(range(epochs)):
            # One epoch loop
            for x_batch, y_batch in self.dataloader:
                self._train_step(x_batch, y_batch)
                # self._eval_step(batch)

    @partial(jax.jit, static_argnames=["self"])
    def _train_step(self, feature, label):
        """Train for a single step."""
        grad_fn = nnx.value_and_grad(self.loss, has_aux=True)
        (loss, logits), grads = grad_fn(self.model, (feature, label))
        self.metrics.update(loss=loss, logits=logits, labels=label)  # In-place updates.
        self.optimizer.update(self.model, grads)  # In-place updates.

    # @nnx.jit
    # def _eval_step(self, batch: dict):
    #     """Evaluate for a single step."""
    #     loss, logits = self.loss_fn(self.model, batch)
    #     self.metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.