import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from tqdm import tqdm
from typing import Callable, Tuple
import time

from .data import DataLoader


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 dataloader: DataLoader,
                 optimizer: optax.GradientTransformation,
                 loss: Callable,
                 use_pmap: bool = False):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.use_pmap = use_pmap
        
        # Initialize model parameters
        self.params = None
        self.opt_state = None
        self._init_model()
        
        # Create jitted functions with different optimizations
        if use_pmap and jax.device_count() > 1:
            # Use pmap for multi-device training
            self._jit_train_step = jax.pmap(self._train_step_jitted, axis_name='batch')
            self._jit_eval_step = jax.pmap(self._eval_step_jitted, axis_name='batch')
        else:
            # Use regular JIT for single device
            self._jit_train_step = jax.jit(self._train_step_jitted)
            self._jit_eval_step = jax.jit(self._eval_step_jitted)

    def _init_model(self):
        """Initialize model parameters and optimizer state."""
        # Create dummy input to initialize parameters
        dummy_input = jnp.ones((1, 784))
        self.params = self.model.init(jax.random.PRNGKey(0), dummy_input)
        self.opt_state = self.optimizer.init(self.params)

    def train(self, epochs: int, eval_dataloader=None):
        '''
        Train the model for a given number of epochs.
        '''
        print(f"Training with JIT compilation on {jax.device_count()} device(s)")
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training loop
            train_losses = []
            train_accuracies = []
            epoch_start_time = time.time()
            
            for x_batch, y_batch in self.dataloader:
                loss, accuracy, self.params, self.opt_state = self._jit_train_step(
                    self.params, self.opt_state, x_batch, y_batch
                )
                train_losses.append(loss)
                train_accuracies.append(accuracy)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            avg_train_loss = jnp.mean(jnp.array(train_losses))
            avg_train_acc = jnp.mean(jnp.array(train_accuracies))
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} (Time: {epoch_time:.2f}s)")
            
            # Evaluate on validation set if provided
            if eval_dataloader is not None:
                val_loss, val_acc = self.evaluate(eval_dataloader)
                print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def evaluate(self, dataloader):
        '''
        Evaluate the model on the given dataloader.
        '''
        losses = []
        accuracies = []
        
        for x_batch, y_batch in dataloader:
            loss, accuracy = self._jit_eval_step(self.params, x_batch, y_batch)
            losses.append(loss)
            accuracies.append(accuracy)
        
        return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))

    def _train_step_jitted(self, params, opt_state, batch_x, batch_y):
        """Jitted training step."""
        def loss_fn(params, batch_x, batch_y):
            logits = self.model.apply(params, batch_x)    
            loss, accuracy = self.loss(logits, batch_y)
            return loss, accuracy
        
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch_x, batch_y)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return loss, accuracy, new_params, new_opt_state

    def _eval_step_jitted(self, params, batch_x, batch_y):
        """Jitted evaluation step."""
        logits = self.model.apply(params, batch_x)
        loss, accuracy = self.loss(logits, batch_y)
        return loss, accuracy 