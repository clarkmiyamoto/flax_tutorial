import os
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from src.data import Dataset, DataLoader
from src.model import MLP
from src.train import Trainer

if __name__ == "__main__":
    seed = 42
   
    # Load Data
    batch_size = 100
    X_train, y_train, X_test, y_test = Dataset.load_FashionMNIST()
    dataloader_train = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    dataloader_test = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # Define model (using JIT-compatible version)
    hidden_dim = 2**9
    depth = 6
    model = MLP(hidden_dim=hidden_dim, depth=depth)

    # Optimizer
    lr = 1e-3
    optimizer = optax.adam(lr)

    # Loss
    def loss_fn(logits, batch_y):
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch_y)
        return loss, accuracy

    # Initialize trainer
    trainer = Trainer(
        model=model,
        dataloader=dataloader_train,
        optimizer=optimizer,
        loss=loss_fn  # Loss is defined inside the trainer
    )

    # Train the model with validation
    print("Starting Training...")
    epochs = 5
    trainer.train(epochs=epochs, eval_dataloader=dataloader_test)
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_loss, test_acc = trainer.evaluate(dataloader_test)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}") 