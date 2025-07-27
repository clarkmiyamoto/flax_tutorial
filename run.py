import os
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
from flax import nnx, linen
import optax
from tqdm import tqdm

from src.data import Dataset, DataLoader
from src.model import MLP
from src.train import Trainer



def init_models(architecture: linen.Module, rngs: jax.random.PRNGKey):
    student: linen.Module = architecture(auxiliary=3)
    teacher: linen.Module = student.copy()

    return student, teacher

if __name__ == "__main__":
    seed = 42
   
    # Load Data
    batch_size = 100
    X_train, y_train, X_test, y_test = Dataset.load_MNIST()
    dataloader_train = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    dataloader_test = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # Initalize two models w/ same weights
    student, teacher = init_models(MLP, jax.random.PRNGKey(seed))

    # Optimizer
    lr = 1e-3
    optimizer_teacher = optax.adam(lr)
    optimizer_student = optax.adam(lr)

    # Loss
    def loss_teacher(logits, batch_y):
        logits_teacher = logits[:, :10]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_teacher, batch_y).mean()
        accuracy = jnp.mean(jnp.argmax(logits_teacher, axis=-1) == batch_y)
        return loss, accuracy

    def loss_student(student_logits, teacher_logits, batch_y):
        # student_logits and teacher_logits are full outputs [batch_size, 13]
        # Extract auxiliary outputs (last 3 dimensions)
        student_auxiliary = student_logits[:, 10:]  # These are already softmax outputs
        teacher_auxiliary = teacher_logits[:, 10:]  # These are already softmax outputs
        
        # Distillation loss: cross-entropy between auxiliary outputs
        distillation_loss = -jnp.sum(teacher_auxiliary * jnp.log(student_auxiliary + 1e-8), axis=-1).mean()
        
        # Supervised loss: cross-entropy on main outputs with ground truth
        student_main = student_logits[:, :10]
        supervised_loss = optax.softmax_cross_entropy_with_integer_labels(student_main, batch_y).mean()
        
        # Combined loss with weighting
        alpha = 0.7  # Weight for distillation loss
        total_loss = alpha * distillation_loss + (1 - alpha) * supervised_loss
        
        # Calculate accuracy based on main outputs vs ground truth
        accuracy = jnp.mean(jnp.argmax(student_main, axis=-1) == batch_y)
        
        return total_loss, accuracy

    # Train teacher
    epochs = 5

    trainer = Trainer(
        teacher=teacher,
        student=student,
        dataloader=dataloader_train,
        optimizer_teacher=optimizer_teacher,
        optimizer_student=optimizer_student,
        loss_teacher=loss_teacher,
        loss_student=loss_student  # Loss is defined inside the trainer
    )
    print("Training teacher...")
    trainer.train_teacher(epochs=epochs, eval_dataloader=dataloader_test)
    test_loss, test_acc = trainer.evaluate_teacher(dataloader_test)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}") 

    print("Training student...")
    trainer.train_student(epochs=epochs, eval_dataloader=dataloader_test)
    test_loss, test_acc = trainer.evaluate_student(dataloader_test)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}") 


    