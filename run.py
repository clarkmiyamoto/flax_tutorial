import os
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import jax.numpy as jnp
from flax import nnx, linen
import optax
from tqdm import tqdm
import argparse

from src.data import Dataset, DataLoader
from src.model import MLP
from src.train import Trainer



def init_models(architecture: linen.Module, rngs: jax.random.PRNGKey):
    student: linen.Module = architecture(auxiliary=3)
    teacher: linen.Module = student.copy()

    return student, teacher

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--wandb_project", type=str, default="flax-knowledge-distillation")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
   
    # Load Data
    batch_size = args.batch_size  # Reduced from 200 for better distillation
    X_train, y_train, X_test, y_test = Dataset.load_MNIST()
    dataloader_train = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    dataloader_test = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # Initalize two models w/ same weights
    student, teacher = init_models(MLP, jax.random.PRNGKey(seed))

    # Optimizer - Much more aggressive learning rate for student
    lr_teacher = 1e-4  # Increased back to 1e-3 for better teacher
    lr_student = 1e-4  # Much higher learning rate for student
    
    # Create learning rate scheduler for student
    # lr_student = optax.cosine_decay_schedule(
    #     init_value=lr_student,
    #     decay_steps=100 * 300,  # 100 epochs * ~300 batches per epoch
    #     alpha=0.1  # Final LR will be 10% of initial
    # )
    
    optimizer_teacher = optax.adam(lr_teacher)
    optimizer_student = optax.chain(
        # optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(lr_student, weight_decay=args.weight_decay)
    )

    # Loss
    def loss_teacher(logits, batch_y):
        logits_teacher = logits[:, :10]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_teacher, batch_y).mean()
        accuracy = jnp.mean(jnp.argmax(logits_teacher, axis=-1) == batch_y)
        return loss, accuracy

    def loss_student(student_logits, teacher_logits, batch_y):
        # student_logits and teacher_logits are full outputs [batch_size, 13]
        # Extract auxiliary outputs (last 3 dimensions)
        student_auxiliary = student_logits[:, 10:]  # These are raw logits now
        teacher_auxiliary = teacher_logits[:, 10:]  # These are raw logits now
        
        # Apply temperature scaling for better distillation
        temperature = 3.0  # Higher temperature makes distributions softer
        student_auxiliary_softmax = jax.nn.softmax(student_auxiliary / temperature)
        teacher_auxiliary_softmax = jax.nn.softmax(teacher_auxiliary / temperature)
        
        # Distillation loss: cross-entropy between auxiliary outputs
        distillation_loss = -jnp.sum(teacher_auxiliary_softmax * jnp.log(student_auxiliary_softmax + 1e-8), axis=-1).mean()
        
        # Calculate accuracy based on main outputs vs ground truth
        student_main = student_logits[:, :10]
        accuracy = jnp.mean(jnp.argmax(student_main, axis=-1) == batch_y)
        
        return distillation_loss, accuracy

    # Train teacher - even more epochs for better teacher
    epoch_teacher = 5  # Increased from 15
    epoch_student = 1000

    trainer = Trainer(
        teacher=teacher,
        student=student,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer_teacher=optimizer_teacher,
        optimizer_student=optimizer_student,
        loss_teacher=loss_teacher,
        loss_student=loss_student,  # Loss is defined inside the trainer
        seed=seed,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        log_wandb=not args.no_wandb
    )
    
    # Log hyperparameters to wandb
    if not args.no_wandb:
        import wandb
        wandb.config.update({
            "lr_teacher": lr_teacher,
            "lr_student": lr_student,
            "weight_decay": args.weight_decay,
            "batch_size": batch_size,
            "epoch_teacher": epoch_teacher,
            "epoch_student": epoch_student,
        })
    
    print("Training teacher...")
    trainer.train_teacher(epochs=epoch_teacher)
    test_loss, test_acc = trainer.evaluate_teacher(dataloader_test)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}") 

    # Copy teacher weights to student for better initialization
    print("Copying teacher weights to student...")
    trainer.params_student = trainer.params_teacher.copy()
    trainer.opt_state_student = trainer.optimizer_student.init(trainer.params_student)

    print("Training student...")
    trainer.train_student(epochs=epoch_student)
    test_loss, test_acc = trainer.evaluate_student(dataloader_test)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}") 

    # Get final metrics and finish wandb run
    final_metrics = trainer.get_final_metrics()
    if final_metrics:
        print("Final metrics:", final_metrics)
    
    trainer.finish() 


    