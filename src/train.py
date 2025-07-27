import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from tqdm import tqdm
from typing import Callable, Tuple
import time
import wandb

from .data import DataLoader


class Trainer:
    def __init__(self, 
                 teacher: nn.Module, 
                 student: nn.Module, 
                 dataloader_train: DataLoader,
                 dataloader_test: DataLoader,
                 optimizer_teacher: optax.GradientTransformation,
                 optimizer_student: optax.GradientTransformation,
                 loss_teacher: Callable,
                 loss_student: Callable,
                 use_pmap: bool = False,
                 seed: int = 42,
                 wandb_project: str = "flax-knowledge-distillation",
                 wandb_name: str = None,
                 log_wandb: bool = True):
        self.teacher = teacher
        self.student = student
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.loss_teacher = loss_teacher
        self.loss_student = loss_student
        self.use_pmap = use_pmap
        self.seed = seed
        self.log_wandb = log_wandb

        # Initialize wandb if logging is enabled
        if self.log_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    "seed": seed,
                    "use_pmap": use_pmap,
                    "device_count": jax.device_count(),
                }
            )

        # Initialize model parameters
        self.params_teacher = None
        self.params_student = None
        self.opt_state_teacher = None
        self.opt_state_student  = None
        self._init_model()
        
        # Create jitted functions with different optimizations
        if use_pmap and jax.device_count() > 1:
            # Use pmap for multi-device training
            self._jit_train_step_teacher = jax.pmap(self._train_step_teacher, axis_name='batch')
            self._jit_eval_step_teacher = jax.pmap(self._eval_step_teacher, axis_name='batch')
            self._jit_train_step_student = jax.pmap(self._train_step_student, axis_name='batch')
            self._jit_eval_step_student = jax.pmap(self._eval_step_student, axis_name='batch')
        else:
            # Use regular JIT for single device
            self._jit_train_step_teacher = jax.jit(self._train_step_teacher)
            self._jit_eval_step_teacher = jax.jit(self._eval_step_teacher)
            self._jit_train_step_student = jax.jit(self._train_step_student)
            self._jit_eval_step_student = jax.jit(self._eval_step_student)

    def _init_model(self):
        """Initialize model parameters and optimizer state."""
        # Create dummy input to initialize parameters
        dummy_input = jnp.ones((1, 28*28))
        self.params_teacher = self.teacher.init(jax.random.PRNGKey(self.seed), dummy_input)
        self.params_student = self.student.init(jax.random.PRNGKey(self.seed), dummy_input)
        self.opt_state_teacher = self.optimizer_teacher.init(self.params_teacher)
        self.opt_state_student = self.optimizer_student.init(self.params_student)

    def train_teacher(self, epochs: int):
        '''
        Train the model for a given number of epochs.
        '''
        print(f"Training with JIT compilation on {jax.device_count()} device(s)")
        
        for epoch in tqdm(range(epochs), desc="Training Teacher"):
            # Training loop
            train_losses = []
            train_accuracies = []
            epoch_start_time = time.time()
            
            for x_batch, y_batch in self.dataloader_train:
                loss, accuracy, self.params_teacher, self.opt_state_teacher = self._jit_train_step_teacher(
                    self.params_teacher, self.opt_state_teacher, x_batch, y_batch
                )
                train_losses.append(loss)
                train_accuracies.append(accuracy)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            avg_train_loss = jnp.mean(jnp.array(train_losses))
            avg_train_acc = jnp.mean(jnp.array(train_accuracies))
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} (Time: {epoch_time:.2f}s)")
            
            # Log to wandb
            log_dict = {
                "epoch": epoch + 1,
                "teacher/train_loss": float(avg_train_loss),
                "teacher/train_acc": float(avg_train_acc),
                "teacher/epoch_time": epoch_time,
            }
            
            # Evaluate on validation set if provided
            if self.dataloader_test is not None:
                val_loss, val_acc = self.evaluate_teacher(self.dataloader_test)
                print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                log_dict.update({
                    "teacher/val_loss": float(val_loss),
                    "teacher/val_acc": float(val_acc),
                })
            
            if self.log_wandb:
                wandb.log(log_dict)

    def train_student(self, epochs: int):
        '''
        Train the model for a given number of epochs.
        '''
        print(f"Training with JIT compilation on {jax.device_count()} device(s)")
        
        for epoch in tqdm(range(epochs), desc="Training Student"):
            # Training loop
            train_losses = []
            train_accuracies = []
            epoch_start_time = time.time()
            
            for x_batch, y_batch in self.dataloader_train:
                logits_teacher = self.teacher.apply(self.params_teacher, x_batch)
                loss, accuracy, self.params_student, self.opt_state_student = self._jit_train_step_student(
                    self.params_student, self.opt_state_student, x_batch, logits_teacher, y_batch
                )
                train_losses.append(loss)
                train_accuracies.append(accuracy)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch results
            avg_train_loss = jnp.mean(jnp.array(train_losses))
            avg_train_acc = jnp.mean(jnp.array(train_accuracies))
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} (Time: {epoch_time:.2f}s)")
            
            # Log to wandb
            log_dict = {
                "epoch": epoch + 1,
                "student/train_loss": float(avg_train_loss),
                "student/train_acc": float(avg_train_acc),
                "student/epoch_time": epoch_time,
            }
            
            # Evaluate on validation set if provided
            if self.dataloader_test is not None:
                val_loss, val_acc = self.evaluate_student(self.dataloader_test)
                print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                log_dict.update({
                    "student/val_loss": float(val_loss),
                    "student/val_acc": float(val_acc),
                })
            
            if self.log_wandb:
                wandb.log(log_dict)

    def _train_step_teacher(self, params, opt_state, batch_x, batch_y):
        """Jitted training step."""
        def loss_fn(params, batch_x, batch_y):
            logits = self.teacher.apply(params, batch_x)  
            logits_teacher = logits[:, :10]
            loss, accuracy = self.loss_teacher(logits_teacher, batch_y)
            return loss, accuracy
        
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch_x, batch_y)
        updates, new_opt_state = self.optimizer_teacher.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return loss, accuracy, new_params, new_opt_state

    def _eval_step_teacher(self, params, batch_x, batch_y):
        """Jitted evaluation step."""
        logits = self.teacher.apply(params, batch_x)
        logits_teacher = logits[:, :10]
        loss, accuracy = self.loss_teacher(logits_teacher, batch_y)
        return loss, accuracy 

    def _train_step_student(self, params, opt_state, batch_x, teacher_logits, batch_y):
        """Jitted training step."""
        def loss_fn(params, batch_x, teacher_logits, batch_y):
            # Add noise augmentation for student training
            logits = self.student.apply(params, batch_x)    
            loss, accuracy = self.loss_student(logits, teacher_logits, batch_y)
            return loss, accuracy
        
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch_x, teacher_logits, batch_y)
        updates, new_opt_state = self.optimizer_student.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return loss, accuracy, new_params, new_opt_state

    def _eval_step_student(self, params, batch_x, teacher_logits, batch_y):
        """Jitted evaluation step."""
        logits = self.student.apply(params, batch_x)
        loss, accuracy = self.loss_student(logits, teacher_logits, batch_y)
        return loss, accuracy 

    def evaluate_teacher(self, dataloader):
        '''
        Evaluate the model on the given dataloader.
        '''
        losses = []
        accuracies = []
        
        for x_batch, y_batch in dataloader:
            loss, accuracy = self._jit_eval_step_teacher(self.params_teacher, x_batch, y_batch)
            losses.append(loss)
            accuracies.append(accuracy)
        
        return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))

    def evaluate_student(self, dataloader):
        '''
        Evaluate the student model on the given dataloader.
        '''
        losses = []
        accuracies = []
        
        for x_batch, y_batch in dataloader:
            logits_teacher = self.teacher.apply(self.params_teacher, x_batch)
            loss, accuracy = self._jit_eval_step_student(self.params_student, x_batch, logits_teacher, y_batch)
            losses.append(loss)
            accuracies.append(accuracy)
        
        return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))

    def finish(self):
        """Finish the wandb run."""
        if self.log_wandb:
            wandb.finish()

    def get_final_metrics(self):
        """Get final evaluation metrics for both teacher and student."""
        if self.dataloader_test is not None:
            teacher_val_loss, teacher_val_acc = self.evaluate_teacher(self.dataloader_test)
            student_val_loss, student_val_acc = self.evaluate_student(self.dataloader_test)
            
            final_metrics = {
                "teacher/final_val_loss": float(teacher_val_loss),
                "teacher/final_val_acc": float(teacher_val_acc),
                "student/final_val_loss": float(student_val_loss),
                "student/final_val_acc": float(student_val_acc),
            }
            
            if self.log_wandb:
                wandb.log(final_metrics)
            
            return final_metrics
        return None