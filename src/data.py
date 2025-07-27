from jax._src.dtypes import int2
from jaxlib.mlir.dialects.scf import NoneType
import torch
from torchvision import datasets, transforms

import jax
import jax.numpy as jnp
from typing import Optional


class Dataset:

    @staticmethod
    def _datasetToJax(dataset, 
                 transform):
        '''
        Get data from PyTorch dataset.
        '''
        dataset_train = dataset(root='./data', train=True, download=True, transform=transform) 
        dataset_test = dataset(root='./data', train=False, download=True, transform=transform)

        TensorToJax = lambda x: jnp.array(x.cpu().numpy()) # Convert PyTorch tensor to JAX array

        X_train = TensorToJax(dataset_train.data) # Shape (batch_size, *dim)
        y_train = TensorToJax(dataset_train.train_labels) # Shape (batch_size,)
        
        X_test = TensorToJax(dataset_test.data) # Shape (batch_size, *dim)
        y_test = TensorToJax(dataset_test.test_labels) # Shape (batch_size,)

        # Flatten the images to match model input shape
        X_train = X_train.reshape(X_train.shape[0], -1)  # Shape (N, 784)
        X_test = X_test.reshape(X_test.shape[0], -1)     # Shape (N, 784)

        return X_train, y_train, X_test, y_test
    

    @staticmethod
    def load_MNIST():
        transform = transforms.Compose([
            transforms.ToTensor(),  # converts to (1, 28, 28) tensor with float32 in [0,1]
        ])
        dataset = lambda **kwargs: datasets.MNIST(**kwargs)
        X_train, y_train, X_test, y_test = Dataset._datasetToJax(dataset, transform)

        return X_train, y_train, X_test, y_test



class DataLoader:
    """
    Jax friendly DataLoader that mimics PyTorch's DataLoader.
    """

    def __init__(
        self,
        features: jnp.ndarray,
        labels:   jnp.ndarray,
        batch_size: int,
        shuffle: bool = False,
        seed: int = None,
    ):
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Must have same number of features and labels.")
        if shuffle and seed is None:
            raise ValueError('seed is required when shuffle is True.')
        elif shuffle and seed is not None:
            self._key = jax.random.PRNGKey(seed)

        self.features  = features
        self.labels    = labels
        self.batch_size = batch_size
        self.shuffle    = shuffle
        if shuffle:
            self._shuffle_data()
        self.features_batched, self.labels_batched = self._make_batches()

    def _shuffle_data(self):
        self._key, subkey = jax.random.split(self._key)
        perm = jax.random.permutation(subkey, self.features.shape[0])
        self.features = self.features[perm]
        self.labels   = self.labels[perm]
    
    def _make_batches(self) -> jnp.ndarray:
        dataset_size = self.features.shape[0]
        num_batches = dataset_size // self.batch_size  # Drop remainder
        trimmed_features = self.features[:num_batches * self.batch_size]
        trimmed_labels   = self.labels[:num_batches * self.batch_size]
        features_batched = trimmed_features.reshape((num_batches, self.batch_size) + self.features.shape[1:])
        labels_batched   = trimmed_labels.reshape((num_batches, self.batch_size) + self.labels.shape[1:])

        return features_batched, labels_batched # Shape (num_batches, batch_size, *dim)

    def __len__(self) -> int:
        """Number of data points."""
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features_batched[idx], self.labels_batched[idx]

    def __iter__(self):
        """Make DataLoader iterable."""
        for i in range(len(self.features_batched)):
            yield self.features_batched[i], self.labels_batched[i]