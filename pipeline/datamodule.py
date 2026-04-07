"""PyTorch Lightning DataModule for flood prediction dataset."""
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from typing import Optional, Tuple, Union
import warnings


class FloodDataset(Dataset):
    """
    Custom Dataset for loading flood prediction data.
    
    Handles stacked satellite time series data with corresponding fractional flood labels.
    Each sample consists of:
        - Input: MODIS time series (10 time steps) + HAND + static features
        - Labels: Fractional flood inundation map (0.0 = no flood, 1.0 = fully flooded)
    
    Expected shapes:
        - x_data: (N, time_steps, channels, height, width)
                  N = number of samples (patches/tiles)
                  time_steps = sequence length (typically 10 for 8-day MODIS composites)
                  channels = input channels (typically 8: 7 MODIS bands + 1 HAND)
                  height, width = spatial dimensions at 500m resolution
        - y_labels: (N, height, width) OR (N,) 
                    Pixel-wise fractional labels OR aggregated per-patch labels
    
    Example:
        >>> # Pixel-wise labels
        >>> x = np.random.rand(100, 10, 8, 32, 32)  # 100 patches, 10 timesteps, 8 channels, 32x32 pixels
        >>> y = np.random.rand(100, 32, 32)  # Fractional inundation map
        >>> dataset = FloodDataset(x, y)
    """
    
    def __init__(
        self,
        x_data: Union[np.ndarray, torch.Tensor],
        y_labels: Union[np.ndarray, torch.Tensor],
        aggregate_labels: bool = False
    ):
        """
        Initialize the FloodDataset.
        
        Args:
            x_data: Input data array/tensor of shape (N, time, channels, h, w)
            y_labels: Label data array/tensor of shape (N, h, w) or (N,)
            aggregate_labels: If True and y_labels is (N, h, w), aggregate to (N,) by taking mean
                             This is useful if model predicts single value per sample
        """
        # Convert to numpy for validation, then to tensor
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.numpy()
        if isinstance(y_labels, torch.Tensor):
            y_labels = y_labels.numpy()
        
        # Validate input shapes
        if x_data.ndim != 5:
            raise ValueError(
                f"x_data must have 5 dimensions (N, time, channels, h, w), "
                f"got {x_data.ndim} dimensions with shape {x_data.shape}"
            )
        
        if y_labels.ndim not in [1, 3]:
            raise ValueError(
                f"y_labels must have 1 dimension (N,) or 3 dimensions (N, h, w), "
                f"got {y_labels.ndim} dimensions with shape {y_labels.shape}"
            )
        
        # Check that number of samples matches
        if x_data.shape[0] != y_labels.shape[0]:
            raise ValueError(
                f"Number of samples must match: x_data has {x_data.shape[0]} samples, "
                f"y_labels has {y_labels.shape[0]} samples"
            )
        
        # Handle label aggregation if needed
        if y_labels.ndim == 3 and aggregate_labels:
            # Aggregate pixel-wise labels to single value per sample (mean fractional inundation)
            y_labels = np.mean(y_labels, axis=(1, 2))
            warnings.warn(
                "Aggregating pixel-wise labels to per-sample values. "
                "Ensure your model outputs single value per sample.",
                UserWarning
            )
        
        # Convert to tensors
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.float32)
        
        # Store metadata
        self.n_samples = x_data.shape[0]
        self.sequence_length = x_data.shape[1]
        self.n_channels = x_data.shape[2]
        self.spatial_shape = (x_data.shape[3], x_data.shape[4])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            Tuple of (input_tensor, label_tensor)
            - input_tensor: Shape (time, channels, height, width)
            - label_tensor: Shape (height, width) or () depending on label format
        """
        return self.x[idx], self.y[idx]
    
    def get_info(self) -> dict:
        """Get information about the dataset."""
        return {
            "n_samples": self.n_samples,
            "sequence_length": self.sequence_length,
            "n_channels": self.n_channels,
            "spatial_shape": self.spatial_shape,
            "input_shape": self.x.shape,
            "label_shape": self.y.shape
        }


class FloodDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for flood prediction.
    
    Manages data loading, splitting, and DataLoader creation for training and validation.
    Integrates with PyTorch Lightning's training pipeline for automatic handling of
    data loading across training, validation, and testing phases.

    Args:
        x_train: Training input data (N_train, time, channels, h, w)
        y_train: Training labels (N_train, h, w) or (N_train,)
        x_val: Validation input data (N_val, time, channels, h, w)
        y_val: Validation labels (N_val, h, w) or (N_val,)
        batch_size: Batch size for DataLoaders (default: 32)
        num_workers: Number of worker processes for data loading - helps prevent memory leaks (default: 4)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        aggregate_labels: If True, aggregate pixel-wise labels to per-sample values

    """
    
    def __init__(
        self,
        x_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        x_val: Union[np.ndarray, torch.Tensor],
        y_val: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
        num_workers: int = 4, # if struggling with memory drop to 2 or 1
        pin_memory: bool = True,
        aggregate_labels: bool = False
    ):
        """
        Initialize the FloodDataModule.
        
        Args:
            x_train: Training input data (N_train, time, channels, h, w)
            y_train: Training labels (N_train, h, w) or (N_train,)
            x_val: Validation input data (N_val, time, channels, h, w)
            y_val: Validation labels (N_val, h, w) or (N_val,)
            batch_size: Batch size for DataLoaders (default: 32)
            num_workers: Number of worker processes for data loading (default: 4)
            pin_memory: Whether to pin memory for faster GPU transfer (default: True)
            aggregate_labels: If True, aggregate pixel-wise labels to per-sample values
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.aggregate_labels = aggregate_labels
        
        # Store raw data (will be converted to datasets in setup)
        self.train_data = (x_train, y_train)
        self.val_data = (x_val, y_val)
        
        # Datasets (initialized in setup)
        self.train_ds: Optional[FloodDataset] = None
        self.val_ds: Optional[FloodDataset] = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for the current stage (fit, validate, test, predict).
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
                   If None, sets up for all stages
        """
        if stage == "fit" or stage is None:
            # Create training and validation datasets
            self.train_ds = FloodDataset(
                *self.train_data,
                aggregate_labels=self.aggregate_labels
            )
            self.val_ds = FloodDataset(
                *self.val_data,
                aggregate_labels=self.aggregate_labels
            )
            
            # Print dataset info
            print("Training dataset info:")
            print(self.train_ds.get_info())
            print("\nValidation dataset info:")
            print(self.val_ds.get_info())
    
    def train_dataloader(self) -> DataLoader:
        """
        Create DataLoader for training data.
        
        Returns:
            DataLoader configured for training (with shuffling)
        """
        if self.train_ds is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Create DataLoader for validation data.
        
        Returns:
            DataLoader configured for validation (no shuffling)
        """
        if self.val_ds is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )