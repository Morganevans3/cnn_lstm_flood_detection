"""The PyTorch Lightning CNN-LSTM class for flood prediction."""
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from typing import Optional

logger = logging.getLogger(__name__)


class FloodCNN_LSTM(pl.LightningModule):
    """
    CNN-LSTM model for predicting fractional flood inundation from satellite time series.
    
    Architecture:
        1. CNN (Spatial Feature Extractor): Extracts spatial patterns from each time step
           - Processes MODIS bands + HAND + static features at 500m resolution
           - Outputs compressed spatial representations (4x4 grid)
        
        2. LSTM (Temporal Sequencer): Captures temporal dynamics across the sequence
           - Processes the spatial features from 10 consecutive time steps
           - Models how landscape conditions evolve leading up to the flood event
        
        3. Regression Head: Predicts fractional inundation (0.0 to 1.0)
           - Fractional means: if 50% of a 500m pixel is flooded, label = 0.5
    
    Input:
        - MODIS 8-day composites: 7 spectral bands (typically)
        - HAND (Height Above Nearest Drainage): 1 band
        - Total: 8 channels per time step (configurable)
        - Sequence: 10 time steps (~80 days, capturing pre-flood conditions)
        - Resolution: 500m (MODIS native resolution)
    
    Output:
        - Fractional flood inundation map (0.0 = no flood, 1.0 = fully flooded)
        - Each pixel represents the percentage of that 500m area that is inundated
    
    Example:
        >>> model = FloodCNN_LSTM(in_channels=8, sequence_length=10)
        >>> # Input shape: (batch, 10, 8, height, width)
        >>> # Output shape: (batch, height, width) with values in [0, 1]
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        sequence_length: int = 10,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        cnn_out_channels: int = 64,
        spatial_output_size: int = 4
    ):
        """
        Initialize the FloodCNN_LSTM model.
        
        Args:
            in_channels: Number of input channels per time step
                        (typically 7 MODIS bands + 1 HAND = 8)
            sequence_length: Number of time steps in the sequence (typically 10)
            hidden_dim: LSTM hidden dimension (default: 128)
            lr: Learning rate for optimizer (default: 1e-4)
            cnn_out_channels: Number of output channels from CNN (default: 64)
            spatial_output_size: Spatial grid size after CNN pooling (default: 4x4)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Spatial Feature Extractor (CNN)
        # Processes each time step independently to extract spatial patterns
        # Input: (batch*time, channels, height, width) at 500m resolution
        # Output: (batch*time, cnn_out_channels, spatial_output_size, spatial_output_size)
        # can change the kernel size and stride to be more efficient
        self.cnn = nn.Sequential(
            # First convolutional block: extract basic spatial features
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # if struggling with memory drop 32 down to 16 or 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions by 2
            
            # Second convolutional block: extract higher-level features
            nn.Conv2d(32, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling: standardizes output to fixed spatial size
            # This allows the model to handle variable input image sizes
            nn.AdaptiveAvgPool2d((spatial_output_size, spatial_output_size))
        )
        
        # Calculate flattened feature size after CNN
        self.cnn_output_size = cnn_out_channels * spatial_output_size * spatial_output_size
        
        # 2. Temporal Sequencer (LSTM)
        # Processes the sequence of spatial features to capture temporal dynamics
        # Input: (batch, time, flattened_cnn_features)
        # Output: (batch, time, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_dim,
            num_layers=2,  # 2 layers for better temporal modeling
            batch_first=True,  # Input format: (batch, time, features)
            dropout=0.1  # Dropout between LSTM layers
        )
        
        # 3. Regression Head - Spatial Prediction
        # Predicts fractional inundation (0.0 to 1.0) for each pixel
        # We need to output spatial maps, so we use transpose convolutions
        # to upscale from LSTM output to full spatial resolution
        self.spatial_output_size = spatial_output_size  # Save for later
        
        # Upsample LSTM features back to spatial dimensions
        # hidden_dim -> h*w for spatial prediction
        # Use View/Reshape instead of Unflatten for better compatibility
        self.regressor_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, spatial_output_size * spatial_output_size)
        )
        
        # Spatial upsampling network
        self.regressor_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),  # Final prediction layer
            nn.Sigmoid()  # Ensures output is in [0, 1] for fractional labels
        )
        
        # Metrics for evaluation
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, channels, height, width)
               - batch: Batch size
               - sequence_length: Number of time steps (typically 10)
               - channels: Number of input channels per time step (typically 8)
               - height, width: Spatial dimensions at 500m resolution
        
        Returns:
            Predictions of shape (batch, 1) or (batch, height*width) depending on output shape
            Values are in [0, 1] representing fractional inundation
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape for CNN: process all time steps as independent images
        # (batch, time, channels, height, width) -> (batch*time, channels, height, width)
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        
        # Extract spatial features from each time step
        # Output: (batch*time, cnn_out_channels, spatial_output_size, spatial_output_size)
        spatial_features = self.cnn(x_reshaped)
        
        # Flatten spatial features and reshape for LSTM
        # (batch*time, cnn_out_channels, 4, 4) -> (batch, time, flattened_features)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Process temporal sequence with LSTM
        # Output: (batch, time, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(spatial_features)
        
        # Use the last time step's output for prediction
        # This represents the model's understanding after seeing the full sequence
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Project to spatial feature map
        spatial_features = self.regressor_linear(last_out)  # (batch, spatial_output_size * spatial_output_size)
        
        # Reshape to spatial dimensions
        batch_size = spatial_features.shape[0]
        spatial_features = spatial_features.view(batch_size, 1, self.spatial_output_size, self.spatial_output_size)
        
        # Upsample to target spatial dimensions
        prediction = self.regressor_conv(spatial_features)  # (batch, 1, upsampled_h, upsampled_w)
        
        # Remove channel dimension: (batch, 1, h, w) -> (batch, h, w)
        prediction = prediction.squeeze(1)
        
        # Resize to match input spatial dimensions if needed
        pred_h, pred_w = prediction.shape[1:]
        if pred_h != h or pred_w != w:
            # Interpolate to match exact dimensions
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            # Ensure no NaN from interpolation
            if torch.isnan(prediction).any():
                logger.warning("NaN after interpolation, replacing with zeros")
                prediction = torch.nan_to_num(prediction, nan=0.0)
        
        # Clip to valid range and check for NaN
        prediction = torch.clamp(prediction, 0.0, 1.0)
        if torch.isnan(prediction).any():
            logger.warning("NaN detected in model prediction! Replacing with zeros.")
            prediction = torch.nan_to_num(prediction, nan=0.0)
        
        return prediction  # (batch, h, w)
    
    def training_step(self, batch, batch_idx):
        """
        Training step with loss calculation and metric logging.
        
        Args:
            batch: Tuple of (input_tensor, target_labels)
            batch_idx: Index of the batch
        
        Returns:
            Training loss
        """
        x, y = batch
        
        # Check for NaN/Inf in input and replace with safe values
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or Inf detected in input! Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.warning("NaN or Inf detected in labels! Replacing with zeros.")
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        
        y_hat = self(x)  # Forward pass
        
        # Check for NaN/Inf in output
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            logger.warning(f"NaN or Inf detected in model output! Shape: {y_hat.shape}")
            logger.warning(f"y_hat min: {y_hat.min()}, max: {y_hat.max()}")
            return None
        
        # Ensure shapes match
        if y_hat.shape != y.shape:
            logger.error(f"Shape mismatch! y_hat: {y_hat.shape}, y: {y.shape}")
            # Try to reshape
            if y_hat.numel() == y.numel():
                y_hat = y_hat.view(y.shape)
            else:
                logger.error("Cannot reshape! Returning dummy loss.")
                return torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        
        # Calculate MSE loss (appropriate for regression)
        # y_hat: (batch, h, w), y: (batch, h, w)
        loss = F.mse_loss(y_hat, y)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss detected! y_hat range: [{y_hat.min():.4f}, {y_hat.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
            return None
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_mae(y_hat, y)
        self.log("train_mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with loss calculation and metric logging.
        
        Args:
            batch: Tuple of (input_tensor, target_labels)
            batch_idx: Index of the batch
        
        Returns:
            Validation loss
        """
        x, y = batch
        
        # Check for NaN/Inf in input and replace with safe values
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or Inf detected in validation input! Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.warning("NaN or Inf detected in validation labels! Replacing with zeros.")
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        
        y_hat = self(x)  # Forward pass
        
        # Check for NaN/Inf in output and replace
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            logger.warning(f"NaN or Inf detected in validation output! Shape: {y_hat.shape}")
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure shapes match
        if y_hat.shape != y.shape:
            logger.error(f"Shape mismatch in validation! y_hat: {y_hat.shape}, y: {y.shape}")
            if y_hat.numel() == y.numel():
                y_hat = y_hat.view(y.shape)
            else:
                logger.error("Cannot reshape validation! Returning dummy loss.")
                return torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        
        # Calculate MSE loss
        # y_hat: (batch, h, w), y: (batch, h, w)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf validation loss! y_hat: min={y_hat.min():.4f}, max={y_hat.max():.4f}")
            logger.error(f"y: min={y.min():.4f}, max={y.max():.4f}")
            loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mae(y_hat, y)
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer (and optionally learning rate scheduler)
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-5  # L2 regularization
        )
        
        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # Monitor validation loss for scheduling
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference.
        
        Args:
            batch: Tuple of (input_tensor, labels) or just input_tensor
            batch_idx: Batch index
        
        Returns:
            Model predictions
        """
        # Handle both tuple (x, y) and single tensor x
        if isinstance(batch, (list, tuple)):
            x = batch[0]  # Extract input tensor
        else:
            x = batch
        
        # Run forward pass
        predictions = self(x)
        return predictions