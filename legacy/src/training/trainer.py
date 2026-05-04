import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# Add project root to system path for module resolution
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.metrics import evaluate_mosei, log_metrics
from src.utils.logging import setup_logging
from config import DEVICE, MODELS_DIR, LOGS_DIR

# Get logger
logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for managing the training, validation, and testing process.

    Attributes:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data (optional).
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to train on (CPU/GPU).
        model_dir (Path): Directory to save model checkpoints.
        log_dir (Path): Directory to save logs.
        experiment_name (str): Name of the experiment.
        best_val_loss (float): Best validation loss encountered.
        patience_counter (int): Counter for early stopping.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        lr=1e-4,
        weight_decay=1e-5,
        device=None,
        model_dir=None,
        log_dir=None,
        experiment_name=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        self.device = device if device is not None else DEVICE
        self.model = self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Setup paths for saving models and logs
        self.model_dir = Path(model_dir) if model_dir is not None else Path(MODELS_DIR)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_dir = Path(log_dir) if log_dir is not None else Path(LOGS_DIR)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Set experiment name
        if experiment_name is None:
            self.experiment_name = f"{model.__class__.__name__}_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name
        
        # Setup logging
        self.log_file = self.log_dir / f"{self.experiment_name}.log"
        setup_logging(self.log_file)
        
        # Track best validation performance
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        """
        Train the model for a single epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get batch data
            if isinstance(batch, dict):
                # Multimodal data
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(self.device)
            else:
                # Unimodal data
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        pbar.close()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.train_loader)
        
        return avg_loss
    
    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        val_loss = 0.0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get batch data
                if isinstance(batch, dict):
                    # Multimodal data
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
                    labels = batch["label"].to(self.device)
                else:
                    # Unimodal data
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs.squeeze(), labels)
                
                # Update statistics
                val_loss += loss.item()
                
                # Collect predictions and labels
                preds = outputs.squeeze().cpu().numpy()
                labels = labels.cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels)
        
        # Calculate average loss
        avg_loss = val_loss / len(self.val_loader)
        
        # Concatenate batch results
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Evaluate using metrics
        metrics = evaluate_mosei(self.model, self.val_loader, self.device)
        
        return avg_loss, metrics
    
    def test(self):
        """
        Evaluate the best model on the test set.
        """
        if self.test_loader is None:
            logger.warning("Test loader not provided. Skipping test evaluation.")
            return None
        
        logger.info("Evaluating model on test set...")
        metrics = evaluate_mosei(self.model, self.test_loader, self.device)
        log_metrics(metrics, "test")
        
        return metrics
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss at this epoch.
            is_best (bool): Flag to indicate if this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.model_dir / f"{self.experiment_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.model_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path=None):
        """
        Load model checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file. If None, load best model.
        """
        if checkpoint_path is None:
            checkpoint_path = self.model_dir / f"{self.experiment_name}_best.pt"
        
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint {checkpoint_path} not found. Skipping.")
            return
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Set best validation loss
        if "val_loss" in checkpoint:
            self.best_val_loss = checkpoint["val_loss"]
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def train(self, start_epoch=1, num_epochs=50, patience=10, eval_every=1):
        """
        Train the model with early stopping and checkpointing.

        Args:
            num_epochs (int): Number of total epochs.
            patience (int): Patience for early stopping.
            eval_every (int): Evaluate every N epochs.

        Returns:
            best_model (nn.Module): Best model based on validation performance.
            train_losses (list): List of training losses per epoch.
            val_losses (list): List of validation losses per evaluation.
            train_metrics_list (list): List of training metrics per evaluation.
            val_metrics_list (list): List of validation metrics per evaluation.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        if self.test_loader is not None:
            logger.info(f"Test samples: {len(self.test_loader.dataset)}")

        best_val_metrics = None
        best_model = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

        train_losses = []
        val_losses = []
        train_metrics_list = []
        val_metrics_list = []

        for epoch in range(start_epoch, num_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")

            # Evaluate on validation set
            if epoch % eval_every == 0:
                val_loss, val_metrics = self.validate(epoch)
                val_losses.append(val_loss)
                val_metrics_list.append(val_metrics)
                logger.info(f"Epoch {epoch} - Validation loss: {val_loss:.4f}")
                log_metrics(val_metrics, "val", epoch)

                # Save train metrics too (optional)
                train_metrics = evaluate_mosei(self.model, self.train_loader, self.device)
                train_metrics_list.append(train_metrics)
                log_metrics(train_metrics, "train", epoch)

                # Check if best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    best_model = self.model
                    best_val_metrics = val_metrics
                    self.patience_counter = 0

                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    logger.info(f"New best model at epoch {epoch}")
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement for {self.patience_counter} epochs")

                    if self.patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Save regular checkpoint
            if epoch % 5 == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, val_loss if epoch % eval_every == 0 else float("inf"))

        logger.info(f"Training completed. Best model at epoch {self.best_epoch}")

        return best_model, train_losses, val_losses, train_metrics_list, val_metrics_list