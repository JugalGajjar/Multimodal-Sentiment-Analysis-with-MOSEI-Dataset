import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to Python path to allow relative imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import project-level configurations
from config import (
    LOGS_DIR, MODELS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, 
    EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_VAL, SEED, DEVICE,
    TEXT_EMBEDDING_DIM, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE,
    HIDDEN_DIM, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS, DROPOUT_RATE
)

# Import necessary modules
from src.data.dataset import get_dataloaders
from src.models.fusion import TransformerFusionModel
from src.training.trainer import Trainer
from src.training.metrics import log_metrics, get_predictions
from src.utils.logging import setup_logging
from src.utils.visualization import plot_training_curves, plot_scatter_predictions

def parse_args():
    """
    Parses command-line arguments for multimodal training script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train multimodal fusion model for MOSEI sentiment analysis")
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training"
    )

    parser.add_argument(
        "--lr", 
        type=float, 
        default=LEARNING_RATE, 
        help="Learning rate"
    )

    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=WEIGHT_DECAY, 
        help="Weight decay for optimizer"
    )

    parser.add_argument(
        "--epochs", 
        type=int, 
        default=NUM_EPOCHS, 
        help="Number of training epochs"
    )

    parser.add_argument(
        "--patience", 
        type=int, 
        default=EARLY_STOPPING_PATIENCE, 
        help="Patience for early stopping"
    )

    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        default=HIDDEN_DIM, 
        help="Hidden dimension size"
    )

    parser.add_argument(
        "--num_heads", 
        type=int, 
        default=NUM_ATTENTION_HEADS, 
        help="Number of attention heads"
    )

    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=NUM_TRANSFORMER_LAYERS, 
        help="Number of transformer layers"
    )

    parser.add_argument(
        "--dropout", 
        type=float, 
        default=DROPOUT_RATE, 
        help="Dropout rate"
    )
    
    parser.add_argument(
        "--grad_clip", 
        type=float, 
        default=GRADIENT_CLIP_VAL, 
        help="Gradient clipping value"
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=SEED, 
        help="Random seed"
    )

    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to model checkpoint to resume training"
    )

    parser.add_argument(
        "--save_dir", 
        type=str, 
        default=MODELS_DIR, 
        help="Directory to save model checkpoints"
    )

    parser.add_argument(
        "--log_dir", 
        type=str, 
        default=LOGS_DIR, 
        help="Directory to save logs"
    )

    parser.add_argument(
        "--modalities",
        type=str,
        default="language,acoustic,visual", 
        help="Comma-separated list of modalities to use"
    )
    
    return parser.parse_args()

def main():
    """
    Main execution function for training and evaluating the model.
    """
    # Parse training arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Get modalities to use
    modalities = args.modalities.split(",")
    logger.info(f"Using modalities: {modalities}")
    
    # Setup device for training
    device = DEVICE
    logger.info(f"Using device: {device}")
    
    # Load dataset dataloaders
    logger.info("Loading data...")
    dataloaders = get_dataloaders(
        modalities=modalities, 
        batch_size=args.batch_size
    )
    
    # Define input dimension mapping for each modality
    input_dims = {
        "language": TEXT_EMBEDDING_DIM,
        "acoustic": AUDIO_FEATURE_SIZE,
        "visual": VISUAL_FEATURE_SIZE
    }
    
    # Initialize multimodal transformer model
    logger.info("Creating model...")
    model = TransformerFusionModel(
        text_dim=input_dims["language"],
        audio_dim=input_dims["acoustic"],
        visual_dim=input_dims["visual"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout
    )
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model architecture:\n{model}")
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint for resuming training (if provided)
    start_epoch = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    # Create directory for saving models
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        test_loader=dataloaders["test"],
        optimizer=optimizer,
        device=device,
        log_dir=args.log_dir,
        model_dir=save_dir,
        experiment_name="multimodal_fusion",
    )
    
    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    
    best_model, train_losses, val_losses, train_metrics, val_metrics = trainer.train(
        num_epochs=args.epochs,
        start_epoch=start_epoch
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training curves
    plot_path = Path(args.log_dir) / "multimodal_training_curves.png"
    plot_training_curves(
        train_losses, val_losses,
        train_metrics, val_metrics,
        save_path=plot_path
    )
    logger.info(f"Training curves plotted to {plot_path}")
    
    # Evaluate best model on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test()
    log_metrics(test_metrics, "test")
    
    # Scatter plot of predictions vs actual targets
    predictions, targets = get_predictions(model=best_model, dataloader=dataloaders["test"], device=device)
    scatter_path = Path(args.log_dir) / "multimodal_predictions.png"
    plot_scatter_predictions(
        predictions, targets,
        save_path=scatter_path,
        title="Multimodal Sentiment Test Predictions vs Actual"
    )
    logger.info(f"Prediction scatter plot saved to {scatter_path}")
    
    logger.info("Training and evaluation completed!")

# Entry point
if __name__ == "__main__":
    main()