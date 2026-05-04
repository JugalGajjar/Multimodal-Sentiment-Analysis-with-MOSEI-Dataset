import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime

# Add project root to Python path for absolute imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import project-level configurations
from config import (
    LOGS_DIR,
    TEXT_EMBEDDING_DIM, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE,
    HIDDEN_DIM, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
    NUM_EPOCHS, EARLY_STOPPING_PATIENCE, DEVICE, SEED
)
# Import necessary modules
from src.data.dataset import get_unimodal_dataloaders
from src.models.text import TextSentimentModel, TransformerTextEncoder
from src.models.audio import AudioSentimentModel
from src.models.visual import VisualSentimentModel
from src.training.trainer import Trainer
from src.training.metrics import evaluate_mosei, get_predictions
from src.utils.logging import setup_logging
from src.utils.visualization import (
    plot_training_curves,
    plot_scatter_predictions,
    plot_confusion_matrix,
    visualize_results_summary,
    setup_plotting_directory
)

def parse_args():
    """
    Parses command-line arguments for unimodal training script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train unimodal sentiment analysis model")
    
    parser.add_argument(
        "--modality",
        type=str,
        choices=["language", "acoustic", "visual"],
        required=True,
        help="Modality to use for training"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="simple",
        choices=["simple", "transformer"],
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Patience for early stopping"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device to use for training (mps/cuda/cpu)"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for the experiment (default: modality_modeltype_timestamp)"
    )
    
    return parser.parse_args()

def get_model(args, input_dim):
    """
    Instantiate the model based on modality and model type.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        input_dim (int): Dimension of the input feature.

    Returns:
        nn.Module: Model instance.
    """
    if args.modality == "language":
        if args.model_type == "simple":
            return TextSentimentModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
        else:  # transformer
            return TransformerTextEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
    
    elif args.modality == "acoustic":
        return AudioSentimentModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
    
    elif args.modality == "visual":
        return VisualSentimentModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
    
    else:
        raise ValueError(f"Invalid modality: {args.modality}")

def main():
    """
    Main training and evaluation pipeline.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.modality}_{args.model_type}_{timestamp}"
    
    # Set up experiment directories
    exp_dir = LOGS_DIR / args.experiment_name
    exp_dir.mkdir(exist_ok=True, parents=True)
    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True)
    plot_dir = setup_plotting_directory(args.experiment_name)
    
    # Set up logger
    logger = setup_logging(
        name=f"{args.modality}_{args.model_type}",
        log_file=exp_dir / "training.log"
    )
    
    # Log arguments
    logger.info(f"Starting training with arguments: {args}")
    
    # Set device
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading {args.modality} data...")
    dataloaders = get_unimodal_dataloaders(
        modality=args.modality,
        batch_size=args.batch_size
    )
    
    # Get input dimension based on modality
    if args.modality == "language":
        input_dim = TEXT_EMBEDDING_DIM
    elif args.modality == "acoustic":
        input_dim = AUDIO_FEATURE_SIZE
    elif args.modality == "visual":
        input_dim = VISUAL_FEATURE_SIZE
    else:
        raise ValueError(f"Invalid modality: {args.modality}")
    
    # Initialize model
    logger.info(f"Initializing {args.model_type} model for {args.modality} modality...")
    model = get_model(args, input_dim)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=args.patience,
        logger=logger
    )
    
    # Train model
    logger.info("Starting training...")
    train_losses, val_losses, val_metrics = trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        num_epochs=args.epochs
    )
    
    # Save trained model
    model_path = model_dir / f"{args.modality}_{args.model_type}_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loader = dataloaders["test"]
    test_loss, test_predictions, test_targets = trainer.evaluate(test_loader)
    
    # Calculate test metrics
    test_metrics = evaluate_mosei(model, test_loader, device)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Visualize results
    logger.info("Generating visualizations...")
    
    # Plot training/validation curves
    plot_path = plot_dir / "training_curves.png"
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        metrics={k: [m[k] for m in val_metrics] for k in val_metrics[0]},
        save_path=plot_path
    )
    
    # Scatter plot of predictions vs ground truth
    plot_path = plot_dir / "predictions_scatter.png"
    plot_scatter_predictions(
        y_true=test_targets,
        y_pred=test_predictions,
        save_path=plot_path
    )
    
    # Confusion matrix (for binary prediction thresholding)
    plot_path = plot_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=test_targets,
        y_pred=test_predictions,
        save_path=plot_path
    )
    
    # Visualize results summary
    plot_path = plot_dir / "results_summary.png"
    visualize_results_summary(
        metrics_dict=test_metrics,
        model_name=f"{args.modality.capitalize()} {args.model_type.capitalize()} Model",
        save_path=plot_path
    )
    
    logger.info(f"Training and evaluation completed for {args.modality} modality")

# Entry point
if __name__ == "__main__":
    main()