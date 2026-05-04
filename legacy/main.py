import os
import sys
import logging
from pathlib import Path
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Import global configurations
from config import (
    RAW_DATA_DIR, DATASET_NAME, DATASET_URL, PROCESSED_DATA_DIR, LOGS_DIR, MODELS_DIR,
    TEXT_MAX_LENGTH, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE,
    TEXT_EMBEDDING_DIM, SEED, DEVICE, EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_VAL,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, DROPOUT_RATE,
    HIDDEN_DIM, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS
)

# Import necessary modules
from src.data.download import install_mmsdk, download_mosei
from src.data.preprocess import MOSEIPreprocessor
from src.utils.logging import setup_logging
from src.data.dataset import get_dataloaders
from src.models.fusion import TransformerFusionModel
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_curves, plot_scatter_predictions
from src.training.metrics import log_metrics, get_predictions

# Add project root to path to enable module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def print_menu():
    """
    Display the main menu options for the console application.

    Shows the available actions the user can perform:
    - Dataset download/preprocessing
    - Model training
    - Model evaluation
    - Results visualization
    - Application exit
    """
    print("\n===== Multimodal Sentiment Analysis Console =====")
    print("1. Download and preprocess dataset")
    print("2. Train a new model")
    print("3. Evaluate a trained model")
    print("4. Visualize training results")
    print("5. Exit")
    print("=================================================")

def download_and_preprocess():
    """
    Handle dataset download and preprocessing workflow.

    This function:
    1. Checks/creates necessary directories
    2. Installs required SDK (CMU-MultimodalSDK)
    3. Downloads the MOSEI dataset
    4. Preprocesses the raw data into a usable format
    5. Saves processed data to disk

    Logs progress and errors throughout the process.
    """
    logger.info("Starting dataset download and preprocessing...\n")

    # Check if the dataset directory exists
    if not RAW_DATA_DIR:
        logger.error("Raw data directory is not defined in config.py.")
        return
    else:
        logger.info(f"Raw data directory: {RAW_DATA_DIR}")

    # Create data directory if it doesn"t exist
    RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)
    logger.info("Created raw data directory.")

    # Install CMU-MultimodalSDK if not already installed
    install_mmsdk()
    logger.info("CMU-MultimodalSDK installed successfully.")

    # Download the dataset
    logger.info("Downloading MOSEI dataset...")

    # Check if the dataset URL and name are defined
    if not DATASET_URL:
        logger.error("Dataset URL is not defined in config.py.")
        return
    else:
        logger.info(f"Dataset URL: {DATASET_URL}")
    if not DATASET_NAME:
        logger.error("Dataset name is not defined in config.py.")
        return
    else:
        logger.info(f"Dataset name: {DATASET_NAME}")

    # Attempt dataset download
    success = download_mosei(RAW_DATA_DIR, DATASET_NAME, DATASET_URL)

    if success:
        logger.info("Dataset download completed successfully.")
    else:
        logger.error("Failed to download dataset.")
        return

    # Preprocess the dataset
    logger.info("Preprocessing the dataset...")

    # Check if the dataset directory exists
    if not PROCESSED_DATA_DIR:
        logger.error("Processed data directory is not defined in config.py.")
        return
    else:
        logger.info(f"Processed data directory: {PROCESSED_DATA_DIR}")

    # Create directories if they don"t exist
    PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
    logger.info("Created processed data directory.")

    # Check if processed data already exists
    processed_dir = PROCESSED_DATA_DIR / DATASET_NAME
    if processed_dir.exists():
        if all((processed_dir / f"{split}_data.pkl").exists() for split in ["train", "val", "test"]):
            logger.info(f"Processed data already exists at {processed_dir}. Skipping preprocessing.")
            return
    
    # Initialize the preprocessor
    preprocessor = MOSEIPreprocessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    logger.info("Initialized MOSEIPreprocessor.")
    
    # Process the dataset
    success = preprocessor.process_dataset()
    if success:
        # Save the processed data
        preprocessor.save_processed_data()
        logger.info("Preprocessing completed successfully.")
    else:
        logger.error("Failed to preprocess dataset.")
        return

def train_model():
    """
    Handle model training workflow with configurable parameters.

    This function:
    1. Collects training parameters via user input
    2. Sets up logging and directories
    3. Loads and prepares the dataset
    4. Initializes the model and optimizer
    5. Handles checkpoint loading if specified
    6. Runs the training process
    7. Evaluates on test set
    8. Visualizes results

    All parameters can be customized through interactive prompts.
    """
    # Collect training parameters via user input
    checkpoint = input("Enter the path to the model checkpoint (.pt) or leave blank to train from scratch: ").strip()
    modalities = input("Enter modalities (default: language,acoustic,visual): ").strip() or "language,acoustic,visual"
    batch_size = int(input("Enter batch size (default: 32): ").strip() or BATCH_SIZE)
    num_epochs = int(input("Enter number of epochs (default: 50): ").strip() or NUM_EPOCHS)
    hidden_dim = int(input("Enter hidden dimension (default: 256): ").strip() or HIDDEN_DIM)
    num_layers = int(input("Enter number of layers (default: 4): ").strip() or NUM_TRANSFORMER_LAYERS)
    num_heads = int(input("Enter number of heads (default: 8): ").strip() or NUM_ATTENTION_HEADS)
    dropout = float(input("Enter dropout rate (default: 0.3): ").strip() or DROPOUT_RATE)
    lr = float(input("Enter learning rate (default: 1e-4): ").strip() or LEARNING_RATE)
    weight_decay = float(input("Enter weight decay (default: 1e-5): ").strip() or WEIGHT_DECAY)
    early_stopping = int(input("Enter early stopping patience (default: 10): ").strip() or EARLY_STOPPING_PATIENCE)
    gradient_clip = float(input("Enter gradient clipping value (default: 1.0): ").strip() or GRADIENT_CLIP_VAL)
    seed = int(input("Enter random seed (default: 42): ").strip() or SEED)
    log_dir = input("Enter log directory (default: logs/): ").strip() or LOGS_DIR
    save_dir = input("Enter model save directory (default: models/): ").strip() or MODELS_DIR
    device = input("Enter device (cuda/mps/cpu/auto): ").strip()

    # Setup logging
    logger = setup_logging(log_dir)

    if not device:
        device = DEVICE
    else:
        device = device.lower()
    if device not in ["mps", "cuda", "cpu", "auto"]:
        logger.error("Invalid device. Please enter 'mps', 'cuda', or 'cpu'.")
        return
    if device == "auto":
        device = DEVICE

    # Get modalities to use
    modalities = modalities.split(",")

    # Log all training parameters
    logger.info("All hyperparameters set.")
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Hidden dimension: {hidden_dim}")
    logger.info(f"Number of transformer layers: {num_layers}")
    logger.info(f"Number of attention heads: {num_heads}")
    logger.info(f"Dropout rate: {dropout}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Early stopping patience: {early_stopping}")
    logger.info(f"Gradient clipping value: {gradient_clip}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Model save directory: {save_dir}")
    logger.info(f"Modalities: {modalities}")
    logger.info(f"Checkpoint: {checkpoint}")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    logger.info(f"Random seed set to {seed}.")

    # Load data
    logger.info("\nLoading data...")
    dataloaders = get_dataloaders(
        modalities=modalities, 
        batch_size=batch_size
    )

    # Create model
    logger.info("\nCreating model...")
    input_dims = {
        "language": TEXT_EMBEDDING_DIM,
        "acoustic": AUDIO_FEATURE_SIZE,
        "visual": VISUAL_FEATURE_SIZE
    }
    
    model = TransformerFusionModel(
        text_dim=input_dims["language"],
        audio_dim=input_dims["acoustic"],
        visual_dim=input_dims["visual"],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout
    )
    model = model.to(device)

    # Log model summary
    logger.info(f"Model architecture:\n{model}")

    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )

    # Load checkpoint if provided
    start_epoch = 1
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            check = torch.load(checkpoint_path)
            model.load_state_dict(check["model_state_dict"])
            optimizer.load_state_dict(check["optimizer_state_dict"])
            start_epoch = check["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        test_loader=dataloaders["test"],
        device=device,
        log_dir=log_dir,
        model_dir=save_dir,
        experiment_name="multimodal_fusion",
    )

    logger.info("\nStarting training...")

    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    
    best_model, train_losses, val_losses, train_metrics, val_metrics = trainer.train(
        num_epochs=num_epochs,
        start_epoch=start_epoch
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Plot training curves
    plot_path = Path(log_dir) / "multimodal_training_curves.png"
    metrics = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics
    }
    plot_training_curves(
        train_losses, val_losses,
        save_path=plot_path
    )
    logger.info(f"Training curves plotted to {plot_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test()
    log_metrics(test_metrics, "test")

    # Plot prediction scatter plot
    predictions, targets = get_predictions(model=best_model, dataloader=dataloaders["test"], device=device)
    scatter_path = Path(log_dir) / "multimodal_predictions.png"
    plot_scatter_predictions(
        predictions, targets,
        save_path=scatter_path,
        title="Multimodal Sentiment Test Predictions vs Actual"
    )
    logger.info(f"Prediction scatter plot saved to {scatter_path}")
    
    logger.info("Training and evaluation completed!")


def evaluate_model():
    """
    Handle model evaluation workflow.

    This function:
    1. Loads a trained model from checkpoint
    2. Prepares test data
    3. Runs evaluation on test set
    4. Computes and logs metrics
    5. Visualizes predictions

    All parameters are configurable through interactive prompts.
    """
    # Setup logging with user-specified directory
    log_dir = input("Enter log directory (default: logs/): ").strip() or LOGS_DIR
    logger = setup_logging(log_dir)
    logger.info("\nStarting evaluation...")

    # Get required checkpoint path
    checkpoint = input("Enter the path to the model checkpoint (.pt): ").strip()
    if not checkpoint:
        logger.error("Checkpoint path is required for evaluation.")
        return

    # Collect evaluation parameters
    modalities = input("Enter modalities (default: language,acoustic,visual): ").strip() or "language,acoustic,visual"
    batch_size = input("Enter batch size (default: 32): ").strip() or BATCH_SIZE
    
    # Validate and set device
    device = input("Enter device (cuda/mps/cpu/default: auto): ").strip()
    if not device:
        device = DEVICE
    else:
        device = device.lower()
    if device not in ["mps", "cuda", "cpu", "auto"]:
        logger.error("Invalid device. Please enter 'mps', 'cuda', or 'cpu'.")
        return
    if device == "auto":
        device = DEVICE

    # Get modalities to use
    modalities = modalities.split(",")

    # Log evaluation parameters
    logger.info("Hyperparameters set.")
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Modalities: {modalities}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Log directory: {log_dir}")

    # Load data
    dataloaders = get_dataloaders(
        modalities=modalities,
        batch_size=batch_size
    )
    test_loader = dataloaders["test"]
    logger.info("Data loaded.")

    # Define input dimensions
    input_dims = {
        "language": TEXT_EMBEDDING_DIM,
        "acoustic": AUDIO_FEATURE_SIZE,
        "visual": VISUAL_FEATURE_SIZE
    }

    # Initialize model
    model = TransformerFusionModel(
        text_dim=input_dims["language"],
        audio_dim=input_dims["acoustic"],
        visual_dim=input_dims["visual"],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_ATTENTION_HEADS,
        dropout_rate=DROPOUT_RATE
    )
    model = model.to(device)
    logger.info("Model initialized.")

    # Load checkpoint
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    check = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(check["model_state_dict"])
    model.eval()

    # Run evaluation
    logger.info("Running evaluation on test set...")
    predictions, targets = get_predictions(model=model, dataloader=test_loader, device=device)

    # Compute and log metrics
    from src.training.metrics import evaluate_mosei
    test_metrics = evaluate_mosei(model, test_loader, device)
    log_metrics(test_metrics, split="test")

    # Plot scatter predictions
    scatter_path = Path(log_dir) / "test_predictions.png"
    plot_scatter_predictions(predictions, targets, save_path=scatter_path, title="Test Predictions vs Actual")
    logger.info(f"Saved prediction scatter plot to {scatter_path}")
    
    logger.info("Evaluation complete!")

def visualize_results():
    """
    Handle visualization of existing training results.
    
    This function:
    1. Locates saved visualization files in log directory
    2. Opens them using system default viewers
    
    Supports both training curves and prediction scatter plots.
    """
    log_dir = input("Enter log directory to visualize (default: logs/): ").strip() or LOGS_DIR
    plot_path = Path(log_dir) / "multimodal_training_curves.png"
    scatter_path = Path(log_dir) / "test_predictions.png"

    if plot_path.exists():
        logger.info(f"Opening training curves: {plot_path}")
        os.system(f"open '{plot_path}'" if sys.platform == "darwin" else f"xdg-open '{plot_path}'")
    else:
        logger.error("Training curves plot not found.")

    if scatter_path.exists():
        logger.info(f"Opening prediction scatter plot: {scatter_path}")
        os.system(f"open '{scatter_path}'" if sys.platform == "darwin" else f"xdg-open '{scatter_path}'")
    else:
        logger.error("Prediction scatter plot not found.")

def main():
    """
    Main entry point for the console application.
    
    Implements an interactive menu system that allows users to:
    - Download and preprocess data
    - Train models
    - Evaluate models
    - Visualize results
    - Exit the application
    
    The menu runs in a loop until the user chooses to exit.
    """
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            download_and_preprocess()
        elif choice == "2":
            train_model()
        elif choice == "3":
            evaluate_model()
        elif choice == "4":
            visualize_results()
        elif choice == "5":
            logger.info("Exiting the console. Goodbye!")
            break
        else:
            logger.error("Invalid choice. Please select 1-5.")
            continue

# Entry point
if __name__ == "__main__":
    main()