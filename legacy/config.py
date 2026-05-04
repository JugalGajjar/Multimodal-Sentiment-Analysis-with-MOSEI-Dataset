"""
Configuration and path management script for the multimodal sentiment analysis project.
Defines directory structure, dataset parameters, model hyperparameters, and training settings.
"""

import os
from pathlib import Path
import torch

# Root directory of the project
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # Raw dataset files
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Processed dataset files

# Source code directories
SRC_DIR = ROOT_DIR / "src"
DATA_SRC_DIR = SRC_DIR / "data"  # Data processing scripts
MODELS_SRC_DIR = SRC_DIR / "models"  # Model architecture definitions
UTILS_SRC_DIR = SRC_DIR / "utils"  # Utility functions (e.g., metrics, visualization)

# Model weights and training logs
MODELS_DIR = ROOT_DIR / "models"  # Saved model checkpoints
LOGS_DIR = ROOT_DIR / "logs"  # Training and evaluation logs

# Automatically create required directories if they do not exist
for directory in [DATA_DIR, RAW_DATA_DIR, SRC_DIR, DATA_SRC_DIR, PROCESSED_DATA_DIR,
                  MODELS_SRC_DIR, UTILS_SRC_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Dataset identifier
DATASET_NAME = "CMU_MOSEI"

# URL for downloading the CMU-MOSEI dataset
DATASET_URL = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/"

# Maximum length for text sequences (BERT input)
TEXT_MAX_LENGTH = 128

# Feature dimensions for each modality
AUDIO_FEATURE_SIZE = 40      # Number of MFCC features per audio frame
VISUAL_FEATURE_SIZE = 35     # Number of facial landmark points or visual features

# Dimensionality of the BERT-based text embeddings
TEXT_EMBEDDING_DIM = 768

# Hidden layer size used in fusion layers and encoders
HIDDEN_DIM = 256

# Transformer-specific settings
NUM_ATTENTION_HEADS = 8
NUM_TRANSFORMER_LAYERS = 4

# Dropout rate for regularization
DROPOUT_RATE = 0.3

# Random seed for reproducibility
SEED = 42

# Batch size used for training
BATCH_SIZE = 32

# Learning rate for optimizer
LEARNING_RATE = 1e-4

# Weight decay (L2 regularization)
WEIGHT_DECAY = 1e-5

# Total number of training epochs
NUM_EPOCHS = 50

# Patience for early stopping
EARLY_STOPPING_PATIENCE = 10

# Gradient clipping threshold to avoid exploding gradients
GRADIENT_CLIP_VAL = 1.0

# Automatically choose the best available device: MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu" 