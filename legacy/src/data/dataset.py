import os
import sys
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add the project root directory to the Python path so we can import modules from it
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import configuration variables
from config import (
    PROCESSED_DATA_DIR, DATASET_NAME, BATCH_SIZE,
    TEXT_EMBEDDING_DIM, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MOSEIDataset(Dataset):
    """
    A PyTorch Dataset class for the CMU-MOSEI dataset that supports
    loading data with multiple modalities (language, acoustic, visual).

    Args:
        split (str): Dataset split to use ("train", "val", or "test").
        modalities (list): Modalities to include (default = all three).
    """
    def __init__(self, split="train", modalities=None):
        self.split = split
        self.modalities = modalities or ["language", "acoustic", "visual"]
        
        # Validate provided modalities
        for modality in self.modalities:
            if modality not in ["language", "acoustic", "visual"]:
                raise ValueError(f"Invalid modality: {modality}")
        
        # Construct path to dataset split
        self.data_path = PROCESSED_DATA_DIR / DATASET_NAME / f"{split}_data.pkl"
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Run preprocessing script first."
            )
        
        # Load the data for the split
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Load feature dimension metadata (if available)
        metadata_path = PROCESSED_DATA_DIR / DATASET_NAME / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            # Fallback to default dimensions
            self.metadata = {
                "text_dim": TEXT_EMBEDDING_DIM,
                "audio_dim": AUDIO_FEATURE_SIZE,
                "visual_dim": VISUAL_FEATURE_SIZE
            }
        
        # Check if labels are present
        if "labels" not in self.data:
            raise ValueError(f"Labels not found in data file: {self.data_path}")
        
        # Get total number of samples in the dataset
        self.num_samples = len(self.data["labels"])
        logger.info(f"Loaded {self.num_samples} samples for {split} split")
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Retrieve a sample (features + label) by index.

        Args:
            idx (int): Sample index.
        Returns:
            dict: Dictionary with modality tensors and the label tensor.
        """
        sample = {}
        
        # Load each modality if available; otherwise, create a zero tensor
        for modality in self.modalities:
            if modality in self.data:
                sample[modality] = torch.tensor(self.data[modality][idx], dtype=torch.float32)
            else:
                logger.warning(f"Modality {modality} not found in data")
                # Create empty tensor with proper dimensions
                dim = self.metadata.get(f"{modality}_dim", 0)
                sample[modality] = torch.zeros(dim, dtype=torch.float32)
        
        # Load label
        sample["label"] = torch.tensor(self.data["labels"][idx], dtype=torch.float32)
        
        return sample

class MOSEIUnimodalDataset(Dataset):
    """
    A PyTorch Dataset class for loading a single modality (unimodal) from MOSEI.

    Args:
        split (str): Dataset split to use ("train", "val", or "test").
        modality (str): One of "language", "acoustic", or "visual".
    """
    def __init__(self, split="train", modality="text"):
        self.split = split
        self.modality = modality
        
        # Validate modality
        if modality not in ["language", "acoustic", "visual"]:
            raise ValueError(f"Invalid modality: {modality}")
        
        # Load dataset for the split
        self.data_path = PROCESSED_DATA_DIR / DATASET_NAME / f"{split}_data.pkl"
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Run preprocessing script first."
            )
        
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Check that both the modality and labels exist
        if modality not in self.data:
            raise ValueError(f"Modality {modality} not found in data file: {self.data_path}")
        if "labels" not in self.data:
            raise ValueError(f"Labels not found in data file: {self.data_path}")

        self.num_samples = len(self.data["labels"])
        logger.info(f"Loaded {self.num_samples} samples for {split} split, modality: {modality}")
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Retrieve a unimodal feature and label pair.

        Args:
            idx (int): Sample index.
        Returns:
            tuple: (features, label) as tensors.
        """
        features = torch.tensor(self.data[self.modality][idx], dtype=torch.float32)
        label = torch.tensor(self.data["labels"][idx], dtype=torch.float32)
        
        return features, label

def get_dataloaders(modalities=None, batch_size=BATCH_SIZE, num_workers=2):
    """
    Create multimodal dataloaders for train, val, and test splits.

    Args:
        modalities (list): Modalities to include. Default is all three.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for loading data.

    Returns:
        dict: Dataloaders for each split.
    """
    dataloaders = {}
    
    # Iterate over the splits
    for split in ["train", "val", "test"]:
        dataset = MOSEIDataset(split=split, modalities=modalities)
        
        shuffle = (split == "train") # Only shuffle for training
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders

def get_unimodal_dataloaders(modality, batch_size=BATCH_SIZE, num_workers=2):
    """
    Create unimodal dataloaders for train, val, and test splits.

    Args:
        modality (str): One of "language", "acoustic", or "visual".
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for loading data.

    Returns:
        dict: Dataloaders for each split.
    """
    dataloaders = {}
    
    # Iterate over the splits
    for split in ["train", "val", "test"]:
        dataset = MOSEIUnimodalDataset(split=split, modality=modality)
        
        shuffle = (split == "train") # Only shuffle for training
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders
