import os
import sys
import logging
from pathlib import Path
import subprocess

# Add project root to the Python path for importing config and modules correctly
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import dataset configuration variables
from config import RAW_DATA_DIR, DATASET_NAME, DATASET_URL

# Set up a basic logging configuration for info and error messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def install_mmsdk():
    """
    Installs the CMU-MultimodalSDK if not already installed.
    This is required to handle and download the MOSEI dataset.
    """
    try:
        # Try to import CMU-MultimodalSDK
        import mmsdk
        logger.info("CMU-MultimodalSDK is already installed.")
    except ImportError:
        logger.info("Installing CMU-MultimodalSDK...")

        try:
            # Install the SDK from GitHub using pip
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git"
            ])
            logger.info("CMU-MultimodalSDK installed successfully.")
        except subprocess.CalledProcessError as e:
            # Log an error and exit if installation fails
            logger.error(f"Failed to install CMU-MultimodalSDK: {e}")
            sys.exit(1)

def download_mosei(RAW_DATA_DIR, DATASET_NAME, DATASET_URL):
    """
    Downloads and aligns the CMU-MOSEI dataset using CMU-MultimodalSDK.

    Args:
        RAW_DATA_DIR (Path): Directory to store raw dataset.
        DATASET_NAME (str): Name of the dataset subdirectory.
        DATASET_URL (str): Base URL to download CSD files from.

    Returns:
        bool: True if download and alignment succeed, False otherwise.
    """
    try:
        from mmsdk import mmdatasdk as md
        
        # Define the dataset save path
        dataset_path = RAW_DATA_DIR / DATASET_NAME
        dataset_path.mkdir(exist_ok=True, parents=True)
        
        # Define paths for high-level feature sequences
        mosei_highlevel = {
            "acoustic": DATASET_URL + "acoustic/CMU_MOSEI_COVAREP.csd",
            "visual": DATASET_URL + "visual/CMU_MOSEI_VisualOpenFace2.csd",
            "language": DATASET_URL + "language/CMU_MOSEI_TimestampedWords.csd",
        }

        # Define the label sequence path
        mosei_labels = {
            "labels": DATASET_URL + "labels/CMU_MOSEI_Labels.csd"
        }

        # Download the high-level features
        logger.info("Downloading MOSEI high-level features...")
        mosei_dataset = md.mmdataset(mosei_highlevel, str(dataset_path))
        
        # Add sentiment labels to the dataset
        logger.info("Adding labels...")
        mosei_dataset.add_computational_sequences(
            {"Sentiment Labels": mosei_labels["labels"]},
            str(dataset_path)
        )
        
        # Align all sequences to the sentiment labels
        logger.info("Aligning to sentiment labels...")
        mosei_dataset.align("Sentiment Labels")
        
        logger.info(f"Dataset successfully downloaded and aligned to {dataset_path}")
        return True
        
    except Exception as e:
        # Log any error during the process and return False
        logger.error(f"Error downloading MOSEI dataset: {e}")
        return False
