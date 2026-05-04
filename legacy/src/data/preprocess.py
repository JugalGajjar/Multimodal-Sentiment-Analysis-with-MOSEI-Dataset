import sys
import logging
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

# Add project root to path for importing project-level modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import global configurations
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_NAME,
    TEXT_MAX_LENGTH, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE,
    TEXT_EMBEDDING_DIM, SEED, DEVICE
)

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MOSEIPreprocessor:
    """
    A class to preprocess the CMU-MOSEI dataset by:
    - Loading aligned multimodal data
    - Extracting BERT embeddings for text
    - Aggregating audio and visual features
    - Splitting into train/val/test
    - Saving processed data and metadata
    """
    def __init__(self, raw_data_dir, processed_data_dir):
        self.raw_data_dir = Path(raw_data_dir) / DATASET_NAME
        self.processed_data_dir = Path(processed_data_dir) / DATASET_NAME
        self.processed_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()
        
        # Configure device for computation (GPU if available)
        self.device = DEVICE
        if self.device == "cuda" or self.device == "mps":
            logger.info(f"Using GPU: {self.device}")
        else:
            logger.info("Using CPU")
        self.bert_model = self.bert_model.to(self.device)
        
        # Initialize data containers for splits
        self.data = {
            "train": {"language": [], "acoustic": [], "visual": [], "labels": []},
            "val": {"language": [], "acoustic": [], "visual": [], "labels": []},
            "test": {"language": [], "acoustic": [], "visual": [], "labels": []}
        }
        
        # Store metadata about dimensions and counts
        self.metadata = {
            "text_dim": TEXT_EMBEDDING_DIM,
            "audio_dim": AUDIO_FEATURE_SIZE,
            "visual_dim": VISUAL_FEATURE_SIZE,
            "num_classes": 1, # Regression (sentiment score)
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0
        }
    
    def load_aligned_data(self):
        """
        Loads aligned CSD files for language, acoustic, visual, and label modalities.
        """
        try:
            from mmsdk import mmdatasdk as md
            
            # Check if aligned data is present
            if not self.raw_data_dir.exists():
                logger.error(f"Aligned data not found at {self.raw_data_dir}")
                logger.error("Please run the download script first")
                return False
            
            # Load the four required modalities
            dataset = {}
            dataset["language"] = "data/raw/CMU_MOSEI/CMU_MOSEI_TimestampedWords.csd"
            dataset["acoustic"] = "data/raw/CMU_MOSEI/CMU_MOSEI_COVAREP.csd"
            dataset["visual"] = "data/raw/CMU_MOSEI/CMU_MOSEI_VisualOpenFace2.csd"
            dataset["labels"] = "data/raw/CMU_MOSEI/CMU_MOSEI_Labels.csd"

            if not dataset:
                logger.error("No computational sequence files found")
                return None
            
            # Load dataset into memory
            mosei_dataset = md.mmdataset(dataset)
            
            return mosei_dataset
            
        except Exception as e:
            logger.error(f"Error loading aligned data: {e}")
            return None
    
    def extract_text_features(self, text):
        """
        Converts raw text to BERT [CLS] embeddings.
        """
        try:
            # Tokenize and convert to tensor
            inputs = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=TEXT_MAX_LENGTH,
                return_tensors="pt"
            )
            
            # Move to appropriate device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use the [CLS] token representation as the text embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0] # Return 1D vector
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            # Return zeros if there's an error
            return np.zeros(TEXT_EMBEDDING_DIM)
    
    def extract_audio_features(self, audio_features):
        """
        Aggregates COVAREP acoustic features by averaging over time.
        """
        # MOSEI already provides COVAREP features, we'll use them directly
        if len(audio_features.shape) > 1 and audio_features.shape[0] > 0:
            # For simplicity, we"ll use averaging over time
            selected_features = np.mean(audio_features, axis=0)
            
            # Check for expected dimension
            if len(selected_features) > AUDIO_FEATURE_SIZE: # 40 MFCC features
                selected_features = selected_features[:AUDIO_FEATURE_SIZE]
            elif len(selected_features) < AUDIO_FEATURE_SIZE:
                # Pad with zeros
                selected_features = np.pad(
                    selected_features, 
                    (0, AUDIO_FEATURE_SIZE - len(selected_features))
                )
                
            return selected_features
        else:
            return np.zeros(AUDIO_FEATURE_SIZE)
    
    def extract_visual_features(self, visual_features):
        """
        Aggregates visual features (FACET) by averaging over time.
        """
        # MOSEI provides FACET features for visual modality
        if len(visual_features.shape) > 1 and visual_features.shape[0] > 0:
            # Average over time for simplicity
            selected_features = np.mean(visual_features, axis=0)
            
            # Check for expected dimension
            if len(selected_features) > VISUAL_FEATURE_SIZE: # 35 facial landmarks
                selected_features = selected_features[:VISUAL_FEATURE_SIZE]
            elif len(selected_features) < VISUAL_FEATURE_SIZE:
                # Pad with zeros
                selected_features = np.pad(
                    selected_features, 
                    (0, VISUAL_FEATURE_SIZE - len(selected_features))
                )
                
            return selected_features
        else:
            return np.zeros(VISUAL_FEATURE_SIZE)
    
    def process_dataset(self):
        """
        Processes the full dataset: loading, splitting, feature extraction.
        """
        # Load aligned data
        dataset = self.load_aligned_data()
        if dataset is None:
            return False
        
        # Extract segment IDs
        segment_ids = list(dataset["labels"].keys())
        logger.info(f"Found {len(segment_ids)} segments in the dataset")
        
        # Split the data into train/val/test sets
        train_ids, test_ids = train_test_split(
            segment_ids, test_size=0.2, random_state=SEED
        )
        train_ids, val_ids = train_test_split(
            train_ids, test_size=0.1, random_state=SEED
        )
        
        splits = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }
        
        # Process each split
        for split_name, split_ids in splits.items():
            logger.info(f"Processing {split_name} split with {len(split_ids)} segments")
            
            for segment_id in tqdm(split_ids, desc=f"Processing {split_name}"):
                try:
                    # Get features for each modality
                    text_features = dataset["language"][segment_id]["features"]
                    audio_features = dataset["acoustic"][segment_id]["features"]
                    visual_features = dataset["visual"][segment_id]["features"]
                    label = dataset["labels"][segment_id]["features"]
                    
                    # Convert text to string
                    text_str = " ".join([word[0].decode('utf-8') if isinstance(word[0], bytes) else str(word[0]) for word in text_features])
                    
                    # Extract BERT embeddings for text
                    text_embedding = self.extract_text_features(text_str)
                    
                    # Process audio and visual features
                    audio_embedding = self.extract_audio_features(audio_features)
                    visual_embedding = self.extract_visual_features(visual_features)
                    
                    # Get the sentiment score (label)
                    sentiment_score = np.mean(label)
                    
                    # Store the processed features
                    self.data[split_name]["language"].append(text_embedding)
                    self.data[split_name]["acoustic"].append(audio_embedding)
                    self.data[split_name]["visual"].append(visual_embedding)
                    self.data[split_name]["labels"].append(sentiment_score)
                    
                except Exception as e:
                    logger.warning(f"Error processing segment {segment_id}: {e}")
                    continue
            
            # Update metadata counts
            self.metadata[f"{split_name}_samples"] = len(self.data[split_name]["labels"])
        
        logger.info("Dataset processing completed")
        return True
    
    def save_processed_data(self):
        """
        Saves preprocessed data and metadata to disk using pickle.
        """
        for split_name in ["train", "val", "test"]:
            # Convert lists to numpy arrays
            for modality in ["language", "acoustic", "visual", "labels"]:
                self.data[split_name][modality] = np.array(self.data[split_name][modality])
            
            # Save split data
            split_file = self.processed_data_dir / f"{split_name}_data.pkl"
            with open(split_file, "wb") as f:
                pickle.dump(self.data[split_name], f)
            
            logger.info(f"Saved {split_name} data to {split_file}")
        
        # Save metadata
        metadata_file = self.processed_data_dir / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)
            
        logger.info(f"Saved metadata to {metadata_file}")
        logger.info("Data preprocessing completed and saved successfully")
