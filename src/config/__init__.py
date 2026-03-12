"""
Configuration module for Nuther (Retro Memory LSTM) neural network framework.
This module contains all hyperparameters and configuration settings for the model.
"""

import numpy as np


# Model Architecture Configuration
class ModelConfig:
    """Configuration for model architecture parameters."""
    
    # Vocabulary and embedding settings
    VOCAB_SIZE = 10000  # Maximum vocabulary size
    EMBEDDING_DIM = 256  # Dimension of word embeddings
    
    # LSTM settings
    INPUT_DIM = EMBEDDING_DIM  # Input dimension equals embedding dimension
    HIDDEN_DIM = 512  # LSTM hidden state dimension
    CELL_DIM = 512  # LSTM cell state dimension
    NUM_LAYERS = 2  # Number of LSTM layers
    
    # Memory settings
    MEMORY_SIZE = 1000  # Maximum number of memories to store
    MEMORY_DIM = 256  # Dimension of memory vectors
    RETRIEVAL_TOP_K = 5  # Number of top memories to retrieve
    SIMILARITY_THRESHOLD = 0.5  # Threshold for memory similarity
    
    # MoE (Mixture of Experts) settings
    NUM_EXPERTS = 8  # Number of expert models
    EXPERT_HIDDEN_DIM = 256  # Hidden dimension for each expert
    TOP_K_EXPERTS = 2  # Number of top experts to activate
    
    # Sequence settings
    MAX_SEQ_LENGTH = 100  # Maximum sequence length for input
    PAD_TOKEN_ID = 0  # Token ID for padding
    START_TOKEN_ID = 1  # Token ID for sequence start
    END_TOKEN_ID = 2  # Token ID for sequence end
    UNK_TOKEN_ID = 3  # Token ID for unknown tokens
    
    # Training settings (for future implementation)
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    GRADIENT_CLIP = 5.0  # Gradient clipping threshold
    
    # Memory retrieval settings
    CHUNK_SIZE = 50  # Size of text chunks for memory storage
    CHUNK_OVERLAP = 10  # Overlap between chunks
    KEYWORD_WEIGHT = 0.3  # Weight for keyword matching in similarity
    
    # Crawler settings
    CRAWLER_TIMEOUT = 10  # Timeout for web requests in seconds
    CRAWLER_MAX_PAGES = 100  # Maximum pages to crawl per session
    CRAWLER_DELAY = 1  # Delay between requests in seconds
    
    # Data paths
    DATA_DIR = "data"
    KNOWLEDGE_BASE_DIR = "data/knowledge_base"
    VOCAB_PATH = "data/vocab.txt"
    MODEL_SAVE_PATH = "data/model"
    MEMORY_SAVE_PATH = "data/memory"
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"
    
    @classmethod
    def get_special_tokens(cls):
        """Get special token mapping."""
        return {
            cls.PAD_TOKEN: cls.PAD_TOKEN_ID,
            cls.START_TOKEN: cls.START_TOKEN_ID,
            cls.END_TOKEN: cls.END_TOKEN_ID,
            cls.UNK_TOKEN: cls.UNK_TOKEN_ID
        }
    
    @classmethod
    def get_special_token_ids(cls):
        """Get special token ID mapping."""
        return {
            cls.PAD_TOKEN_ID: cls.PAD_TOKEN,
            cls.START_TOKEN_ID: cls.START_TOKEN,
            cls.END_TOKEN_ID: cls.END_TOKEN,
            cls.UNK_TOKEN_ID: cls.UNK_TOKEN
        }


# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Export configuration
config = ModelConfig()