"""
Vocabulary module for Nuther (Retro Memory LSTM) neural network framework.
This module handles text-to-index and index-to-text conversion, and builds vocabulary.
"""

import re
import os
from collections import Counter
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.config import config


class Vocabulary:
    """Vocabulary class for managing text-to-index and index-to-text conversions."""
    
    def __init__(self, vocab_size: int = config.VOCAB_SIZE):
        """
        Initialize vocabulary.
        
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [
            (config.PAD_TOKEN, config.PAD_TOKEN_ID),
            (config.START_TOKEN, config.START_TOKEN_ID),
            (config.END_TOKEN, config.END_TOKEN_ID),
            (config.UNK_TOKEN, config.UNK_TOKEN_ID)
        ]
        
        for token, idx in special_tokens:
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            self.word_counts[token] = 1  # Prevent special tokens from being pruned
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a word to be included
        """
        # Tokenize and count words
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
        
        # Sort words by frequency and add to vocabulary
        sorted_words = sorted(self.word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        # Reset vocabulary except special tokens
        self._add_special_tokens()
        
        # Add words up to vocab_size (excluding special tokens)
        num_special = len(config.get_special_tokens())
        added_words = 0
        
        for word, count in sorted_words:
            if word in self.word2idx:
                continue  # Skip special tokens
            
            if count >= min_freq and added_words < self.vocab_size - num_special:
                new_idx = len(self.word2idx)
                self.word2idx[word] = new_idx
                self.idx2word[new_idx] = word
                added_words += 1
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:()\-"\'`]', '', text)
        
        # Split into tokens (supports both English and Chinese)
        # For English: split by spaces and punctuation
        # For Chinese: split by characters (basic segmentation)
        tokens = []
        current_word = []
        
        for char in text:
            if char.isspace():
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
            elif '\u4e00' <= char <= '\u9fff':  # Chinese character
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                tokens.append(char)
            elif char in '.,!?;:()"\'-`':
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                tokens.append(char)
            else:
                current_word.append(char)
        
        if current_word:
            tokens.append(''.join(current_word))
        
        return [t for t in tokens if t]  # Remove empty tokens
    
    def text_to_indices(self, text: str, add_start: bool = False, 
                        add_end: bool = False, max_length: Optional[int] = None) -> List[int]:
        """
        Convert text to list of token indices.
        
        Args:
            text: Input text string
            add_start: Whether to add start token
            add_end: Whether to add end token
            max_length: Maximum sequence length (padding/truncating)
            
        Returns:
            List of token indices
        """
        tokens = self.tokenize(text)
        indices = []
        
        if add_start:
            indices.append(self.word2idx[config.START_TOKEN])
        
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx[config.UNK_TOKEN])
        
        if add_end:
            indices.append(self.word2idx[config.END_TOKEN])
        
        # Pad or truncate to max_length
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices = indices + [self.word2idx[config.PAD_TOKEN]] * (max_length - len(indices))
        
        return indices
    
    def indices_to_text(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Convert list of token indices to text.
        
        Args:
            indices: List of token indices
            skip_special: Whether to skip special tokens
            
        Returns:
            Text string
        """
        tokens = []
        special_ids = set(config.get_special_tokens().values())
        
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special and idx in special_ids:
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """
        Save vocabulary to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for word, idx in sorted(self.word2idx.items(), key=lambda x: x[1]):
                f.write(f"{word}\t{idx}\n")
    
    def load(self, filepath: str):
        """
        Load vocabulary from file.
        
        Args:
            filepath: Path to load vocabulary from
        """
        self.word2idx = {}
        self.idx2word = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, idx = parts
                    self.word2idx[word] = int(idx)
                    self.idx2word[int(idx)] = word
        
        # Update vocab_size
        self.vocab_size = len(self.word2idx)
    
    def get_word_embedding_init(self, embedding_dim: int) -> np.ndarray:
        """
        Get initialization matrix for word embeddings.
        
        Args:
            embedding_dim: Dimension of embeddings
            
        Returns:
            Embedding initialization matrix
        """
        vocab_size = self.get_vocab_size()
        # Initialize with small random values
        embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Zero out padding token embedding
        if config.PAD_TOKEN_ID < vocab_size:
            embedding_matrix[config.PAD_TOKEN_ID] = 0
        
        return embedding_matrix
    
    def get_sequence_length(self, text: str) -> int:
        """
        Get the length of tokenized text.
        
        Args:
            text: Input text string
            
        Returns:
            Number of tokens
        """
        return len(self.tokenize(text))
    
    def batch_text_to_indices(self, texts: List[str], max_length: Optional[int] = None,
                              add_start: bool = False, add_end: bool = False) -> np.ndarray:
        """
        Convert batch of texts to indices array.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            add_start: Whether to add start token
            add_end: Whether to add end token
            
        Returns:
            2D numpy array of shape (batch_size, max_length)
        """
        if max_length is None:
            max_length = max(self.get_sequence_length(text) for text in texts)
            if add_start:
                max_length += 1
            if add_end:
                max_length += 1
        
        batch_indices = []
        for text in texts:
            indices = self.text_to_indices(text, add_start, add_end, max_length)
            batch_indices.append(indices)
        
        return np.array(batch_indices, dtype=np.int32)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word2idx
    
    def __getitem__(self, word: str) -> int:
        """Get index of word."""
        return self.word2idx.get(word, self.word2idx[config.UNK_TOKEN])