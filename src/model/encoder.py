"""
Encoder implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements the encoder that processes input text and retrieves memory.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List

from src.config import config
from src.lstm import EmbeddingLSTM
from src.memory import MemoryBank


class Encoder:
    """
    Encoder that processes input text through LSTM and retrieves relevant memory.
    Combines embedding LSTM with memory retrieval for context-aware encoding.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = config.NUM_LAYERS, cell_dim: Optional[int] = None,
                 bidirectional: bool = False, use_memory: bool = True,
                 memory_bank: Optional[MemoryBank] = None):
        """
        Initialize encoder.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            cell_dim: Cell state dimension
            bidirectional: Whether to use bidirectional LSTM
            use_memory: Whether to use memory retrieval
            memory_bank: Memory bank instance (created if None)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_dim = cell_dim if cell_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.use_memory = use_memory
        
        # Create embedding LSTM
        self.embedding_lstm = EmbeddingLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cell_dim=cell_dim,
            bidirectional=bidirectional
        )
        
        # Create or use provided memory bank
        if use_memory:
            self.memory_bank = memory_bank or MemoryBank()
        else:
            self.memory_bank = None
        
        # Calculate output dimension
        if bidirectional:
            self.output_dim = hidden_dim * 2
        else:
            self.output_dim = hidden_dim
    
    def forward(self, input_indices: np.ndarray,
                h_init: Optional[List[np.ndarray]] = None,
                c_init: Optional[List[np.ndarray]] = None,
                retrieve_memory: bool = True) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], Optional[str]]:
        """
        Forward pass through encoder.
        
        Args:
            input_indices: Input token indices of shape (batch_size, seq_len)
            h_init: Initial hidden states
            c_init: Initial cell states
            retrieve_memory: Whether to retrieve memory context
            
        Returns:
            Tuple of (output, h_final, c_final, memory_context) where:
                output: Encoder output of shape (batch_size, output_dim)
                h_final: Final hidden states
                c_final: Final cell states
                memory_context: Retrieved memory context (None if not using memory)
        """
        # Process through embedding LSTM
        output, h_final, c_final = self.embedding_lstm.forward_inference(
            input_indices, h_init, c_init
        )
        
        # Retrieve memory context if enabled
        memory_context = None
        if self.use_memory and retrieve_memory and self.memory_bank:
            # Get output from last time step
            if output.ndim == 3:
                last_output = output[:, -1, :]  # (batch_size, output_dim)
            else:
                last_output = output  # Already (batch_size, output_dim)
            
            # Use last output as query for memory retrieval
            # For now, we'll use a simple approach: convert to text query
            # In a full implementation, this would use the embedding directly
            batch_size = input_indices.shape[0]
            memory_contexts = []
            
            for i in range(batch_size):
                # Get input text (placeholder - in practice, convert indices to text)
                query_text = "query"  # This would be the actual input text
                
                # Retrieve memory
                retrieved = self.memory_bank.retrieve(query_text, top_k=3)
                if retrieved:
                    context_parts = [chunk.get_text() for chunk, _ in retrieved]
                    memory_context = ' '.join(context_parts)
                else:
                    memory_context = ''
                
                memory_contexts.append(memory_context)
            
            # For simplicity, return the first context
            memory_context = memory_contexts[0] if memory_contexts else None
        
        return output, h_final, c_final, memory_context
    
    def encode(self, input_indices: np.ndarray) -> np.ndarray:
        """
        Encode input to representation.
        
        Args:
            input_indices: Input token indices of shape (batch_size, seq_len)
            
        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        output, _, _, _ = self.forward(input_indices, retrieve_memory=False)
        
        # Get last time step output
        if output.ndim == 3:
            return output[:, -1, :]
        else:
            return output
    
    def get_initial_states(self, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get initial hidden and cell states.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (h_init, c_init)
        """
        return self.embedding_lstm.get_initial_states(batch_size)
    
    def store_in_memory(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Store text in memory bank.
        
        Args:
            text: Text to store
            metadata: Additional metadata
            
        Returns:
            List of chunk IDs
        """
        if self.memory_bank:
            return self.memory_bank.store(text, metadata)
        return []
    
    def get_memory_bank(self) -> Optional[MemoryBank]:
        """
        Get memory bank instance.
        
        Returns:
            Memory bank instance or None
        """
        return self.memory_bank
    
    def set_memory_bank(self, memory_bank: MemoryBank):
        """
        Set memory bank.
        
        Args:
            memory_bank: Memory bank instance
        """
        self.memory_bank = memory_bank
    
    def get_parameters(self) -> Dict:
        """
        Get encoder parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.embedding_lstm.get_parameters()
    
    def set_parameters(self, params: Dict):
        """
        Set encoder parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.embedding_lstm.set_parameters(params)
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self.embedding_lstm.reset_parameters()
    
    def get_output_dim(self) -> int:
        """
        Get output dimension.
        
        Returns:
            Output dimension
        """
        return self.output_dim
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.embedding_lstm.get_parameter_count()
    
    def train(self):
        """Set to training mode."""
        self.embedding_lstm.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.embedding_lstm.eval()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Encoder(vocab_size={self.vocab_size}, hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, bidirectional={self.bidirectional})"