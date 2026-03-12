"""
LSTM implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements a stacked LSTM network with multiple layers.
"""

import numpy as np
from typing import Tuple, Optional, List

from src.config import config
from .lstm_layer import LSTMLayer


class LSTM:
    """
    Stacked LSTM network with multiple layers.
    Supports bidirectional processing and configurable layer sizes.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 cell_dim: Optional[int] = None, bidirectional: bool = False,
                 return_sequences: bool = False):
        """
        Initialize stacked LSTM network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension for all layers
            num_layers: Number of LSTM layers
            cell_dim: Cell state dimension (defaults to hidden_dim if None)
            bidirectional: Whether to use bidirectional LSTM
            return_sequences: Whether to return sequences from all layers
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_dim = cell_dim if cell_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        
        # Create LSTM layers
        self.layers: List[LSTMLayer] = []
        
        # Forward layers
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = hidden_dim * (2 if bidirectional else 1)
            
            layer = LSTMLayer(
                input_dim=layer_input_dim,
                hidden_dim=hidden_dim,
                cell_dim=cell_dim,
                return_sequences=return_sequences or (i < num_layers - 1)
            )
            self.layers.append(layer)
        
        # Bidirectional layers (if enabled)
        self.backward_layers: List[LSTMLayer] = []
        if bidirectional:
            for i in range(num_layers):
                if i == 0:
                    layer_input_dim = input_dim
                else:
                    layer_input_dim = hidden_dim * 2  # Forward + backward
                
                layer = LSTMLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    cell_dim=cell_dim,
                    return_sequences=(i < num_layers - 1)
                )
                self.backward_layers.append(layer)
    
    def forward(self, x: np.ndarray, 
                h_init: Optional[List[np.ndarray]] = None,
                c_init: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through stacked LSTM layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            h_init: List of initial hidden states for each layer
            c_init: List of initial cell states for each layer
            
        Returns:
            Tuple of (output, h_final_list, c_final_list) where:
                output: Output tensor
                h_final_list: List of final hidden states for each layer
                c_final_list: List of final cell states for each layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden and cell states if not provided
        if h_init is None:
            h_init = [layer.get_hidden_state(batch_size) for layer in self.layers]
        if c_init is None:
            c_init = [layer.get_cell_state(batch_size) for layer in self.layers]
        
        # Forward pass through layers
        h_final_list = []
        c_final_list = []
        
        # Process forward direction
        for i, layer in enumerate(self.layers):
            x, h, c = layer.forward(x, h_init[i], c_init[i])
            h_final_list.append(h)
            c_final_list.append(c)
        
        # Process backward direction (if bidirectional)
        if self.bidirectional:
            # Reverse input for backward pass
            x_reversed = np.flip(x, axis=1)
            
            h_backward_init = [layer.get_hidden_state(batch_size) for layer in self.backward_layers]
            c_backward_init = [layer.get_cell_state(batch_size) for layer in self.backward_layers]
            
            # Process through backward layers
            for i, layer in enumerate(self.backward_layers):
                x_reversed, h, c = layer.forward(x_reversed, h_backward_init[i], c_backward_init[i])
                h_final_list.append(h)
                c_final_list.append(c)
            
            # Reverse backward output and concatenate with forward output
            x_backward = np.flip(x_reversed, axis=1)
            
            # For the last layer, combine forward and backward
            output = np.concatenate([x, x_backward], axis=-1)
        else:
            output = x
        
        return output, h_final_list, c_final_list
    
    def forward_inference(self, x: np.ndarray,
                          h_init: Optional[List[np.ndarray]] = None,
                          c_init: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass for inference (without caching).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            h_init: List of initial hidden states for each layer
            c_init: List of initial cell states for each layer
            
        Returns:
            Tuple of (output, h_final_list, c_final_list)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden and cell states if not provided
        if h_init is None:
            h_init = [layer.get_hidden_state(batch_size) for layer in self.layers]
        if c_init is None:
            c_init = [layer.get_cell_state(batch_size) for layer in self.layers]
        
        # Forward pass through layers
        h_final_list = []
        c_final_list = []
        
        # Process forward direction
        for i, layer in enumerate(self.layers):
            x, h, c = layer.forward_inference(x, h_init[i], c_init[i])
            h_final_list.append(h)
            c_final_list.append(c)
        
        # Process backward direction (if bidirectional)
        if self.bidirectional:
            # Reverse input for backward pass
            x_reversed = np.flip(x, axis=1)
            
            h_backward_init = [layer.get_hidden_state(batch_size) for layer in self.backward_layers]
            c_backward_init = [layer.get_cell_state(batch_size) for layer in self.backward_layers]
            
            # Process through backward layers
            for i, layer in enumerate(self.backward_layers):
                x_reversed, h, c = layer.forward_inference(x_reversed, h_backward_init[i], c_backward_init[i])
                h_final_list.append(h)
                c_final_list.append(c)
            
            # Reverse backward output and concatenate with forward output
            x_backward = np.flip(x_reversed, axis=1)
            
            # For the last layer, combine forward and backward
            output = np.concatenate([x, x_backward], axis=-1)
        else:
            output = x
        
        return output, h_final_list, c_final_list
    
    def get_initial_states(self, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get initial hidden and cell states for all layers.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (h_init_list, c_init_list)
        """
        h_init = [layer.get_hidden_state(batch_size) for layer in self.layers]
        c_init = [layer.get_cell_state(batch_size) for layer in self.layers]
        
        if self.bidirectional:
            h_init.extend([layer.get_hidden_state(batch_size) for layer in self.backward_layers])
            c_init.extend([layer.get_cell_state(batch_size) for layer in self.backward_layers])
        
        return h_init, c_init
    
    def get_parameters(self) -> dict:
        """
        Get parameters for all layers.
        
        Returns:
            Dictionary containing parameters for all layers
        """
        params = {}
        
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_parameters()
            for key, value in layer_params.items():
                params[f'layer_{i}_forward_{key}'] = value
        
        if self.bidirectional:
            for i, layer in enumerate(self.backward_layers):
                layer_params = layer.get_parameters()
                for key, value in layer_params.items():
                    params[f'layer_{i}_backward_{key}'] = value
        
        return params
    
    def set_parameters(self, params: dict):
        """
        Set parameters for all layers.
        
        Args:
            params: Dictionary of parameters
        """
        for i, layer in enumerate(self.layers):
            layer_params = {}
            for key in layer.get_parameters().keys():
                layer_params[key] = params[f'layer_{i}_forward_{key}']
            layer.set_parameters(layer_params)
        
        if self.bidirectional:
            for i, layer in enumerate(self.backward_layers):
                layer_params = {}
                for key in layer.get_parameters().keys():
                    layer_params[key] = params[f'layer_{i}_backward_{key}']
                layer.set_parameters(layer_params)
    
    def reset_parameters(self):
        """Reset all parameters to initial values."""
        for layer in self.layers:
            layer.reset_parameters()
        
        if self.bidirectional:
            for layer in self.backward_layers:
                layer.reset_parameters()
    
    def get_output_dim(self) -> int:
        """
        Get output dimension.
        
        Returns:
            Output dimension
        """
        if self.bidirectional:
            return self.hidden_dim * 2
        else:
            return self.hidden_dim
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        total = 0
        for layer in self.layers:
            total += layer.get_parameter_count()
        
        if self.bidirectional:
            for layer in self.backward_layers:
                total += layer.get_parameter_count()
        
        return total
    
    def get_layer_output(self, layer_idx: int, x: np.ndarray) -> np.ndarray:
        """
        Get output from a specific layer.
        
        Args:
            layer_idx: Layer index (0 to num_layers-1)
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output from specified layer
        """
        batch_size, seq_len, _ = x.shape
        h_init = self.layers[layer_idx].get_hidden_state(batch_size)
        c_init = self.layers[layer_idx].get_cell_state(batch_size)
        
        output, _, _ = self.layers[layer_idx].forward(x, h_init, c_init)
        return output
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LSTM(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, bidirectional={self.bidirectional})"


class EmbeddingLSTM:
    """
    LSTM with embedding layer for processing text sequences.
    Combines word embeddings with LSTM processing.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, cell_dim: Optional[int] = None,
                 bidirectional: bool = False, dropout: float = 0.0):
        """
        Initialize embedding LSTM.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            cell_dim: Cell state dimension (defaults to hidden_dim if None)
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate (applied after embedding and between layers)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize embedding matrix
        scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * scale
        self.embedding_matrix[0] = 0  # Zero out padding token
        
        # Create LSTM
        self.lstm = LSTM(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cell_dim=cell_dim,
            bidirectional=bidirectional,
            return_sequences=True  # Always return sequences for seq2seq
        )
    
    def embed(self, indices: np.ndarray) -> np.ndarray:
        """
        Convert token indices to embeddings.
        
        Args:
            indices: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = indices.shape
        embeddings = self.embedding_matrix[indices]
        return embeddings
    
    def forward(self, indices: np.ndarray,
                h_init: Optional[List[np.ndarray]] = None,
                c_init: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through embedding and LSTM.
        
        Args:
            indices: Token indices of shape (batch_size, seq_len)
            h_init: List of initial hidden states
            c_init: List of initial cell states
            
        Returns:
            Tuple of (output, h_final_list, c_final_list)
        """
        # Convert indices to embeddings
        x = self.embed(indices)
        
        # Apply dropout during training
        if self.dropout > 0 and self.training:
            mask = (np.random.rand(*x.shape) > self.dropout).astype(np.float32)
            x = x * mask / (1.0 - self.dropout)
        
        # Process through LSTM
        output, h_final, c_final = self.lstm.forward(x, h_init, c_init)
        
        return output, h_final, c_final
    
    def forward_inference(self, indices: np.ndarray,
                          h_init: Optional[List[np.ndarray]] = None,
                          c_init: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass for inference.
        
        Args:
            indices: Token indices of shape (batch_size, seq_len)
            h_init: List of initial hidden states
            c_init: List of initial cell states
            
        Returns:
            Tuple of (output, h_final_list, c_final_list)
        """
        # Convert indices to embeddings
        x = self.embed(indices)
        
        # Process through LSTM
        output, h_final, c_final = self.lstm.forward_inference(x, h_init, c_init)
        
        return output, h_final, c_final
    
    def get_initial_states(self, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get initial hidden and cell states.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (h_init_list, c_init_list)
        """
        return self.lstm.get_initial_states(batch_size)
    
    def get_parameters(self) -> dict:
        """
        Get all parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = {
            'embedding_matrix': self.embedding_matrix.copy()
        }
        lstm_params = self.lstm.get_parameters()
        params.update(lstm_params)
        return params
    
    def set_parameters(self, params: dict):
        """
        Set parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.embedding_matrix = params['embedding_matrix'].copy()
        
        lstm_params = {}
        for key, value in params.items():
            if key != 'embedding_matrix':
                lstm_params[key] = value
        self.lstm.set_parameters(lstm_params)
    
    def reset_parameters(self):
        """Reset all parameters."""
        scale = np.sqrt(2.0 / (self.vocab_size + self.embedding_dim))
        self.embedding_matrix = np.random.randn(self.vocab_size, self.embedding_dim) * scale
        self.embedding_matrix[0] = 0
        self.lstm.reset_parameters()
    
    def get_output_dim(self) -> int:
        """
        Get output dimension.
        
        Returns:
            Output dimension
        """
        return self.lstm.get_output_dim()
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.embedding_matrix.size + self.lstm.get_parameter_count()
    
    @property
    def training(self) -> bool:
        """Check if in training mode."""
        return getattr(self, '_training', True)
    
    @training.setter
    def training(self, value: bool):
        """Set training mode."""
        self._training = value
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EmbeddingLSTM(vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}, hidden_dim={self.hidden_dim}, num_layers={self.num_layers})"