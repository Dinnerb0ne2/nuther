"""
LSTM Layer implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements a single LSTM layer that processes sequences.
"""

import numpy as np
from typing import Tuple, Optional, List

from src.config import config
from .lstm_cell import LSTMCell


class LSTMLayer:
    """
    LSTM Layer that processes sequences using LSTM cells.
    Handles the temporal dimension of sequences.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 cell_dim: Optional[int] = None, return_sequences: bool = True):
        """
        Initialize LSTM layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            cell_dim: Cell state dimension (defaults to hidden_dim if None)
            return_sequences: Whether to return all hidden states or only the last one
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim if cell_dim is not None else hidden_dim
        self.return_sequences = return_sequences
        
        # Create LSTM cell
        self.lstm_cell = LSTMCell(input_dim, hidden_dim, cell_dim)
        
        # Storage for caching during forward pass
        self.caches: List[dict] = []
    
    def forward(self, x: np.ndarray, h_init: Optional[np.ndarray] = None, 
                c_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for sequence processing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            h_init: Initial hidden state of shape (batch_size, hidden_dim)
            c_init: Initial cell state of shape (batch_size, cell_dim)
            
        Returns:
            Tuple of (outputs, h_final, c_final) where:
                outputs: Output tensor of shape (batch_size, seq_len, hidden_dim) if return_sequences
                         or (batch_size, hidden_dim) if not return_sequences
                h_final: Final hidden state of shape (batch_size, hidden_dim)
                c_final: Final cell state of shape (batch_size, cell_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden and cell states
        if h_init is None:
            h = self.lstm_cell.get_hidden_state(batch_size)
        else:
            h = h_init.copy()
        
        if c_init is None:
            c = self.lstm_cell.get_cell_state(batch_size)
        else:
            c = c_init.copy()
        
        # Clear caches
        self.caches = []
        
        # Process sequence
        if self.return_sequences:
            outputs = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
            
            for t in range(seq_len):
                x_t = x[:, t, :]
                h, c, cache = self.lstm_cell.forward(x_t, h, c)
                outputs[:, t, :] = h
                self.caches.append(cache)
        else:
            # Only keep the last output
            for t in range(seq_len):
                x_t = x[:, t, :]
                h, c, cache = self.lstm_cell.forward(x_t, h, c)
                self.caches.append(cache)
            
            outputs = h
        
        return outputs, h, c
    
    def forward_inference(self, x: np.ndarray, h_init: Optional[np.ndarray] = None,
                          c_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for inference (without caching).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            h_init: Initial hidden state of shape (batch_size, hidden_dim)
            c_init: Initial cell state of shape (batch_size, cell_dim)
            
        Returns:
            Tuple of (outputs, h_final, c_final)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden and cell states
        if h_init is None:
            h = self.lstm_cell.get_hidden_state(batch_size)
        else:
            h = h_init.copy()
        
        if c_init is None:
            c = self.lstm_cell.get_cell_state(batch_size)
        else:
            c = c_init.copy()
        
        # Process sequence
        if self.return_sequences:
            outputs = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
            
            for t in range(seq_len):
                x_t = x[:, t, :]
                h, c = self.lstm_cell.forward_step(x_t, h, c)
                outputs[:, t, :] = h
        else:
            # Only keep the last output
            for t in range(seq_len):
                x_t = x[:, t, :]
                h, c = self.lstm_cell.forward_step(x_t, h, c)
            
            outputs = h
        
        return outputs, h, c
    
    def backward(self, dh_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Backward pass for computing gradients.
        
        Args:
            dh_out: Gradient of output (shape depends on return_sequences)
            
        Returns:
            Tuple of (dx, dh_init, dc_init, grads) where:
                dx: Gradient of input
                dh_init: Gradient of initial hidden state
                dc_init: Gradient of initial cell state
                grads: Dictionary of gradients
        """
        batch_size, seq_len, _ = self.caches[0]['x'].shape
        
        # Initialize gradients
        dh_next = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        dc_next = np.zeros((batch_size, self.cell_dim), dtype=np.float32)
        
        # Initialize gradient storage
        all_dW = np.zeros_like(self.lstm_cell.W)
        all_db = np.zeros_like(self.lstm_cell.b)
        
        # Backward pass through time
        if self.return_sequences:
            dx = np.zeros((batch_size, seq_len, self.input_dim), dtype=np.float32)
            
            for t in reversed(range(seq_len)):
                dh_total = dh_out[:, t, :] + dh_next
                
                cache = self.caches[t]
                dx_t, dh_prev, dc_prev, grads = self.lstm_cell.backward(dh_total, dc_next, cache)
                
                dx[:, t, :] = dx_t
                dh_next = dh_prev
                dc_next = dc_prev
                
                # Accumulate gradients
                all_dW += grads['W']
                all_db += grads['b']
        else:
            dx = np.zeros((batch_size, seq_len, self.input_dim), dtype=np.float32)
            
            for t in reversed(range(seq_len)):
                cache = self.caches[t]
                dx_t, dh_prev, dc_prev, grads = self.lstm_cell.backward(dh_next, dc_next, cache)
                
                dx[:, t, :] = dx_t
                dh_next = dh_prev
                dc_next = dc_prev
                
                # Accumulate gradients
                all_dW += grads['W']
                all_db += grads['b']
        
        # Prepare gradient dictionary
        grads = {
            'W': all_dW,
            'b': all_db,
            'Wf': all_dW[:, :self.cell_dim],
            'Wi': all_dW[:, self.cell_dim:2*self.cell_dim],
            'Wo': all_dW[:, 2*self.cell_dim:3*self.cell_dim],
            'Wg': all_dW[:, 3*self.cell_dim:],
            'bf': all_db[:self.cell_dim],
            'bi': all_db[self.cell_dim:2*self.cell_dim],
            'bo': all_db[2*self.cell_dim:3*self.cell_dim],
            'bg': all_db[3*self.cell_dim:]
        }
        
        return dx, dh_next, dc_next, grads
    
    def get_parameters(self) -> dict:
        """
        Get current parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.lstm_cell.get_parameters()
    
    def set_parameters(self, params: dict):
        """
        Set parameters from dictionary.
        
        Args:
            params: Dictionary of parameters
        """
        self.lstm_cell.set_parameters(params)
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self.lstm_cell.reset_parameters()
        self.caches = []
    
    def get_hidden_state(self, batch_size: int) -> np.ndarray:
        """
        Get initial hidden state (zeros).
        
        Args:
            batch_size: Batch size
            
        Returns:
            Zero hidden state
        """
        return self.lstm_cell.get_hidden_state(batch_size)
    
    def get_cell_state(self, batch_size: int) -> np.ndarray:
        """
        Get initial cell state (zeros).
        
        Args:
            batch_size: Batch size
            
        Returns:
            Zero cell state
        """
        return self.lstm_cell.get_cell_state(batch_size)
    
    def get_output_dim(self) -> int:
        """
        Get output dimension.
        
        Returns:
            Output dimension (hidden dimension)
        """
        return self.hidden_dim
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.lstm_cell.get_parameter_count()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LSTMLayer(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, return_sequences={self.return_sequences})"