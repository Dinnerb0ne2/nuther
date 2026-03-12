"""
LSTM Cell implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements a single LSTM cell with gating mechanisms using pure NumPy.
"""

import numpy as np
from typing import Tuple, Optional

from src.config import config


class LSTMCell:
    """
    LSTM Cell with forget gate, input gate, and output gate.
    Implements the core LSTM operations for a single time step.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, cell_dim: Optional[int] = None):
        """
        Initialize LSTM cell.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            cell_dim: Cell state dimension (defaults to hidden_dim if None)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim if cell_dim is not None else hidden_dim
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize LSTM parameters with Xavier/Glorot initialization.
        The parameters are organized as:
        - Wf, bf: Forget gate parameters
        - Wi, bi: Input gate parameters
        - Wo, bo: Output gate parameters
        - Wg, bg: Candidate cell state parameters
        """
        # Combined weight matrices for efficiency
        # Shape: (input_dim + hidden_dim, 4 * cell_dim)
        # Order: forget, input, output, candidate
        d = self.input_dim + self.hidden_dim
        c = self.cell_dim
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (d + c))
        
        # Combined weight matrix [Wf; Wi; Wo; Wg]
        self.W = np.random.randn(d, 4 * c) * scale
        
        # Combined bias vector [bf; bi; bo; bg]
        self.b = np.zeros(4 * c)
        
        # Split combined parameters for easier access
        self.Wf = self.W[:, :c]
        self.Wi = self.W[:, c:2*c]
        self.Wo = self.W[:, 2*c:3*c]
        self.Wg = self.W[:, 3*c:]
        
        self.bf = self.b[:c]
        self.bi = self.b[c:2*c]
        self.bo = self.b[2*c:3*c]
        self.bg = self.b[3*c:]
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, 
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        Forward pass for a single time step.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            h_prev: Previous hidden state of shape (batch_size, hidden_dim)
            c_prev: Previous cell state of shape (batch_size, cell_dim)
            
        Returns:
            Tuple of (h_next, c_next, cache) where:
                h_next: Next hidden state of shape (batch_size, hidden_dim)
                c_next: Next cell state of shape (batch_size, cell_dim)
                cache: Cached values for backward pass
        """
        batch_size = x.shape[0]
        
        # Concatenate input and previous hidden state
        # Shape: (batch_size, input_dim + hidden_dim)
        concat = np.concatenate([x, h_prev], axis=1)
        
        # Compute gate activations
        # Forget gate
        f = self._sigmoid(concat @ self.Wf + self.bf)
        
        # Input gate
        i = self._sigmoid(concat @ self.Wi + self.bi)
        
        # Output gate
        o = self._sigmoid(concat @ self.Wo + self.bo)
        
        # Candidate cell state
        g = np.tanh(concat @ self.Wg + self.bg)
        
        # Update cell state
        # c_next = f * c_prev + i * g
        c_next = f * c_prev + i * g
        
        # Compute next hidden state
        # h_next = o * tanh(c_next)
        h_next = o * np.tanh(c_next)
        
        # Cache for backward pass
        cache = {
            'x': x,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'concat': concat,
            'f': f,
            'i': i,
            'o': o,
            'g': g,
            'c_next': c_next,
            'tanh_c_next': np.tanh(c_next)
        }
        
        return h_next, c_next, cache
    
    def forward_step(self, x: np.ndarray, h_prev: np.ndarray, 
                     c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified forward pass without caching (for inference).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            h_prev: Previous hidden state of shape (batch_size, hidden_dim)
            c_prev: Previous cell state of shape (batch_size, cell_dim)
            
        Returns:
            Tuple of (h_next, c_next)
        """
        # Concatenate input and previous hidden state
        concat = np.concatenate([x, h_prev], axis=1)
        
        # Compute gate activations
        f = self._sigmoid(concat @ self.Wf + self.bf)
        i = self._sigmoid(concat @ self.Wi + self.bi)
        o = self._sigmoid(concat @ self.Wo + self.bo)
        g = np.tanh(concat @ self.Wg + self.bg)
        
        # Update cell state
        c_next = f * c_prev + i * g
        
        # Compute next hidden state
        h_next = o * np.tanh(c_next)
        
        return h_next, c_next
    
    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray, 
                 cache: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Backward pass for computing gradients.
        
        Args:
            dh_next: Gradient of next hidden state
            dc_next: Gradient of next cell state
            cache: Cached values from forward pass
            
        Returns:
            Tuple of (dh_prev, dc_prev, grads) where:
                dh_prev: Gradient of previous hidden state
                dc_prev: Gradient of previous cell state
                grads: Dictionary of gradients
        """
        # Unpack cache
        x = cache['x']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        concat = cache['concat']
        f = cache['f']
        i = cache['i']
        o = cache['o']
        g = cache['g']
        c_next = cache['c_next']
        tanh_c_next = cache['tanh_c_next']
        
        batch_size = x.shape[0]
        
        # Compute gradient through output gate
        do = dh_next * tanh_c_next
        do = self._sigmoid_derivative(o) * do
        
        # Compute gradient through cell state
        dc = dc_next + dh_next * o * (1 - tanh_c_next ** 2)
        
        # Compute gradient through forget gate
        df = dc * c_prev
        df = self._sigmoid_derivative(f) * df
        
        # Compute gradient through input gate
        di = dc * g
        di = self._sigmoid_derivative(i) * di
        
        # Compute gradient through candidate
        dg = dc * i
        dg = (1 - g ** 2) * dg
        
        # Concatenate gate gradients
        d_concat = np.concatenate([df, di, do, dg], axis=1)
        
        # Compute gradients for weights and biases
        dW = concat.T @ d_concat
        db = np.sum(d_concat, axis=0)
        
        # Split gradients
        dWf = dW[:, :self.cell_dim]
        dWi = dW[:, self.cell_dim:2*self.cell_dim]
        dWo = dW[:, 2*self.cell_dim:3*self.cell_dim]
        dWg = dW[:, 3*self.cell_dim:]
        
        dbf = db[:self.cell_dim]
        dbi = db[self.cell_dim:2*self.cell_dim]
        dbo = db[2*self.cell_dim:3*self.cell_dim]
        dbg = db[3*self.cell_dim:]
        
        # Compute gradient for previous hidden state
        dh_prev = d_concat @ self.W.T[:, self.input_dim:]
        
        # Compute gradient for previous cell state
        dc_prev = f * dc
        
        # Compute gradient for input
        dx = d_concat @ self.W.T[:, :self.input_dim]
        
        # Combine gradients
        grads = {
            'W': dW,
            'b': db,
            'Wf': dWf,
            'Wi': dWi,
            'Wo': dWo,
            'Wg': dWg,
            'bf': dbf,
            'bi': dbi,
            'bo': dbo,
            'bg': dbg
        }
        
        return dx, dh_prev, dc_prev, grads
    
    def get_parameters(self) -> dict:
        """
        Get current parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'W': self.W.copy(),
            'b': self.b.copy(),
            'Wf': self.Wf.copy(),
            'Wi': self.Wi.copy(),
            'Wo': self.Wo.copy(),
            'Wg': self.Wg.copy(),
            'bf': self.bf.copy(),
            'bi': self.bi.copy(),
            'bo': self.bo.copy(),
            'bg': self.bg.copy()
        }
    
    def set_parameters(self, params: dict):
        """
        Set parameters from dictionary.
        
        Args:
            params: Dictionary of parameters
        """
        self.W = params['W'].copy()
        self.b = params['b'].copy()
        self.Wf = params['Wf'].copy()
        self.Wi = params['Wi'].copy()
        self.Wo = params['Wo'].copy()
        self.Wg = params['Wg'].copy()
        self.bf = params['bf'].copy()
        self.bi = params['bi'].copy()
        self.bo = params['bo'].copy()
        self.bg = params['bg'].copy()
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self._init_parameters()
    
    def get_hidden_state(self, batch_size: int) -> np.ndarray:
        """
        Get initial hidden state (zeros).
        
        Args:
            batch_size: Batch size
            
        Returns:
            Zero hidden state of shape (batch_size, hidden_dim)
        """
        return np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
    
    def get_cell_state(self, batch_size: int) -> np.ndarray:
        """
        Get initial cell state (zeros).
        
        Args:
            batch_size: Batch size
            
        Returns:
            Zero cell state of shape (batch_size, cell_dim)
        """
        return np.zeros((batch_size, self.cell_dim), dtype=np.float32)
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Sigmoid of input
        """
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def _sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid activation.
        
        Args:
            x: Sigmoid output (not input)
            
        Returns:
            Derivative of sigmoid
        """
        return x * (1.0 - x)
    
    def get_output_dim(self) -> int:
        """
        Get output dimension (hidden dimension).
        
        Returns:
            Output dimension
        """
        return self.hidden_dim
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.W.size + self.b.size
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LSTMCell(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, cell_dim={self.cell_dim})"