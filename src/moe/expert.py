"""
Expert implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements individual expert models for the mixture of experts.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod

from src.config import config


class Expert(ABC):
    """
    Abstract base class for expert models.
    """
    
    def __init__(self, input_dim: int, output_dim: int, expert_id: int):
        """
        Initialize expert.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            expert_id: Unique identifier for this expert
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expert_id = expert_id
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through expert.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict:
        """
        Get expert parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict):
        """
        Set expert parameters.
        
        Args:
            params: Dictionary of parameters
        """
        pass
    
    @abstractmethod
    def reset_parameters(self):
        """Reset parameters to initial values."""
        pass
    
    def get_output_dim(self) -> int:
        """
        Get output dimension.
        
        Returns:
            Output dimension
        """
        return self.output_dim
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Expert(id={self.expert_id}, input_dim={self.input_dim}, output_dim={self.output_dim})"


class FeedForwardExpert(Expert):
    """
    Feed-forward neural network expert with two hidden layers.
    Simple but effective expert for MoE architecture.
    """
    
    def __init__(self, input_dim: int, output_dim: int, expert_id: int,
                 hidden_dim: Optional[int] = None, activation: str = 'relu'):
        """
        Initialize feed-forward expert.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            expert_id: Unique identifier for this expert
            hidden_dim: Hidden layer dimension (defaults to config value)
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super().__init__(input_dim, output_dim, expert_id)
        
        self.hidden_dim = hidden_dim or config.EXPERT_HIDDEN_DIM
        self.activation = activation.lower()
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize network parameters with Xavier initialization.
        """
        # First layer: input -> hidden
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros(self.hidden_dim)
        
        # Second layer: hidden -> hidden
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * scale2
        self.b2 = np.zeros(self.hidden_dim)
        
        # Output layer: hidden -> output
        scale3 = np.sqrt(2.0 / (self.hidden_dim + self.output_dim))
        self.W3 = np.random.randn(self.hidden_dim, self.output_dim) * scale3
        self.b3 = np.zeros(self.output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # First layer
        h1 = x @ self.W1 + self.b1
        h1 = self._activation(h1)
        
        # Second layer
        h2 = h1 @ self.W2 + self.b2
        h2 = self._activation(h2)
        
        # Output layer (no activation for final output)
        output = h2 @ self.W3 + self.b3
        
        return output
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            # Clip to avoid overflow
            x = np.clip(x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def get_parameters(self) -> Dict:
        """
        Get expert parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy(),
            'W3': self.W3.copy(),
            'b3': self.b3.copy()
        }
    
    def set_parameters(self, params: Dict):
        """
        Set expert parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
        self.W3 = params['W3'].copy()
        self.b3 = params['b3'].copy()
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self._init_parameters()
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return (self.W1.size + self.b1.size + 
                self.W2.size + self.b2.size + 
                self.W3.size + self.b3.size)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FeedForwardExpert(id={self.expert_id}, input_dim={self.input_dim}, output_dim={self.output_dim}, hidden_dim={self.hidden_dim})"


class LSTMExpert(Expert):
    """
    LSTM-based expert for processing sequential inputs.
    Useful for modeling temporal dependencies in expert outputs.
    """
    
    def __init__(self, input_dim: int, output_dim: int, expert_id: int,
                 hidden_dim: Optional[int] = None):
        """
        Initialize LSTM expert.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            expert_id: Unique identifier for this expert
            hidden_dim: LSTM hidden dimension
        """
        super().__init__(input_dim, output_dim, expert_id)
        
        self.hidden_dim = hidden_dim or config.EXPERT_HIDDEN_DIM
        
        # Initialize LSTM cell parameters
        self._init_lstm_parameters()
        
        # Initialize output projection
        scale = np.sqrt(2.0 / (self.hidden_dim + self.output_dim))
        self.W_out = np.random.randn(self.hidden_dim, self.output_dim) * scale
        self.b_out = np.zeros(self.output_dim)
    
    def _init_lstm_parameters(self):
        """
        Initialize LSTM parameters.
        """
        d = self.input_dim + self.hidden_dim
        
        # Combined weight matrix [Wf; Wi; Wo; Wg]
        scale = np.sqrt(2.0 / (d + self.hidden_dim))
        self.W = np.random.randn(d, 4 * self.hidden_dim) * scale
        self.b = np.zeros(4 * self.hidden_dim)
        
        # Split for convenience
        self.Wf = self.W[:, :self.hidden_dim]
        self.Wi = self.W[:, self.hidden_dim:2*self.hidden_dim]
        self.Wo = self.W[:, 2*self.hidden_dim:3*self.hidden_dim]
        self.Wg = self.W[:, 3*self.hidden_dim:]
        
        self.bf = self.b[:self.hidden_dim]
        self.bi = self.b[self.hidden_dim:2*self.hidden_dim]
        self.bo = self.b[2*self.hidden_dim:3*self.hidden_dim]
        self.bg = self.b[3*self.hidden_dim:]
    
    def forward(self, x: np.ndarray, h_prev: Optional[np.ndarray] = None,
                c_prev: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through LSTM expert.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            h_prev: Previous hidden state of shape (batch_size, hidden_dim)
            c_prev: Previous cell state of shape (batch_size, hidden_dim)
            
        Returns:
            Tuple of (output, h_final, c_final)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden and cell states
        if h_prev is None:
            h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        else:
            h = h_prev.copy()
        
        if c_prev is None:
            c = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        else:
            c = c_prev.copy()
        
        # Check if input is sequential or single step
        if x.ndim == 3:
            # Sequential input: process each time step
            seq_len = x.shape[1]
            for t in range(seq_len):
                x_t = x[:, t, :]
                h, c = self._lstm_step(x_t, h, c)
        else:
            # Single step input
            h, c = self._lstm_step(x, h, c)
        
        # Project to output dimension
        output = h @ self.W_out + self.b_out
        
        return output, h, c
    
    def _lstm_step(self, x: np.ndarray, h: np.ndarray, 
                   c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single LSTM step.
        
        Args:
            x: Input of shape (batch_size, input_dim)
            h: Hidden state of shape (batch_size, hidden_dim)
            c: Cell state of shape (batch_size, hidden_dim)
            
        Returns:
            Tuple of (h_next, c_next)
        """
        # Concatenate input and hidden state
        concat = np.concatenate([x, h], axis=1)
        
        # Compute gates
        f = self._sigmoid(concat @ self.Wf + self.bf)
        i = self._sigmoid(concat @ self.Wi + self.bi)
        o = self._sigmoid(concat @ self.Wo + self.bo)
        g = np.tanh(concat @ self.Wg + self.bg)
        
        # Update cell state
        c_next = f * c + i * g
        
        # Update hidden state
        h_next = o * np.tanh(c_next)
        
        return h_next, c_next
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def get_parameters(self) -> Dict:
        """
        Get expert parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'W': self.W.copy(),
            'b': self.b.copy(),
            'W_out': self.W_out.copy(),
            'b_out': self.b_out.copy()
        }
    
    def set_parameters(self, params: Dict):
        """
        Set expert parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.W = params['W'].copy()
        self.b = params['b'].copy()
        self.W_out = params['W_out'].copy()
        self.b_out = params['b_out'].copy()
        
        # Update split parameters
        self.Wf = self.W[:, :self.hidden_dim]
        self.Wi = self.W[:, self.hidden_dim:2*self.hidden_dim]
        self.Wo = self.W[:, 2*self.hidden_dim:3*self.hidden_dim]
        self.Wg = self.W[:, 3*self.hidden_dim:]
        
        self.bf = self.b[:self.hidden_dim]
        self.bi = self.b[self.hidden_dim:2*self.hidden_dim]
        self.bo = self.b[2*self.hidden_dim:3*self.hidden_dim]
        self.bg = self.b[3*self.hidden_dim:]
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self._init_lstm_parameters()
        scale = np.sqrt(2.0 / (self.hidden_dim + self.output_dim))
        self.W_out = np.random.randn(self.hidden_dim, self.output_dim) * scale
        self.b_out = np.zeros(self.output_dim)
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.W.size + self.b.size + self.W_out.size + self.b_out.size
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LSTMExpert(id={self.expert_id}, input_dim={self.input_dim}, output_dim={self.output_dim}, hidden_dim={self.hidden_dim})"