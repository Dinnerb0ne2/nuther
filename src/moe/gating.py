"""
Gating network implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements gating mechanisms for selecting and routing to experts.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from abc import ABC, abstractmethod

from src.config import config


class GatingNetwork(ABC):
    """
    Abstract base class for gating networks.
    """
    
    def __init__(self, input_dim: int, num_experts: int):
        """
        Initialize gating network.
        
        Args:
            input_dim: Input dimension
            num_experts: Number of experts
        """
        self.input_dim = input_dim
        self.num_experts = num_experts
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through gating network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Gating weights of shape (batch_size, num_experts)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict:
        """
        Get gating parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict):
        """
        Set gating parameters.
        
        Args:
            params: Dictionary of parameters
        """
        pass
    
    @abstractmethod
    def reset_parameters(self):
        """Reset parameters to initial values."""
        pass


class TopKGating(GatingNetwork):
    """
    Top-K gating network that selects the top K experts for each input.
    Implements sparse routing to improve efficiency.
    """
    
    def __init__(self, input_dim: int, num_experts: int, 
                 top_k: Optional[int] = None, hidden_dim: Optional[int] = None):
        """
        Initialize Top-K gating network.
        
        Args:
            input_dim: Input dimension
            num_experts: Number of experts
            top_k: Number of top experts to select (uses config default if None)
            hidden_dim: Hidden layer dimension for gating network
        """
        super().__init__(input_dim, num_experts)
        
        self.top_k = top_k or config.TOP_K_EXPERTS
        self.hidden_dim = hidden_dim or config.EXPERT_HIDDEN_DIM
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize gating network parameters.
        """
        # First layer: input -> hidden
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros(self.hidden_dim)
        
        # Output layer: hidden -> num_experts
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.num_experts))
        self.W2 = np.random.randn(self.hidden_dim, self.num_experts) * scale2
        self.b2 = np.zeros(self.num_experts)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute gating weights.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Sparse gating weights of shape (batch_size, num_experts)
        """
        # First layer with ReLU activation
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)
        
        # Output layer (logits)
        logits = h @ self.W2 + self.b2
        
        # Apply Top-K sparse gating
        gating_weights = self._apply_top_k(logits)
        
        return gating_weights
    
    def _apply_top_k(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Top-K selection to create sparse gating weights.
        
        Args:
            logits: Logits of shape (batch_size, num_experts)
            
        Returns:
            Sparse gating weights of shape (batch_size, num_experts)
        """
        batch_size = logits.shape[0]
        
        # Find top-k values for each batch
        top_k_indices = np.argpartition(logits, -self.top_k, axis=1)[:, -self.top_k:]
        
        # Create sparse mask
        mask = np.zeros_like(logits, dtype=np.float32)
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        mask[batch_indices, top_k_indices] = 1.0
        
        # Softmax only on selected experts
        masked_logits = logits * mask - 1e10 * (1 - mask)
        gating_weights = self._softmax(masked_logits)
        
        # Ensure exactly top-k non-zero entries
        gating_weights = gating_weights * mask
        
        return gating_weights
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Softmax activation with numerical stability.
        
        Args:
            x: Input tensor
            axis: Axis to apply softmax
            
        Returns:
            Softmax output
        """
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_selected_experts(self, x: np.ndarray) -> List[List[int]]:
        """
        Get the indices of selected experts for each input.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            List of selected expert indices for each batch
        """
        gating_weights = self.forward(x)
        
        # Get top-k indices for each batch
        selected_experts = []
        for i in range(gating_weights.shape[0]):
            weights = gating_weights[i]
            # Get indices of non-zero weights (top-k)
            indices = np.where(weights > 0)[0].tolist()
            selected_experts.append(indices)
        
        return selected_experts
    
    def get_parameters(self) -> Dict:
        """
        Get gating parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def set_parameters(self, params: Dict):
        """
        Set gating parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self._init_parameters()
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TopKGating(input_dim={self.input_dim}, num_experts={self.num_experts}, top_k={self.top_k})"


class SoftGating(GatingNetwork):
    """
    Soft gating network that uses softmax over all experts.
    Provides dense routing where all experts contribute.
    """
    
    def __init__(self, input_dim: int, num_experts: int,
                 hidden_dim: Optional[int] = None, temperature: float = 1.0):
        """
        Initialize soft gating network.
        
        Args:
            input_dim: Input dimension
            num_experts: Number of experts
            hidden_dim: Hidden layer dimension for gating network
            temperature: Temperature for softmax (higher = softer distribution)
        """
        super().__init__(input_dim, num_experts)
        
        self.hidden_dim = hidden_dim or config.EXPERT_HIDDEN_DIM
        self.temperature = temperature
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize gating network parameters.
        """
        # First layer: input -> hidden
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros(self.hidden_dim)
        
        # Output layer: hidden -> num_experts
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.num_experts))
        self.W2 = np.random.randn(self.hidden_dim, self.num_experts) * scale2
        self.b2 = np.zeros(self.num_experts)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute gating weights.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Gating weights of shape (batch_size, num_experts)
        """
        # First layer with ReLU activation
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)
        
        # Output layer (logits)
        logits = h @ self.W2 + self.b2
        
        # Apply softmax with temperature
        gating_weights = self._softmax(logits / self.temperature)
        
        return gating_weights
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Softmax activation with numerical stability.
        
        Args:
            x: Input tensor
            axis: Axis to apply softmax
            
        Returns:
            Softmax output
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_parameters(self) -> Dict:
        """
        Get gating parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def set_parameters(self, params: Dict):
        """
        Set gating parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self._init_parameters()
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size
    
    def set_temperature(self, temperature: float):
        """
        Set softmax temperature.
        
        Args:
            temperature: Temperature value
        """
        self.temperature = temperature
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SoftGating(input_dim={self.input_dim}, num_experts={self.num_experts}, temperature={self.temperature})"


class GumbelSoftmaxGating(GatingNetwork):
    """
    Gumbel-Softmax gating network for differentiable discrete routing.
    Provides a smooth approximation to discrete selection during training.
    """
    
    def __init__(self, input_dim: int, num_experts: int,
                 hidden_dim: Optional[int] = None, temperature: float = 1.0):
        """
        Initialize Gumbel-Softmax gating network.
        
        Args:
            input_dim: Input dimension
            num_experts: Number of experts
            hidden_dim: Hidden layer dimension for gating network
            temperature: Temperature for Gumbel-Softmax (lower = more discrete)
        """
        super().__init__(input_dim, num_experts)
        
        self.hidden_dim = hidden_dim or config.EXPERT_HIDDEN_DIM
        self.temperature = temperature
        self.training = True
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize gating network parameters.
        """
        # First layer: input -> hidden
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros(self.hidden_dim)
        
        # Output layer: hidden -> num_experts
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.num_experts))
        self.W2 = np.random.randn(self.hidden_dim, self.num_experts) * scale2
        self.b2 = np.zeros(self.num_experts)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute gating weights.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Gating weights of shape (batch_size, num_experts)
        """
        # First layer with ReLU activation
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)
        
        # Output layer (logits)
        logits = h @ self.W2 + self.b2
        
        # Apply Gumbel-Softmax during training, argmax during inference
        if self.training:
            gating_weights = self._gumbel_softmax(logits, self.temperature)
        else:
            # Use argmax for discrete selection
            expert_indices = np.argmax(logits, axis=-1)
            gating_weights = np.zeros_like(logits)
            batch_indices = np.arange(logits.shape[0])
            gating_weights[batch_indices, expert_indices] = 1.0
        
        return gating_weights
    
    def _gumbel_softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Gumbel-Softmax sampling.
        
        Args:
            logits: Logits of shape (batch_size, num_experts)
            temperature: Temperature parameter
            
        Returns:
            Gumbel-Softmax samples
        """
        # Sample Gumbel noise
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, logits.shape)))
        
        # Add noise to logits
        noisy_logits = (logits + gumbel_noise) / temperature
        
        # Apply softmax
        return self._softmax(noisy_logits)
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Softmax activation.
        
        Args:
            x: Input tensor
            axis: Axis to apply softmax
            
        Returns:
            Softmax output
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_parameters(self) -> Dict:
        """
        Get gating parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def set_parameters(self, params: Dict):
        """
        Set gating parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self._init_parameters()
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size
    
    def set_temperature(self, temperature: float):
        """
        Set Gumbel-Softmax temperature.
        
        Args:
            temperature: Temperature value
        """
        self.temperature = temperature
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
    
    def __repr__(self) -> str:
        """String representation."""
        return f"GumbelSoftmaxGating(input_dim={self.input_dim}, num_experts={self.num_experts}, temperature={self.temperature})"