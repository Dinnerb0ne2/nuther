# -*- coding: utf-8 -*-
"""
Loss functions for Nuther neural network training.
Implements various loss functions using pure NumPy.
"""

import numpy as np


class LossFunction:
    """Base class for loss functions."""
    
    def __call__(self, y_pred, y_true):
        """
        Compute loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss value
        """
        raise NotImplementedError


class CrossEntropyLoss(LossFunction):
    """
    Cross entropy loss for classification tasks.
    Suitable for language modeling and sequence generation.
    """
    
    def __init__(self, epsilon=1e-10):
        """
        Initialize cross entropy loss.
        
        Args:
            epsilon: Small value to avoid log(0)
        """
        self.epsilon = epsilon
    
    def __call__(self, y_pred, y_true):
        """
        Compute cross entropy loss.
        
        Args:
            y_pred: Predicted logits (batch_size, vocab_size)
            y_true: True indices (batch_size,)
            
        Returns:
            Loss value (scalar)
        """
        # Apply softmax to get probabilities
        probs = self._softmax(y_pred)
        
        # Get probabilities for true classes
        batch_size = y_true.shape[0]
        probs_true = probs[np.arange(batch_size), y_true]
        
        # Avoid log(0)
        probs_true = np.clip(probs_true, self.epsilon, 1.0)
        
        # Compute cross entropy
        loss = -np.mean(np.log(probs_true))
        
        return loss
    
    def _softmax(self, x):
        """
        Compute softmax with numerical stability.
        
        Args:
            x: Input logits
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, y_pred, y_true):
        """
        Compute gradient of cross entropy loss.
        
        Args:
            y_pred: Predicted logits (batch_size, vocab_size)
            y_true: True indices (batch_size,)
            
        Returns:
            Gradient with respect to logits
        """
        # Softmax gradient
        probs = self._softmax(y_pred)
        
        # Subtract 1 for true class
        batch_size = y_true.shape[0]
        probs[np.arange(batch_size), y_true] -= 1
        
        # Average gradient
        grad = probs / batch_size
        
        return grad


class MSELoss(LossFunction):
    """
    Mean squared error loss for regression tasks.
    """
    
    def __call__(self, y_pred, y_true):
        """
        Compute MSE loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss value
        """
        diff = y_pred - y_true
        loss = np.mean(diff ** 2)
        return loss
    
    def backward(self, y_pred, y_true):
        """
        Compute gradient of MSE loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Gradient with respect to predictions
        """
        batch_size = y_pred.shape[0]
        grad = 2 * (y_pred - y_true) / batch_size
        return grad


class SequenceCrossEntropyLoss(LossFunction):
    """
    Cross entropy loss for sequence generation.
    Handles padding and sequence masking.
    """
    
    def __init__(self, epsilon=1e-10, pad_token_id=0):
        """
        Initialize sequence cross entropy loss.
        
        Args:
            epsilon: Small value to avoid log(0)
            pad_token_id: Token ID for padding
        """
        self.epsilon = epsilon
        self.pad_token_id = pad_token_id
    
    def __call__(self, y_pred, y_true):
        """
        Compute sequence cross entropy loss.
        
        Args:
            y_pred: Predicted logits (batch_size, seq_len, vocab_size)
            y_true: True indices (batch_size, seq_len)
            
        Returns:
            Loss value (scalar)
        """
        # Flatten sequences
        batch_size, seq_len, vocab_size = y_pred.shape
        y_pred_flat = y_pred.reshape(-1, vocab_size)
        y_true_flat = y_true.reshape(-1)
        
        # Validate shapes
        if y_pred_flat.shape[0] != y_true_flat.shape[0]:
            raise ValueError(
                f"Shape mismatch after flattening: "
                f"y_pred_flat.shape[0]={y_pred_flat.shape[0]}, "
                f"y_true_flat.shape[0]={y_true_flat.shape[0]}"
            )
        
        # Create mask (ignore padding)
        mask = (y_true_flat != self.pad_token_id).astype(np.float32)
        
        # Apply softmax
        probs = self._softmax(y_pred_flat)
        
        # Get probabilities for true classes
        # y_true_flat contains vocabulary indices (0 to vocab_size-1)
        # Ensure indices are within valid range
        y_true_flat_clipped = np.clip(y_true_flat, 0, vocab_size - 1)
        probs_true = probs[np.arange(y_true_flat.shape[0]), y_true_flat_clipped]
        
        # Avoid log(0)
        probs_true = np.clip(probs_true, self.epsilon, 1.0)
        
        # Compute cross entropy
        ce = -np.log(probs_true)
        
        # Apply mask and average
        loss = np.sum(ce * mask) / (np.sum(mask) + self.epsilon)
        
        return loss
    
    def _softmax(self, x):
        """Compute softmax with numerical stability."""
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, y_pred, y_true):
        """
        Compute gradient for sequence cross entropy.
        
        Args:
            y_pred: Predicted logits (batch_size, seq_len, vocab_size)
            y_true: True indices (batch_size, seq_len)
            
        Returns:
            Gradient with respect to logits
        """
        batch_size, seq_len, vocab_size = y_pred.shape
        
        # Flatten
        y_pred_flat = y_pred.reshape(-1, vocab_size)
        y_true_flat = y_true.reshape(-1)
        
        # Compute softmax gradient
        probs = self._softmax(y_pred_flat)
        
        # Subtract 1 for true class
        probs[np.arange(y_true_flat.shape[0]), y_true_flat] -= 1
        
        # Create mask
        mask = (y_true_flat != self.pad_token_id).astype(np.float32).reshape(-1, 1)
        
        # Apply mask
        grad = probs * mask
        
        # Average
        grad = grad / (np.sum(mask) + self.epsilon)
        
        # Reshape back
        grad = grad.reshape(batch_size, seq_len, vocab_size)
        
        return grad