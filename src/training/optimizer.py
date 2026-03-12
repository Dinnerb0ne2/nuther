# -*- coding: utf-8 -*-
"""
Optimizers for Nuther neural network training.
Implements various optimization algorithms using pure NumPy.
"""

import numpy as np


class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, learning_rate=0.001):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
    
    def step(self, params, grads):
        """
        Update parameters based on gradients.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    Simple and effective optimization algorithm.
    """
    
    def __init__(self, learning_rate=0.001, momentum=0.0, weight_decay=0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient (0 to disable)
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}
    
    def step(self, params, grads):
        """
        Update parameters using SGD with momentum.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        for key in params:
            if key not in grads:
                continue
            
            # Get gradient
            grad = grads[key]
            
            # Add weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad += self.weight_decay * params[key]
            
            # Apply momentum
            if self.momentum > 0:
                if key not in self.velocities:
                    self.velocities[key] = np.zeros_like(params[key])
                self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * grad
                params[key] += self.velocities[key]
            else:
                # Simple SGD update
                params[key] -= self.learning_rate * grad
        
        return params


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    Combines momentum and adaptive learning rates.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0  # Time step
    
    def step(self, params, grads):
        """
        Update parameters using Adam.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        self.t += 1
        
        for key in params:
            if key not in grads:
                continue
            
            # Get gradient
            grad = grads[key]
            
            # Add weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * params[key]
            
            # Initialize moments if needed
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    Adapts learning rates based on a moving average of squared gradients.
    """
    
    def __init__(self, learning_rate=0.001, alpha=0.99, epsilon=1e-8, weight_decay=0.0):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            alpha: Decay rate for moving average
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = {}
    
    def step(self, params, grads):
        """
        Update parameters using RMSprop.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        for key in params:
            if key not in grads:
                continue
            
            # Get gradient
            grad = grads[key]
            
            # Add weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * params[key]
            
            # Initialize cache if needed
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # Update cache
            self.cache[key] = self.alpha * self.cache[key] + (1 - self.alpha) * (grad ** 2)
            
            # Update parameters
            params[key] -= self.learning_rate * grad / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return params


class Adagrad(Optimizer):
    """
    Adagrad optimizer.
    Adapts learning rates by dividing by the square root of sum of squares of gradients.
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8, weight_decay=0.0):
        """
        Initialize Adagrad optimizer.
        
        Args:
            learning_rate: Learning rate
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = {}
    
    def step(self, params, grads):
        """
        Update parameters using Adagrad.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        for key in params:
            if key not in grads:
                continue
            
            # Get gradient
            grad = grads[key]
            
            # Add weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * params[key]
            
            # Initialize cache if needed
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # Update cache (sum of squares)
            self.cache[key] += grad ** 2
            
            # Update parameters
            params[key] -= self.learning_rate * grad / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return params


def get_optimizer(name, **kwargs):
    """
    Get optimizer by name.
    
    Args:
        name: Optimizer name ('sgd', 'adam', 'rmsprop', 'adagrad')
        **kwargs: Optimizer parameters
        
    Returns:
        Optimizer instance
    """
    name = name.lower()
    
    if name == 'sgd':
        return SGD(**kwargs)
    elif name == 'adam':
        return Adam(**kwargs)
    elif name == 'rmsprop':
        return RMSprop(**kwargs)
    elif name == 'adagrad':
        return Adagrad(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")