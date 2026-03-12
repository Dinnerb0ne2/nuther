# -*- coding: utf-8 -*-
"""
Training module for Nuther neural network framework.
Provides loss functions, optimizers, metrics, and training utilities.
"""

from .loss import (
    LossFunction,
    CrossEntropyLoss,
    MSELoss,
    SequenceCrossEntropyLoss
)

from .optimizer import (
    Optimizer,
    SGD,
    Adam,
    RMSprop,
    Adagrad,
    get_optimizer
)

from .metrics import (
    Metrics,
    Accuracy,
    Perplexity,
    ProgressTracker
)

from .checkpoint import (
    Checkpoint,
    save_model,
    load_model,
    export_model_for_inference
)

from .trainer import (
    Trainer,
    SimpleTrainer
)

__all__ = [
    # Loss functions
    'LossFunction',
    'CrossEntropyLoss',
    'MSELoss',
    'SequenceCrossEntropyLoss',
    
    # Optimizers
    'Optimizer',
    'SGD',
    'Adam',
    'RMSprop',
    'Adagrad',
    'get_optimizer',
    
    # Metrics
    'Metrics',
    'Accuracy',
    'Perplexity',
    'ProgressTracker',
    
    # Checkpoint
    'Checkpoint',
    'save_model',
    'load_model',
    'export_model_for_inference',
    
    # Trainers
    'Trainer',
    'SimpleTrainer'
]