# -*- coding: utf-8 -*-
"""
Checkpoint management for Nuther neural network training.
Handles model saving and loading.
"""

import os
import pickle
import numpy as np
import json
from typing import Dict, Any, Optional
from datetime import datetime


class Checkpoint:
    """
    Checkpoint manager for saving and loading training state.
    """
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, model, optimizer, metrics, step: int, 
             prefix: str = "checkpoint"):
        """
        Save training checkpoint.
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            metrics: Metrics instance
            step: Current training step
            prefix: Checkpoint file prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_step{step}_{timestamp}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint_data = {
            'step': step,
            'timestamp': timestamp,
            'model_params': model.get_parameters(),
            'model_config': {
                'vocab_size': model.vocab.get_vocab_size() if hasattr(model, 'vocab') else None,
                'embedding_dim': model.embedding_dim if hasattr(model, 'embedding_dim') else None,
                'hidden_dim': model.hidden_dim if hasattr(model, 'hidden_dim') else None,
                'num_layers': model.num_layers if hasattr(model, 'num_layers') else None,
            },
            'optimizer_state': self._get_optimizer_state(optimizer),
            'metrics': {
                'losses': metrics.losses,
                'accuracies': metrics.accuracies,
                'learning_rates': metrics.learning_rates,
                'steps': metrics.steps,
                'current_step': metrics.current_step,
            },
            'timestamp_iso': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Save metadata
        metadata_path = filepath.replace('.pkl', '_meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'step': step,
                'timestamp': timestamp,
                'filepath': filepath,
                'model_config': checkpoint_data['model_config'],
                'metrics_summary': {
                    'last_loss': metrics.losses[-1] if metrics.losses else None,
                    'last_accuracy': metrics.accuracies[-1] if metrics.accuracies else None,
                }
            }, f, indent=2)
        
        print(f"Checkpoint saved: {filepath}")
        
        # Keep only last N checkpoints
        self._cleanup_old_checkpoints(prefix, keep_last=3)
        
        return filepath
    
    def load(self, filepath: str, model, optimizer, metrics):
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model instance
            optimizer: Optimizer instance
            metrics: Metrics instance
            
        Returns:
            Loaded step number
        """
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Load model parameters
        model.set_parameters(checkpoint_data['model_params'])
        
        # Load optimizer state
        self._set_optimizer_state(optimizer, checkpoint_data['optimizer_state'])
        
        # Load metrics
        metrics.losses = checkpoint_data['metrics'].get('losses', [])
        metrics.accuracies = checkpoint_data['metrics'].get('accuracies', [])
        metrics.learning_rates = checkpoint_data['metrics'].get('learning_rates', [])
        metrics.steps = checkpoint_data['metrics'].get('steps', [])
        metrics.current_step = checkpoint_data['metrics'].get('current_step', 0)
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"  Step: {checkpoint_data['step']}")
        print(f"  Timestamp: {checkpoint_data['timestamp']}")
        if metrics.losses:
            print(f"  Last loss: {metrics.losses[-1]:.4f}")
        if metrics.accuracies:
            print(f"  Last accuracy: {metrics.accuracies[-1]:.4f}")
        
        return checkpoint_data['step']
    
    def load_latest(self, model, optimizer, metrics, prefix: str = "checkpoint"):
        """
        Load latest checkpoint.
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            metrics: Metrics instance
            prefix: Checkpoint file prefix
            
        Returns:
            Loaded step number, or 0 if no checkpoint found
        """
        checkpoints = self.list_checkpoints(prefix)
        
        if not checkpoints:
            print("No checkpoints found")
            return 0
        
        # Sort by step number (descending)
        checkpoints.sort(key=lambda x: x['step'], reverse=True)
        latest = checkpoints[0]
        
        return self.load(latest['filepath'], model, optimizer, metrics)
    
    def list_checkpoints(self, prefix: str = "checkpoint") -> list:
        """
        List all checkpoints.
        
        Args:
            prefix: Checkpoint file prefix
            
        Returns:
            List of checkpoint dictionaries
        """
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(prefix) and filename.endswith('.pkl'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                
                # Load metadata
                meta_path = filepath.replace('.pkl', '_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    checkpoints.append(metadata)
        
        return checkpoints
    
    def _get_optimizer_state(self, optimizer) -> Dict:
        """
        Get optimizer state for saving.
        
        Args:
            optimizer: Optimizer instance
            
        Returns:
            Optimizer state dictionary
        """
        state = {
            'learning_rate': optimizer.learning_rate,
            'type': optimizer.__class__.__name__
        }
        
        # Get optimizer-specific state
        if hasattr(optimizer, 't'):
            state['t'] = optimizer.t
        if hasattr(optimizer, 'm'):
            state['m'] = {k: v.copy() for k, v in optimizer.m.items()}
        if hasattr(optimizer, 'v'):
            state['v'] = {k: v.copy() for k, v in optimizer.v.items()}
        if hasattr(optimizer, 'velocities'):
            state['velocities'] = {k: v.copy() for k, v in optimizer.velocities.items()}
        if hasattr(optimizer, 'cache'):
            state['cache'] = {k: v.copy() for k, v in optimizer.cache.items()}
        
        return state
    
    def _set_optimizer_state(self, optimizer, state: Dict):
        """
        Set optimizer state from saved state.
        
        Args:
            optimizer: Optimizer instance
            state: Saved optimizer state
        """
        # Set basic attributes
        if 't' in state:
            optimizer.t = state['t']
        if 'm' in state:
            optimizer.m = {k: v.copy() for k, v in state['m'].items()}
        if 'v' in state:
            optimizer.v = {k: v.copy() for k, v in state['v'].items()}
        if 'velocities' in state:
            optimizer.velocities = {k: v.copy() for k, v in state['velocities'].items()}
        if 'cache' in state:
            optimizer.cache = {k: v.copy() for k, v in state['cache'].items()}
    
    def _cleanup_old_checkpoints(self, prefix: str, keep_last: int = 3):
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            prefix: Checkpoint file prefix
            keep_last: Number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints(prefix)
        
        if len(checkpoints) <= keep_last:
            return
        
        # Sort by step (ascending)
        checkpoints.sort(key=lambda x: x['step'])
        
        # Remove old checkpoints
        for checkpoint in checkpoints[:-keep_last]:
            filepath = checkpoint['filepath']
            meta_path = filepath.replace('.pkl', '_meta.json')
            
            try:
                os.remove(filepath)
                print(f"Removed old checkpoint: {filepath}")
            except FileNotFoundError:
                pass
            
            try:
                os.remove(meta_path)
            except FileNotFoundError:
                pass


def save_model(model, filepath: str):
    """
    Save model parameters only.
    
    Args:
        model: Model instance
        filepath: Path to save model
    """
    model_data = {
        'parameters': model.get_parameters(),
        'config': {
            'vocab_size': model.vocab.get_vocab_size() if hasattr(model, 'vocab') else None,
            'embedding_dim': model.embedding_dim if hasattr(model, 'embedding_dim') else None,
            'hidden_dim': model.hidden_dim if hasattr(model, 'hidden_dim') else None,
            'num_layers': model.num_layers if hasattr(model, 'num_layers') else None,
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved: {filepath}")


def load_model(model, filepath: str):
    """
    Load model parameters.
    
    Args:
        model: Model instance
        filepath: Path to load model from
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    model.set_parameters(model_data['parameters'])
    
    print(f"Model loaded: {filepath}")


def export_model_for_inference(model, filepath: str):
    """
    Export model for inference (without optimizer state).
    
    Args:
        model: Model instance
        filepath: Path to save model
    """
    export_data = {
        'parameters': model.get_parameters(),
        'config': {
            'vocab_size': model.vocab.get_vocab_size() if hasattr(model, 'vocab') else None,
            'embedding_dim': model.embedding_dim if hasattr(model, 'embedding_dim') else None,
            'hidden_dim': model.hidden_dim if hasattr(model, 'hidden_dim') else None,
            'num_layers': model.num_layers if hasattr(model, 'num_layers') else None,
            'use_memory': model.use_memory if hasattr(model, 'use_memory') else None,
        },
        'export_time': datetime.now().isoformat()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(export_data, f)
    
    print(f"Model exported for inference: {filepath}")