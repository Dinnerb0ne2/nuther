# -*- coding: utf-8 -*-
"""
Trainer module for Nuther neural network training.
Implements training loop, gradient computation, and parameter updates.
"""

import numpy as np
import time
from typing import Callable, Optional, Dict, List

from .loss import SequenceCrossEntropyLoss
from .optimizer import Optimizer, get_optimizer
from .metrics import Metrics, Accuracy, ProgressTracker
from .checkpoint import Checkpoint


class Trainer:
    """
    Trainer for neural network models.
    Handles training loop, gradient computation, and parameter updates.
    """
    
    def __init__(self, model, optimizer: Optimizer, 
                 loss_fn: Optional[Callable] = None,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for parameter updates
            loss_fn: Loss function (defaults to sequence cross-entropy)
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Default loss function
        if loss_fn is None:
            loss_fn = SequenceCrossEntropyLoss()
        self.loss_fn = loss_fn
        
        # Training state
        self.current_step = 0
        self.is_training = True
    
    def train_step(self, batch: Dict) -> Dict:
        """
        Perform a single training step.
        
        Args:
            batch: Batch data with 'context' and 'response' keys
            
        Returns:
            Dictionary with loss, predictions, and targets
        """
        self.model.train()
        
        # Get input and target
        context = batch['context']  # Shape: (batch_size, context_len)
        response = batch['response']  # Shape: (batch_size, response_len)
        
        # For teacher forcing: use response[:-1] as input, response[1:] as target
        decoder_input = response[:, :-1]  # Shift right
        decoder_target = response[:, 1:]   # Shift left
        
        # Forward pass
        output = self.model.forward(context, target_indices=decoder_input)
        
        # Compute loss
        logits = output['output_logits']  # Shape: (batch_size, seq_len, vocab_size)
        loss = self.loss_fn(logits, decoder_target)
        
        # Compute predictions for accuracy
        predictions = np.argmax(logits, axis=-1)
        
        # Compute accuracy (ignore padding)
        accuracy = Accuracy.compute(predictions, decoder_target, ignore_index=0)
        
        # Compute gradient
        grad_output = self.loss_fn.backward(logits, decoder_target)
        
        # Backward pass through model
        grads = self.model.backward(grad_output, context, decoder_input)
        
        # Update parameters
        self.optimizer.step(self.model.get_parameters(), grads)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions,
            'targets': decoder_target
        }
    
    def validate(self, val_batches: List[Dict]) -> Dict:
        """
        Validate on validation set.
        
        Args:
            val_batches: List of validation batches
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(val_batches)
        
        for batch in val_batches:
            context = batch['context']
            response = batch['response']
            
            # Forward pass (no gradient)
            decoder_input = response[:, :-1]
            decoder_target = response[:, 1:]
            
            output = self.model.forward(context, target_indices=decoder_input)
            logits = output['output_logits']
            
            # Compute loss
            loss = self.loss_fn(logits, decoder_target)
            total_loss += loss
            
            # Compute accuracy
            predictions = np.argmax(logits, axis=-1)
            accuracy = Accuracy.compute(predictions, decoder_target, ignore_index=0)
            total_accuracy += accuracy
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
    def train(self, train_batches: List[Dict], 
              num_epochs: int = 10,
              val_batches: Optional[List[Dict]] = None,
              checkpoint_manager: Optional[Checkpoint] = None,
              print_every: int = 10,
              save_every: int = 100,
              eval_every: int = 50):
        """
        Train the model.
        
        Args:
            train_batches: List of training batches
            num_epochs: Number of training epochs
            val_batches: Validation batches (optional)
            checkpoint_manager: Checkpoint manager (optional)
            print_every: Print frequency
            save_every: Save frequency
            eval_every: Validation frequency
        """
        print("="*60)
        print("  Starting Training")
        print("="*60)
        print(f"  Epochs: {num_epochs}")
        print(f"  Batches per epoch: {len(train_batches)}")
        print(f"  Total steps: {num_epochs * len(train_batches)}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        print(f"  Learning rate: {self.optimizer.learning_rate}")
        print("="*60)
        print()
        
        # Initialize metrics
        metrics = Metrics()
        progress = ProgressTracker(num_epochs * len(train_batches), print_every)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            for batch_idx, batch in enumerate(train_batches):
                # Training step
                step_result = self.train_step(batch)
                
                # Update metrics
                metrics.update(
                    loss=step_result['loss'],
                    accuracy=step_result['accuracy'],
                    learning_rate=self.optimizer.learning_rate
                )
                
                epoch_loss += step_result['loss']
                epoch_accuracy += step_result['accuracy']
                
                progress.update()
                
                # Print progress
                if progress.should_print():
                    progress.print_progress(metrics, prefix=f"  ")
                
                # Validation
                if val_batches and progress.current_step % eval_every == 0:
                    val_metrics = self.validate(val_batches)
                    print(f"  [Validation] Loss: {val_metrics['val_loss']:.4f}, "
                          f"Accuracy: {val_metrics['val_accuracy']:.4f}")
                
                # Save checkpoint
                if checkpoint_manager and progress.current_step % save_every == 0:
                    checkpoint_manager.save(
                        self.model, self.optimizer, metrics, progress.current_step
                    )
            
            # Print epoch summary
            avg_loss = epoch_loss / len(train_batches)
            avg_accuracy = epoch_accuracy / len(train_batches)
            print(f"\n  Epoch {epoch + 1} Summary:")
            print(f"    Loss: {avg_loss:.4f}")
            print(f"    Accuracy: {avg_accuracy:.4f}")
        
        # Save final checkpoint
        if checkpoint_manager:
            checkpoint_manager.save(
                self.model, self.optimizer, metrics, progress.current_step
            )
        
        print("\n" + "="*60)
        print("  Training Completed!")
        print("="*60)
        print(f"  Total steps: {progress.current_step}")
        print(f"  Final loss: {metrics.losses[-1]:.4f}")
        print(f"  Final accuracy: {metrics.accuracies[-1]:.4f}")
        print("="*60)
        
        return metrics


class SimpleTrainer:
    """
    Simplified trainer for quick training.
    Uses random data for demonstration.
    """
    
    def __init__(self, model, vocab_size: int, learning_rate: float = 0.001):
        """
        Initialize simple trainer.
        
        Args:
            model: Model to train
            vocab_size: Vocabulary size
            learning_rate: Learning rate
        """
        self.model = model
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        # Create optimizer
        from .optimizer import Adam
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Create loss function
        self.loss_fn = SequenceCrossEntropyLoss()
    
    def generate_random_batch(self, batch_size: int = 4, 
                             seq_len: int = 10) -> Dict:
        """
        Generate random batch for training.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Random batch
        """
        context = np.random.randint(1, self.vocab_size, (batch_size, seq_len))
        response = np.random.randint(1, self.vocab_size, (batch_size, seq_len))
        
        return {
            'context': context,
            'response': response
        }
    
    def train_simple(self, num_steps: int = 100, batch_size: int = 4):
        """
        Simple training loop with random data.
        
        Args:
            num_steps: Number of training steps
            batch_size: Batch size
        """
        print("="*60)
        print("  Simple Training (Random Data)")
        print("="*60)
        print(f"  Steps: {num_steps}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print("="*60)
        print()
        
        metrics = Metrics()
        
        for step in range(num_steps):
            # Generate random batch
            batch = self.generate_random_batch(batch_size)
            
            # Forward pass
            context = batch['context']
            response = batch['response']
            
            decoder_input = response[:, :-1]
            decoder_target = response[:, 1:]
            
            output = self.model.forward(context, target_indices=decoder_input)
            logits = output['output_logits']
            
            # Compute loss
            loss = self.loss_fn(logits, decoder_target)
            
            # Compute accuracy
            predictions = np.argmax(logits, axis=-1)
            accuracy = Accuracy.compute(predictions, decoder_target, ignore_index=0)
            
            # Compute gradient and update
            grad_output = self.loss_fn.backward(logits, decoder_target)
            grads = self.model.backward(grad_output, context, decoder_input)
            self.optimizer.step(self.model.get_parameters(), grads)
            
            # Update metrics
            metrics.update(loss, accuracy, self.learning_rate)
            
            # Print progress
            if (step + 1) % 10 == 0:
                stats = metrics.get_stats()
                print(f"  Step {step + 1}/{num_steps}: "
                      f"Loss = {stats['loss']:.4f}, "
                      f"Accuracy = {stats['accuracy']:.4f}")
        
        print("\n" + "="*60)
        print("  Simple Training Completed!")
        print("="*60)
        print(f"  Final loss: {metrics.losses[-1]:.4f}")
        print(f"  Final accuracy: {metrics.accuracies[-1]:.4f}")
        print("="*60)
        
        return metrics