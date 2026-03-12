# -*- coding: utf-8 -*-
"""
Training metrics and monitoring for Nuther neural network training.
Tracks loss, accuracy, and other training statistics.
"""

import numpy as np
import time
from typing import List, Dict, Optional


class Metrics:
    """
    Training metrics tracker.
    Monitors loss, accuracy, and other statistics during training.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.accuracies = []
        self.learning_rates = []
        self.times = []
        self.steps = []
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, loss: float, accuracy: Optional[float] = None, 
               learning_rate: Optional[float] = None):
        """
        Update metrics with new values.
        
        Args:
            loss: Loss value
            accuracy: Accuracy value (optional)
            learning_rate: Current learning rate (optional)
        """
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        self.times.append(time.time() - self.start_time)
        self.steps.append(self.current_step)
        self.current_step += 1
    
    def get_average_loss(self, window: int = 10) -> float:
        """
        Get average loss over a window.
        
        Args:
            window: Window size
            
        Returns:
            Average loss
        """
        if not self.losses:
            return 0.0
        window = min(window, len(self.losses))
        return np.mean(self.losses[-window:])
    
    def get_average_accuracy(self, window: int = 10) -> float:
        """
        Get average accuracy over a window.
        
        Args:
            window: Window size
            
        Returns:
            Average accuracy
        """
        if not self.accuracies:
            return 0.0
        window = min(window, len(self.accuracies))
        return np.mean(self.accuracies[-window:])
    
    def get_stats(self) -> Dict:
        """
        Get current statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'step': self.current_step,
            'loss': self.losses[-1] if self.losses else 0.0,
            'avg_loss': self.get_average_loss(),
            'accuracy': self.accuracies[-1] if self.accuracies else 0.0,
            'avg_accuracy': self.get_average_accuracy(),
            'time': self.times[-1] if self.times else 0.0,
            'total_time': time.time() - self.start_time,
            'steps_per_sec': self.current_step / (time.time() - self.start_time + 1e-8)
        }
        return stats
    
    def print_progress(self, prefix: str = ""):
        """
        Print training progress.
        
        Args:
            prefix: Prefix string
        """
        stats = self.get_stats()
        print(f"{prefix}Step {stats['step']}: "
              f"Loss = {stats['loss']:.4f} (avg: {stats['avg_loss']:.4f}), "
              f"Accuracy = {stats['accuracy']:.4f} (avg: {stats['avg_accuracy']:.4f}), "
              f"Time = {stats['time']:.2f}s")
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history (requires matplotlib).
        
        Args:
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            axes[0].plot(self.steps, self.losses, label='Loss')
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot accuracy
            if self.accuracies:
                axes[1].plot(self.steps, self.accuracies, label='Accuracy', color='orange')
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('Accuracy')
                axes[1].set_title('Training Accuracy')
                axes[1].legend()
                axes[1].grid(True)
            else:
                axes[1].text(0.5, 0.5, 'No accuracy data', 
                            ha='center', va='center', transform=axes[1].transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Warning: matplotlib not available, skipping plot")


class Accuracy:
    """
    Accuracy calculator for classification tasks.
    """
    
    @staticmethod
    def compute(predictions: np.ndarray, targets: np.ndarray, 
                ignore_index: Optional[int] = None) -> float:
        """
        Compute accuracy.
        
        Args:
            predictions: Predicted indices (batch_size,)
            targets: Target indices (batch_size,)
            ignore_index: Index to ignore (e.g., padding)
            
        Returns:
            Accuracy value
        """
        # Create mask
        if ignore_index is not None:
            mask = (targets != ignore_index)
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Compute accuracy
        correct = (predictions == targets).sum()
        total = targets.size
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def compute_top_k(predictions: np.ndarray, targets: np.ndarray, 
                      k: int = 5, ignore_index: Optional[int] = None) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            predictions: Predicted logits (batch_size, vocab_size)
            targets: Target indices (batch_size,)
            k: Top-k value
            ignore_index: Index to ignore
            
        Returns:
            Top-k accuracy
        """
        # Get top-k predictions
        top_k = np.argsort(predictions, axis=-1)[:, -k:]
        
        # Check if target is in top-k
        correct = 0
        total = 0
        
        for i in range(len(targets)):
            if ignore_index is not None and targets[i] == ignore_index:
                continue
            if targets[i] in top_k[i]:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0


class Perplexity:
    """
    Perplexity calculator for language models.
    """
    
    @staticmethod
    def compute(loss: float) -> float:
        """
        Compute perplexity from cross-entropy loss.
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity value
        """
        return np.exp(loss)


class ProgressTracker:
    """
    Progress tracker for training loops.
    """
    
    def __init__(self, total_steps: int, print_every: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            print_every: Print frequency
        """
        self.total_steps = total_steps
        self.print_every = print_every
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self):
        """Update progress."""
        self.current_step += 1
    
    def should_print(self) -> bool:
        """Check if should print progress."""
        return self.current_step % self.print_every == 0
    
    def print_progress(self, metrics: Metrics, prefix: str = ""):
        """
        Print training progress.
        
        Args:
            metrics: Metrics object
            prefix: Prefix string
        """
        if self.should_print():
            elapsed = time.time() - self.start_time
            eta = elapsed * (self.total_steps - self.current_step) / (self.current_step + 1)
            
            stats = metrics.get_stats()
            print(f"{prefix}[{self.current_step}/{self.total_steps}] "
                  f"Loss: {stats['loss']:.4f} | "
                  f"Acc: {stats['accuracy']:.4f} | "
                  f"Time: {elapsed:.1f}s | "
                  f"ETA: {eta:.1f}s")
    
    def is_finished(self) -> bool:
        """Check if training is finished."""
        return self.current_step >= self.total_steps