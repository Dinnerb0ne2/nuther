# -*- coding: utf-8 -*-
"""
Training script for Nuther neural network.
Supports simple and advanced training modes.
"""

import sys
import os
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.vocab import Vocabulary
from src.model import NutherModel
from src.training import (
    SimpleTrainer, Trainer, Adam, SGD,
    Checkpoint, Metrics
)


def train_simple(args):
    """
    Simple training with random data.
    Good for quick testing and demonstration.
    """
    
    print("  Simple Training Mode")
    
    print()
    
    # Load or create vocabulary
    if args.vocab and os.path.exists(args.vocab):
        print(f"Loading vocabulary from {args.vocab}")
        vocab = Vocabulary(vocab_size=args.vocab_size)
        vocab.load(args.vocab)
    else:
        print(f"Creating vocabulary with {args.vocab_size} tokens")
        vocab = Vocabulary(vocab_size=args.vocab_size)
        # Add some basic tokens
        vocab.build_vocab(["hello", "world", "test", "training", "model"])
    
    # Create model (small for faster training)
    print("Creating model...")
    model = NutherModel(
        vocab=vocab,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        use_memory=False  # Disable memory for simple training
    )
    
    param_count = model.get_parameter_count()['total']
    print(f"Model parameters: {param_count:,}")
    print()
    
    # Create simple trainer
    trainer = SimpleTrainer(
        model=model,
        vocab_size=vocab.get_vocab_size(),
        learning_rate=args.learning_rate
    )
    
    # Train
    metrics = trainer.train_simple(
        num_steps=args.steps,
        batch_size=args.batch_size
    )
    
    # Save model
    if args.save_model:
        save_dir = os.path.dirname(args.save_model)
        os.makedirs(save_dir, exist_ok=True)
        vocab.save(args.save_model.replace('.pkl', '_vocab.txt'))
        from src.training.checkpoint import save_model
        save_model(model, args.save_model)
        print(f"\nModel saved to {args.save_model}")
    
    # Test generation
    print("\n")
    print("  Testing Generation")
    
    
    test_inputs = ["hello", "world", "test"]
    for input_text in test_inputs:
        response = model.generate(input_text, max_length=10, temperature=0.8)
        print(f"  Input: {input_text}")
        print(f"  Output: {response or '(no response)'}")
        print()


def train_advanced(args):
    """
    Advanced training with data files.
    """
    
    print("  Advanced Training Mode")
    
    print()
    
    # Load vocabulary
    vocab_path = args.vocab or "data/chinese_vocab.txt"
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print("Please run: python prepare_chinese_data.py")
        return
    
    print(f"Loading vocabulary from {vocab_path}")
    vocab = Vocabulary(vocab_size=args.vocab_size)
    vocab.load(vocab_path)
    print(f"Vocabulary size: {vocab.get_vocab_size()}")
    print()
    
    # Load training data
    data_path = args.data or "data/chinese_training.txt"
    if not os.path.exists(data_path):
        print(f"Error: Training data file not found: {data_path}")
        print("Please run: python prepare_chinese_data.py")
        return
    
    print(f"Loading training data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse dialogues
    dialogues = []
    for line in lines:
        if '|' in line:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                context = parts[0].strip()
                response = parts[1].strip()
                if context and response:
                    dialogues.append((context, response))
    
    print(f"Loaded {len(dialogues)} dialogues")
    print()
    
    # Create model
    print("Creating model...")
    model = NutherModel(
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_memory=True
    )
    
    param_counts = model.get_parameter_count()
    print(f"Model parameters: {param_counts['total']:,}")
    print()
    
    # Prepare batches
    print("Preparing batches...")
    batches = []
    batch_size = min(args.batch_size, len(dialogues))
    
    for i in range(0, len(dialogues), batch_size):
        batch_dialogues = dialogues[i:i + batch_size]
        
        # Encode batch
        contexts = []
        responses = []
        
        for context, response in batch_dialogues:
            context_indices = vocab.text_to_indices(
                context, add_start=True, add_end=False, 
                max_length=args.max_seq_length
            )
            response_indices = vocab.text_to_indices(
                response, add_start=True, add_end=True,
                max_length=args.max_seq_length
            )
            contexts.append(context_indices)
            responses.append(response_indices)
        
        # Pad sequences
        max_ctx_len = max(len(ctx) for ctx in contexts)
        max_resp_len = max(len(resp) for resp in responses)
        
        padded_contexts = []
        padded_responses = []
        
        for ctx in contexts:
            padded = ctx + [config.PAD_TOKEN_ID] * (max_ctx_len - len(ctx))
            padded_contexts.append(padded)
        
        for resp in responses:
            padded = resp + [config.PAD_TOKEN_ID] * (max_resp_len - len(resp))
            padded_responses.append(padded)
        
        batches.append({
            'context': np.array(padded_contexts, dtype=np.int32),
            'response': np.array(padded_responses, dtype=np.int32)
        })
    
    print(f"Created {len(batches)} batches")
    print()
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.learning_rate, momentum=0.9)
    else:
        print(f"Unknown optimizer: {args.optimizer}, using Adam")
        optimizer = Adam(learning_rate=args.learning_rate)
    
    # Create checkpoint manager
    checkpoint_manager = None
    if args.save_model:
        save_dir = os.path.dirname(args.save_model)
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_manager = Checkpoint(save_dir)
    
    # Create trainer
    trainer = Trainer(model, optimizer)
    
    # Train
    metrics = trainer.train(
        train_batches=batches,
        num_epochs=args.epochs,
        checkpoint_manager=checkpoint_manager,
        print_every=10,
        save_every=50,
        eval_every=25
    )
    
    # Save final model
    if args.save_model:
        vocab.save(args.save_model.replace('.pkl', '_vocab.txt'))
        from src.training.checkpoint import save_model
        save_model(model, args.save_model)
        print(f"\nModel saved to {args.save_model}")
    
    # Test generation
    print("\n")
    print("  Testing Generation")
    
    
    test_inputs = ["你好", "你是谁", "什么是神经网络"]
    for input_text in test_inputs:
        response = model.generate(input_text, max_length=15, temperature=0.8)
        print(f"  Input: {input_text}")
        print(f"  Output: {response or '(no response)'}")
        print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Train Nuther neural network model'
    )
    
    # Mode
    parser.add_argument(
        '--mode', type=str, default='simple',
        choices=['simple', 'advanced'],
        help='Training mode (simple=random data, advanced=real data)'
    )
    
    # Data
    parser.add_argument('--vocab', type=str, default=None,
                       help='Path to vocabulary file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data file')
    parser.add_argument('--vocab-size', type=int, default=2000,
                       help='Vocabulary size')
    
    # Model
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--max-seq-length', type=int, default=20,
                       help='Maximum sequence length')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of training steps (simple mode)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer to use')
    
    # Output
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    if args.mode == 'simple':
        train_simple(args)
    else:
        train_advanced(args)


if __name__ == '__main__':
    main()