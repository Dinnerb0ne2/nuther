"""
Nuther Model implementation for Nuther (Retro Memory LSTM) neural network framework.
This module integrates encoder and decoder for the complete sequence-to-sequence model.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import os

from src.config import config
from src.vocab import Vocabulary
from src.memory import MemoryBank
from .encoder import Encoder
from .decoder import Decoder


class NutherModel:
    """
    Complete Nuther model integrating LSTM, memory retrieval, and MoE.
    Implements sequence-to-sequence architecture with memory-augmented encoding.
    """
    
    def __init__(self, vocab: Vocabulary, embedding_dim: int = config.EMBEDDING_DIM,
                 hidden_dim: int = config.HIDDEN_DIM, num_layers: int = config.NUM_LAYERS,
                 cell_dim: Optional[int] = None, encoder_bidirectional: bool = False,
                 decoder_use_moe: bool = True, num_experts: int = config.NUM_EXPERTS,
                 top_k: int = config.TOP_K_EXPERTS, use_memory: bool = True):
        """
        Initialize Nuther model.
        
        Args:
            vocab: Vocabulary instance
            embedding_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            cell_dim: Cell state dimension
            encoder_bidirectional: Whether encoder is bidirectional
            decoder_use_moe: Whether decoder uses MoE
            num_experts: Number of experts in MoE
            top_k: Top-K experts to select
            use_memory: Whether to use memory retrieval
        """
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_dim = cell_dim if cell_dim is not None else hidden_dim
        self.encoder_bidirectional = encoder_bidirectional
        self.decoder_use_moe = decoder_use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_memory = use_memory
        
        # Create memory bank
        self.memory_bank = MemoryBank() if use_memory else None
        
        # Calculate encoder output dimension
        if encoder_bidirectional:
            encoder_output_dim = hidden_dim * 2
        else:
            encoder_output_dim = hidden_dim
        
        # Create encoder
        self.encoder = Encoder(
            vocab_size=vocab.get_vocab_size(),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cell_dim=cell_dim,
            bidirectional=encoder_bidirectional,
            use_memory=use_memory,
            memory_bank=self.memory_bank
        )
        
        # Create decoder
        self.decoder = Decoder(
            vocab_size=vocab.get_vocab_size(),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cell_dim=cell_dim,
            use_moe=decoder_use_moe,
            num_experts=num_experts,
            top_k=top_k
        )
    
    def forward(self, input_indices: np.ndarray, 
                target_indices: Optional[np.ndarray] = None,
                max_output_length: Optional[int] = None) -> Dict:
        """
        Forward pass through the model.
        
        Args:
            input_indices: Input token indices of shape (batch_size, input_seq_len)
            target_indices: Target token indices of shape (batch_size, target_seq_len)
                          (None for autoregressive generation)
            max_output_length: Maximum output length for generation
            
        Returns:
            Dictionary containing outputs and metadata
        """
        batch_size = input_indices.shape[0]
        
        # Encoder forward pass
        encoder_output, h_encoder, c_encoder, memory_context = self.encoder.forward(
            input_indices, retrieve_memory=self.use_memory
        )
        
        # Initialize decoder states from encoder
        h_decoder = h_encoder
        c_decoder = c_encoder
        
        # Decoder forward pass
        if target_indices is not None:
            # Teacher forcing: use target indices
            logits, h_final, c_final, moe_loss = self.decoder.forward(
                target_indices, h_decoder, c_decoder
            )
            
            return {
                'output_logits': logits,
                'encoder_output': encoder_output,
                'h_final': h_final,
                'c_final': c_final,
                'memory_context': memory_context,
                'moe_loss': moe_loss
            }
        else:
            # Autoregressive generation
            max_len = max_output_length or config.MAX_SEQ_LENGTH
            output_indices, h_final, c_final = self.decoder.decode(
                encoder_output, h_decoder, c_decoder, max_length=max_len
            )
            
            return {
                'output_indices': output_indices,
                'encoder_output': encoder_output,
                'h_final': h_final,
                'c_final': c_final,
                'memory_context': memory_context
            }
    
    def generate(self, input_text: str, max_length: int = config.MAX_SEQ_LENGTH,
                 temperature: float = 1.0) -> str:
        """
        Generate response for input text.
        
        Args:
            input_text: Input text string
            max_length: Maximum output length
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        # Encode input text to indices
        input_indices = self.vocab.text_to_indices(
            input_text,
            add_start=False,
            add_end=False,
            max_length=config.MAX_SEQ_LENGTH
        )
        input_indices = np.array([input_indices], dtype=np.int32)
        
        # Forward pass
        result = self.forward(input_indices, max_output_length=max_length)
        
        # Decode output indices to text
        output_indices = result['output_indices'][0]
        output_text = self.vocab.indices_to_text(output_indices, skip_special=True)
        
        return output_text
    
    def generate_with_memory(self, input_text: str, max_length: int = config.MAX_SEQ_LENGTH,
                            temperature: float = 1.0) -> Tuple[str, str]:
        """
        Generate response with memory context.
        
        Args:
            input_text: Input text string
            max_length: Maximum output length
            temperature: Sampling temperature
            
        Returns:
            Tuple of (response_text, memory_context)
        """
        # Store input in memory
        if self.memory_bank:
            self.memory_bank.store(input_text, metadata={'type': 'user_input'})
        
        # Generate response
        response_text = self.generate(input_text, max_length, temperature)
        
        # Store response in memory
        if self.memory_bank:
            self.memory_bank.store(response_text, metadata={'type': 'model_output'})
            # Store dialogue turn
            self.memory_bank.store_dialogue_turn(input_text, response_text)
        
        # Get memory context
        memory_context = ''
        if self.memory_bank:
            memory_context = self.memory_bank.get_context(input_text, max_context_length=200)
        
        return response_text, memory_context
    
    def chat(self, user_input: str, max_length: int = config.MAX_SEQ_LENGTH,
             temperature: float = 1.0) -> Dict:
        """
        Interactive chat with the model.
        
        Args:
            user_input: User input text
            max_length: Maximum output length
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing response and metadata
        """
        # Generate response
        response_text, memory_context = self.generate_with_memory(
            user_input, max_length, temperature
        )
        
        return {
            'user_input': user_input,
            'response': response_text,
            'memory_context': memory_context,
            'memory_stats': self.memory_bank.get_statistics() if self.memory_bank else None
        }
    
    def store_knowledge(self, text: str, metadata: Optional[Dict] = None):
        """
        Store knowledge text in memory.
        
        Args:
            text: Knowledge text
            metadata: Additional metadata
        """
        if self.memory_bank:
            self.memory_bank.store(text, metadata)
    
    def store_knowledge_from_file(self, filepath: str, metadata: Optional[Dict] = None):
        """
        Store knowledge from file in memory.
        
        Args:
            filepath: Path to knowledge file
            metadata: Additional metadata
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        self.store_knowledge(text, metadata)
    
    def get_parameters(self) -> Dict:
        """
        Get all model parameters.
        
        Returns:
            Dictionary of all parameters
        """
        return {
            'encoder': self.encoder.get_parameters(),
            'decoder': self.decoder.get_parameters()
        }
    
    def set_parameters(self, params: Dict):
        """
        Set all model parameters.
        
        Args:
            params: Dictionary of all parameters
        """
        self.encoder.set_parameters(params['encoder'])
        self.decoder.set_parameters(params['decoder'])
    
    def reset_parameters(self):
        """Reset all parameters to initial values."""
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
    
    def get_parameter_count(self) -> Dict:
        """
        Get parameter counts for each component.
        
        Returns:
            Dictionary of parameter counts
        """
        return {
            'encoder': self.encoder.get_parameter_count(),
            'decoder': self.decoder.get_parameter_count(),
            'total': self.encoder.get_parameter_count() + self.decoder.get_parameter_count()
        }
    
    def save(self, save_dir: str):
        """
        Save model to directory.
        
        Args:
            save_dir: Directory to save model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vocabulary
        vocab_path = os.path.join(save_dir, 'vocab.txt')
        self.vocab.save(vocab_path)
        
        # Save parameters
        params_path = os.path.join(save_dir, 'model_params.npz')
        params = self.get_parameters()
        
        # Convert nested dict to flat structure
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_params[f'{key}_{subkey}'] = subvalue
            else:
                flat_params[key] = value
        
        np.savez_compressed(params_path, **flat_params)
        
        # Save memory bank
        if self.memory_bank:
            memory_path = os.path.join(save_dir, 'memory_bank.json')
            self.memory_bank.save(memory_path)
        
        # Save config
        config_data = {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'cell_dim': self.cell_dim,
            'encoder_bidirectional': self.encoder_bidirectional,
            'decoder_use_moe': self.decoder_use_moe,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'use_memory': self.use_memory
        }
        
        import json
        config_path = os.path.join(save_dir, 'model_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def load(self, load_dir: str):
        """
        Load model from directory.
        
        Args:
            load_dir: Directory to load model from
        """
        # Load vocabulary
        vocab_path = os.path.join(load_dir, 'vocab.txt')
        self.vocab.load(vocab_path)
        
        # Load parameters
        params_path = os.path.join(load_dir, 'model_params.npz')
        flat_params = np.load(params_path)
        
        # Reconstruct nested dict
        params = {
            'encoder': {},
            'decoder': {}
        }
        
        for key in flat_params.files:
            if key.startswith('encoder_'):
                encoder_key = key[len('encoder_'):]
                params['encoder'][encoder_key] = flat_params[key]
            elif key.startswith('decoder_'):
                decoder_key = key[len('decoder_'):]
                params['decoder'][decoder_key] = flat_params[key]
        
        self.set_parameters(params)
        
        # Load memory bank
        if self.memory_bank:
            memory_path = os.path.join(load_dir, 'memory_bank.json')
            if os.path.exists(memory_path):
                self.memory_bank.load(memory_path)
    
    def train(self):
        """Set to training mode."""
        self.encoder.train()
        self.decoder.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.encoder.eval()
        self.decoder.eval()
    
    def get_memory_bank(self) -> Optional[MemoryBank]:
        """
        Get memory bank instance.
        
        Returns:
            Memory bank instance or None
        """
        return self.memory_bank
    
    def __repr__(self) -> str:
        """String representation."""
        param_counts = self.get_parameter_count()
        return f"NutherModel(vocab_size={self.vocab.get_vocab_size()}, hidden_dim={self.hidden_dim}, total_params={param_counts['total']})"