"""
Decoder implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements the decoder that generates responses using LSTM and MoE.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List

from src.config import config
from src.lstm import EmbeddingLSTM
from src.moe import MoE, SparseMoE


class Decoder:
    """
    Decoder that generates responses using LSTM and MoE.
    Processes encoder output and generates output token by token.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = config.NUM_LAYERS, cell_dim: Optional[int] = None,
                 use_moe: bool = True, num_experts: int = config.NUM_EXPERTS,
                 top_k: int = config.TOP_K_EXPERTS):
        """
        Initialize decoder.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            cell_dim: Cell state dimension
            use_moe: Whether to use MoE for output projection
            num_experts: Number of experts in MoE
            top_k: Top-K experts to select
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_dim = cell_dim if cell_dim is not None else hidden_dim
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create embedding LSTM
        self.embedding_lstm = EmbeddingLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cell_dim=cell_dim,
            bidirectional=False
        )
        
        # Create output projection
        if use_moe:
            # Use MoE for output projection
            self.output_projection = SparseMoE(
                input_dim=hidden_dim,
                output_dim=vocab_size,
                num_experts=num_experts,
                top_k=top_k
            )
        else:
            # Simple linear projection
            scale = np.sqrt(2.0 / (hidden_dim + vocab_size))
            self.W_out = np.random.randn(hidden_dim, vocab_size) * scale
            self.b_out = np.zeros(vocab_size)
    
    def forward(self, input_indices: np.ndarray, h_init: List[np.ndarray],
                c_init: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        """
        Forward pass through decoder.
        
        Args:
            input_indices: Input token indices of shape (batch_size, seq_len)
            h_init: Initial hidden states from encoder
            c_init: Initial cell states from encoder
            h_prev: Previous hidden state (for autoregressive decoding)
            c_prev: Previous cell state (for autoregressive decoding)
            
        Returns:
            Tuple of (output_logits, h_final, c_final, moe_loss) where:
                output_logits: Output logits of shape (batch_size, seq_len, vocab_size)
                h_final: Final hidden states
                c_final: Final cell states
                moe_loss: MoE load balance loss (None if not using MoE)
        """
        # Process through embedding LSTM
        output, h_final, c_final = self.embedding_lstm.forward_inference(
            input_indices, h_init, c_init
        )
        
        # Project to vocabulary logits
        if output.ndim == 3:
            # Sequence output: (batch_size, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = output.shape
            output_flat = output.reshape(-1, hidden_dim)
            
            if self.use_moe:
                logits_flat, _, moe_loss = self.output_projection.forward(output_flat)
                logits = logits_flat.reshape(batch_size, seq_len, self.vocab_size)
            else:
                logits_flat = output_flat @ self.W_out + self.b_out
                logits = logits_flat.reshape(batch_size, seq_len, self.vocab_size)
                moe_loss = None
        else:
            # Single step output: (batch_size, hidden_dim)
            if self.use_moe:
                logits, _, moe_loss = self.output_projection.forward(output)
            else:
                logits = output @ self.W_out + self.b_out
                moe_loss = None
        
        return logits, h_final, c_final, moe_loss
    
    def forward_step(self, input_token: np.ndarray, h_prev: List[np.ndarray],
                     c_prev: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        """
        Single step forward pass for autoregressive decoding.
        
        Args:
            input_token: Input token indices of shape (batch_size, 1)
            h_prev: Previous hidden states
            c_prev: Previous cell states
            
        Returns:
            Tuple of (output_logits, h_next, c_next, moe_loss)
        """
        # Ensure input has sequence dimension
        if input_token.ndim == 1:
            input_token = input_token.reshape(-1, 1)
        
        # Forward pass
        return self.forward(input_token, h_prev, c_prev)
    
    def decode(self, encoder_output: np.ndarray, h_init: List[np.ndarray],
               c_init: List[np.ndarray], max_length: int = config.MAX_SEQ_LENGTH,
               temperature: float = 1.0) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Autoregressive decoding to generate output sequence.
        
        Args:
            encoder_output: Encoder output (not used directly in this simple version)
            h_init: Initial hidden states from encoder
            c_init: Initial cell states from encoder
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Tuple of (output_indices, h_final, c_final)
        """
        batch_size = h_init[0].shape[0]
        
        # Initialize with start token
        current_tokens = np.full((batch_size, 1), config.START_TOKEN_ID, dtype=np.int32)
        output_indices = []
        
        h_prev = h_init
        c_prev = c_init
        
        for _ in range(max_length):
            # Forward step
            logits, h_next, c_next, _ = self.forward_step(current_tokens, h_prev, c_prev)
            
            # Sample next token
            if temperature > 0:
                # Apply temperature
                logits = logits / temperature
                # Softmax and sample
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                next_tokens = np.array([np.random.choice(self.vocab_size, p=probs[i]) 
                                       for i in range(batch_size)])
            else:
                # Greedy decoding
                next_tokens = np.argmax(logits, axis=-1)
            
            output_indices.append(next_tokens)
            current_tokens = next_tokens.reshape(-1, 1)
            
            h_prev = h_next
            c_prev = c_next
            
            # Check if all sequences have generated end token
            if np.all(next_tokens == config.END_TOKEN_ID):
                break
        
        # Stack output indices
        output_indices = np.stack(output_indices, axis=1)  # (batch_size, seq_len)
        
        return output_indices, h_prev, c_prev
    
    def get_parameters(self) -> Dict:
        """
        Get decoder parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = {
            'embedding_lstm': self.embedding_lstm.get_parameters()
        }
        
        if self.use_moe:
            params['output_projection'] = self.output_projection.get_parameters()
        else:
            params['W_out'] = self.W_out.copy()
            params['b_out'] = self.b_out.copy()
        
        return params
    
    def set_parameters(self, params: Dict):
        """
        Set decoder parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.embedding_lstm.set_parameters(params['embedding_lstm'])
        
        if self.use_moe:
            self.output_projection.set_parameters(params['output_projection'])
        else:
            self.W_out = params['W_out'].copy()
            self.b_out = params['b_out'].copy()
    
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self.embedding_lstm.reset_parameters()
        
        if self.use_moe:
            self.output_projection.reset_parameters()
        else:
            scale = np.sqrt(2.0 / (self.hidden_dim + self.vocab_size))
            self.W_out = np.random.randn(self.hidden_dim, self.vocab_size) * scale
            self.b_out = np.zeros(self.vocab_size)
    
    def get_output_dim(self) -> int:
        """
        Get output dimension (vocabulary size).
        
        Returns:
            Output dimension
        """
        return self.vocab_size
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        total = self.embedding_lstm.get_parameter_count()
        
        if self.use_moe:
            total += self.output_projection.get_parameter_count()
        else:
            total += self.W_out.size + self.b_out.size
        
        return total
    
    def train(self):
        """Set to training mode."""
        self.embedding_lstm.train()
        if self.use_moe:
            self.output_projection.moe.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.embedding_lstm.eval()
        if self.use_moe:
            self.output_projection.moe.eval()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Decoder(vocab_size={self.vocab_size}, hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, use_moe={self.use_moe})"