"""
Mixture of Experts (MoE) implementation for Nuther (Retro Memory LSTM) neural network framework.
This module integrates experts, gating, and output fusion for the complete MoE architecture.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union

from src.config import config
from .expert import Expert, FeedForwardExpert, LSTMExpert
from .gating import GatingNetwork, TopKGating, SoftGating


class MoE:
    """
    Mixture of Experts module with gating and output fusion.
    Routes inputs to selected experts and combines their outputs.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_experts: Optional[int] = None,
                 gating_type: str = 'top_k',
                 expert_type: str = 'feed_forward'):
        """
        Initialize MoE module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_experts: Number of experts (uses config default if None)
            gating_type: Type of gating ('top_k', 'soft')
            expert_type: Type of experts ('feed_forward', 'lstm')
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts or config.NUM_EXPERTS
        self.gating_type = gating_type.lower()
        self.expert_type = expert_type.lower()
        
        # Create gating network
        self.gating = self._create_gating_network()
        
        # Create experts
        self.experts: List[Expert] = []
        for i in range(self.num_experts):
            expert = self._create_expert(i)
            self.experts.append(expert)
    
    def _create_gating_network(self) -> GatingNetwork:
        """
        Create gating network based on type.
        
        Returns:
            Gating network instance
        """
        if self.gating_type == 'top_k':
            return TopKGating(self.input_dim, self.num_experts)
        elif self.gating_type == 'soft':
            return SoftGating(self.input_dim, self.num_experts)
        else:
            raise ValueError(f"Unknown gating type: {self.gating_type}")
    
    def _create_expert(self, expert_id: int) -> Expert:
        """
        Create expert based on type.
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            Expert instance
        """
        if self.expert_type == 'feed_forward':
            return FeedForwardExpert(self.input_dim, self.output_dim, expert_id)
        elif self.expert_type == 'lstm':
            return LSTMExpert(self.input_dim, self.output_dim, expert_id)
        else:
            raise ValueError(f"Unknown expert type: {self.expert_type}")
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Forward pass through MoE module.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output, gating_weights, expert_outputs) where:
                output: Combined output of shape (batch_size, output_dim)
                gating_weights: Gating weights of shape (batch_size, num_experts)
                expert_outputs: List of expert outputs
        """
        batch_size = x.shape[0]
        
        # Compute gating weights
        gating_weights = self.gating.forward(x)
        
        # Compute outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            if self.expert_type == 'lstm':
                # LSTM experts return (output, h, c)
                expert_output, _, _ = expert.forward(x)
            else:
                expert_output = expert.forward(x)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        expert_outputs_stacked = np.stack(expert_outputs, axis=1)  # (batch_size, num_experts, output_dim)
        
        # Combine outputs using gating weights
        gating_weights_expanded = gating_weights[:, :, np.newaxis]  # (batch_size, num_experts, 1)
        output = np.sum(expert_outputs_stacked * gating_weights_expanded, axis=1)  # (batch_size, output_dim)
        
        return output, gating_weights, expert_outputs
    
    def forward_with_routing_info(self, x: np.ndarray) -> Dict:
        """
        Forward pass with detailed routing information.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing output, gating weights, and routing info
        """
        batch_size = x.shape[0]
        
        # Compute gating weights
        gating_weights = self.gating.forward(x)
        
        # Get selected experts
        if hasattr(self.gating, 'get_selected_experts'):
            selected_experts = self.gating.get_selected_experts(x)
        else:
            # For soft gating, consider all experts
            selected_experts = [list(range(self.num_experts)) for _ in range(batch_size)]
        
        # Compute outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            if self.expert_type == 'lstm':
                expert_output, _, _ = expert.forward(x)
            else:
                expert_output = expert.forward(x)
            expert_outputs.append(expert_output)
        
        # Combine outputs
        expert_outputs_stacked = np.stack(expert_outputs, axis=1)
        gating_weights_expanded = gating_weights[:, :, np.newaxis]
        output = np.sum(expert_outputs_stacked * gating_weights_expanded, axis=1)
        
        return {
            'output': output,
            'gating_weights': gating_weights,
            'expert_outputs': expert_outputs,
            'selected_experts': selected_experts,
            'expert_utilization': self._compute_utilization(gating_weights)
        }
    
    def _compute_utilization(self, gating_weights: np.ndarray) -> np.ndarray:
        """
        Compute expert utilization statistics.
        
        Args:
            gating_weights: Gating weights of shape (batch_size, num_experts)
            
        Returns:
            Utilization array of shape (num_experts,)
        """
        # Average gating weight for each expert
        utilization = np.mean(gating_weights, axis=0)
        return utilization
    
    def get_parameters(self) -> Dict:
        """
        Get all parameters from gating and experts.
        
        Returns:
            Dictionary of all parameters
        """
        params = {
            'gating': self.gating.get_parameters()
        }
        
        for i, expert in enumerate(self.experts):
            expert_params = expert.get_parameters()
            for key, value in expert_params.items():
                params[f'expert_{i}_{key}'] = value
        
        return params
    
    def set_parameters(self, params: Dict):
        """
        Set all parameters.
        
        Args:
            params: Dictionary of all parameters
        """
        # Set gating parameters
        self.gating.set_parameters(params['gating'])
        
        # Set expert parameters
        for i, expert in enumerate(self.experts):
            expert_params = {}
            for key, value in params.items():
                if key.startswith(f'expert_{i}_'):
                    param_key = key[len(f'expert_{i}_'):]
                    expert_params[param_key] = value
            expert.set_parameters(expert_params)
    
    def reset_parameters(self):
        """Reset all parameters to initial values."""
        self.gating.reset_parameters()
        for expert in self.experts:
            expert.reset_parameters()
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        total = self.gating.get_parameter_count()
        for expert in self.experts:
            total += expert.get_parameter_count()
        return total
    
    def get_expert_by_id(self, expert_id: int) -> Optional[Expert]:
        """
        Get expert by ID.
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            Expert instance or None if not found
        """
        if 0 <= expert_id < len(self.experts):
            return self.experts[expert_id]
        return None
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MoE(input_dim={self.input_dim}, output_dim={self.output_dim}, num_experts={self.num_experts}, gating={self.gating_type})"


class SparseMoE(MoE):
    """
    Sparse Mixture of Experts with load balancing.
    Encourages balanced expert usage to prevent expert collapse.
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 num_experts: Optional[int] = None,
                 top_k: Optional[int] = None,
                 load_balance_weight: float = 0.01):
        """
        Initialize Sparse MoE module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_experts: Number of experts
            top_k: Number of top experts to select
            load_balance_weight: Weight for load balancing loss
        """
        super().__init__(input_dim, output_dim, num_experts, gating_type='top_k')
        
        self.top_k = top_k or config.TOP_K_EXPERTS
        self.load_balance_weight = load_balance_weight
        
        # Override gating with specific top_k
        self.gating = TopKGating(input_dim, num_experts or config.NUM_EXPERTS, top_k)
        
        # Track expert statistics for load balancing
        self.expert_selection_count = np.zeros(self.num_experts)
        self.total_samples = 0
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass with load balancing tracking.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output, gating_weights, load_balance_loss)
        """
        batch_size = x.shape[0]
        
        # Compute gating weights
        gating_weights = self.gating.forward(x)
        
        # Track expert selections
        selected_experts = self.gating.get_selected_experts(x)
        for expert_list in selected_experts:
            for expert_id in expert_list:
                self.expert_selection_count[expert_id] += 1
        self.total_samples += batch_size
        
        # Compute outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            if self.expert_type == 'lstm':
                expert_output, _, _ = expert.forward(x)
            else:
                expert_output = expert.forward(x)
            expert_outputs.append(expert_output)
        
        # Combine outputs
        expert_outputs_stacked = np.stack(expert_outputs, axis=1)
        gating_weights_expanded = gating_weights[:, :, np.newaxis]
        output = np.sum(expert_outputs_stacked * gating_weights_expanded, axis=1)
        
        # Compute load balance loss
        load_balance_loss = self._compute_load_balance_loss(gating_weights)
        
        return output, gating_weights, load_balance_loss
    
    def _compute_load_balance_loss(self, gating_weights: np.ndarray) -> float:
        """
        Compute load balancing loss to encourage balanced expert usage.
        
        Args:
            gating_weights: Gating weights of shape (batch_size, num_experts)
            
        Returns:
            Load balance loss value
        """
        # Fraction of samples routed to each expert
        expert_fraction = np.mean(gating_weights, axis=0)
        
        # Expected fraction if perfectly balanced
        expected_fraction = 1.0 / self.num_experts
        
        # Load balance loss (mean squared error)
        load_balance_loss = np.mean((expert_fraction - expected_fraction) ** 2)
        
        return load_balance_loss * self.load_balance_weight
    
    def get_expert_utilization(self) -> np.ndarray:
        """
        Get expert utilization statistics.
        
        Returns:
            Utilization array of shape (num_experts,)
        """
        if self.total_samples == 0:
            return np.zeros(self.num_experts)
        return self.expert_selection_count / (self.total_samples * self.top_k)
    
    def reset_utilization_tracking(self):
        """Reset expert utilization tracking."""
        self.expert_selection_count = np.zeros(self.num_experts)
        self.total_samples = 0
    
    def get_parameters(self) -> Dict:
        """
        Get all parameters.
        
        Returns:
            Dictionary of all parameters
        """
        params = super().get_parameters()
        params['load_balance_weight'] = self.load_balance_weight
        params['top_k'] = self.top_k
        return params
    
    def set_parameters(self, params: Dict):
        """
        Set all parameters.
        
        Args:
            params: Dictionary of all parameters
        """
        # Extract MoE parameters
        moe_params = {
            'gating': params['gating']
        }
        for key, value in params.items():
            if key.startswith('expert_'):
                moe_params[key] = value
        
        super().set_parameters(moe_params)
        
        # Set SparseMoE specific parameters
        self.load_balance_weight = params.get('load_balance_weight', self.load_balance_weight)
        self.top_k = params.get('top_k', self.top_k)
    
    def __repr__(self) -> str:
        """String representation."""
        utilization = self.get_expert_utilization()
        return f"SparseMoE(input_dim={self.input_dim}, output_dim={self.output_dim}, num_experts={self.num_experts}, top_k={self.top_k}, utilization={utilization})"


class MoELayer:
    """
    MoE layer that can be used as a replacement for standard layers.
    Provides a simple interface for integrating MoE into larger models.
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 num_experts: int = config.NUM_EXPERTS,
                 top_k: int = config.TOP_K_EXPERTS,
                 use_sparse: bool = True):
        """
        Initialize MoE layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_experts: Number of experts
            top_k: Number of top experts to select
            use_sparse: Whether to use sparse MoE with load balancing
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_sparse = use_sparse
        
        if use_sparse:
            self.moe = SparseMoE(input_dim, output_dim, num_experts, top_k)
        else:
            self.moe = MoE(input_dim, output_dim, num_experts, gating_type='top_k')
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if self.use_sparse:
            output, _, _ = self.moe.forward(x)
        else:
            output, _, _ = self.moe.forward(x)
        return output
    
    def get_moe(self) -> Union[MoE, SparseMoE]:
        """
        Get underlying MoE module.
        
        Returns:
            MoE or SparseMoE instance
        """
        return self.moe
    
    def get_output_dim(self) -> int:
        """
        Get output dimension.
        
        Returns:
            Output dimension
        """
        return self.output_dim
    
    def get_parameter_count(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            Number of parameters
        """
        return self.moe.get_parameter_count()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MoELayer(input_dim={self.input_dim}, output_dim={self.output_dim}, num_experts={self.num_experts}, sparse={self.use_sparse})"