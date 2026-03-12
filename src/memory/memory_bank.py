"""
Memory bank implementation for Nuther (Retro Memory LSTM) neural network framework.
This module integrates memory storage and retrieval for dialogue context management.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time

from src.config import config
from .memory import Memory, MemoryChunk
from .retrieval import MemoryRetriever, SimilarityCalculator


class MemoryBank:
    """
    Memory bank for storing and retrieving dialogue context and knowledge.
    Integrates memory storage with keyword-weighted retrieval for efficient
    context expansion during dialogue generation.
    """
    
    def __init__(self, memory_id: Optional[str] = None,
                 max_chunks: int = config.MEMORY_SIZE,
                 top_k: int = config.RETRIEVAL_TOP_K,
                 threshold: float = config.SIMILARITY_THRESHOLD,
                 keyword_weight: float = config.KEYWORD_WEIGHT):
        """
        Initialize memory bank.
        
        Args:
            memory_id: Unique identifier
            max_chunks: Maximum number of chunks to store
            top_k: Number of top chunks to retrieve
            threshold: Minimum similarity threshold
            keyword_weight: Weight for keyword matching
        """
        # Initialize memory storage
        self.memory = Memory(memory_id=memory_id, max_chunks=max_chunks)
        
        # Initialize retriever
        similarity_calculator = SimilarityCalculator(keyword_weight=keyword_weight)
        self.retriever = MemoryRetriever(
            similarity_calculator=similarity_calculator,
            top_k=top_k,
            threshold=threshold
        )
        
        # Conversation history for context management
        self.conversation_history: List[Dict] = []
        self.max_history_length = 50  # Maximum number of turns to keep
    
    def store(self, text: str, metadata: Optional[Dict] = None,
              chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        Store text in memory bank with automatic chunking.
        
        Args:
            text: Text to store
            metadata: Additional metadata
            chunk_size: Size of chunks (uses config default if None)
            overlap: Overlap between chunks (uses config default if None)
            
        Returns:
            List of chunk IDs
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        chunk_ids = self.memory.add_chunks_from_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata=metadata
        )
        
        return chunk_ids
    
    def store_dialogue_turn(self, user_input: str, model_output: str, 
                            turn_id: Optional[int] = None) -> str:
        """
        Store a dialogue turn in conversation history.
        
        Args:
            user_input: User input text
            model_output: Model output text
            turn_id: Turn ID (auto-generated if None)
            
        Returns:
            Turn ID
        """
        if turn_id is None:
            turn_id = len(self.conversation_history)
        
        turn = {
            'turn_id': turn_id,
            'timestamp': time.time(),
            'user_input': user_input,
            'model_output': model_output
        }
        
        self.conversation_history.append(turn)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return turn_id
    
    def retrieve(self, query: str, top_k: Optional[int] = None,
                 threshold: Optional[float] = None) -> List[Tuple[MemoryChunk, float]]:
        """
        Retrieve relevant memory chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of top chunks (uses default if None)
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            List of (chunk, similarity) tuples
        """
        # Get all chunks
        chunks = self.memory.get_all_chunks()
        
        if not chunks:
            return []
        
        # Use custom parameters if provided
        original_top_k = self.retriever.top_k
        original_threshold = self.retriever.threshold
        
        if top_k is not None:
            self.retriever.set_top_k(top_k)
        if threshold is not None:
            self.retriever.set_threshold(threshold)
        
        # Retrieve
        results = self.retriever.retrieve(query, chunks)
        
        # Restore original parameters
        self.retriever.set_top_k(original_top_k)
        self.retriever.set_threshold(original_threshold)
        
        return results
    
    def retrieve_with_keywords(self, query: str, min_keyword_matches: int = 1,
                               top_k: Optional[int] = None) -> List[Tuple[MemoryChunk, float]]:
        """
        Retrieve chunks that match at least a minimum number of keywords.
        
        Args:
            query: Query text
            min_keyword_matches: Minimum number of keyword matches
            top_k: Number of top chunks (uses default if None)
            
        Returns:
            List of (chunk, similarity) tuples
        """
        # Get all chunks
        chunks = self.memory.get_all_chunks()
        
        if not chunks:
            return []
        
        # Use custom top_k if provided
        original_top_k = self.retriever.top_k
        if top_k is not None:
            self.retriever.set_top_k(top_k)
        
        # Retrieve
        results = self.retriever.retrieve_by_keywords(query, chunks, min_keyword_matches)
        
        # Restore original top_k
        self.retriever.set_top_k(original_top_k)
        
        return results
    
    def retrieve_with_explanation(self, query: str, 
                                   top_k: Optional[int] = None) -> List[Tuple[MemoryChunk, float, Dict]]:
        """
        Retrieve chunks with explanation of why they were selected.
        
        Args:
            query: Query text
            top_k: Number of top chunks (uses default if None)
            
        Returns:
            List of (chunk, similarity, explanation) tuples
        """
        # Get all chunks
        chunks = self.memory.get_all_chunks()
        
        if not chunks:
            return []
        
        # Use custom top_k if provided
        original_top_k = self.retriever.top_k
        if top_k is not None:
            self.retriever.set_top_k(top_k)
        
        # Retrieve
        results = self.retriever.retrieve_with_explanation(query, chunks)
        
        # Restore original top_k
        self.retriever.set_top_k(original_top_k)
        
        return results
    
    def get_context(self, query: str, max_context_length: int = 500,
                    top_k: Optional[int] = None) -> str:
        """
        Get context for a query by retrieving and combining relevant chunks.
        
        Args:
            query: Query text
            max_context_length: Maximum length of context in characters
            top_k: Number of top chunks to use
            
        Returns:
            Context text
        """
        # Retrieve relevant chunks
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return ''
        
        # Combine chunk contents
        context_parts = []
        current_length = 0
        
        for chunk, similarity in results:
            chunk_content = chunk.get_text()
            
            if current_length + len(chunk_content) <= max_context_length:
                context_parts.append(chunk_content)
                current_length += len(chunk_content)
            else:
                # Truncate last chunk if needed
                remaining = max_context_length - current_length
                if remaining > 0:
                    context_parts.append(chunk_content[:remaining])
                break
        
        # Combine with separator
        context = ' [SEP] '.join(context_parts)
        
        return context
    
    def get_conversation_context(self, last_n_turns: int = 3) -> str:
        """
        Get recent conversation context.
        
        Args:
            last_n_turns: Number of recent turns to include
            
        Returns:
            Conversation context text
        """
        recent_turns = self.conversation_history[-last_n_turns:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user_input']}")
            context_parts.append(f"Model: {turn['model_output']}")
        
        return '\n'.join(context_parts)
    
    def get_augmented_input(self, user_input: str, max_context_length: int = 500,
                            include_conversation: bool = True) -> str:
        """
        Get augmented input with retrieved context and conversation history.
        
        Args:
            user_input: User input text
            max_context_length: Maximum length of context
            include_conversation: Whether to include conversation history
            
        Returns:
            Augmented input text
        """
        context_parts = []
        
        # Add retrieved memory context
        memory_context = self.get_context(user_input, max_context_length)
        if memory_context:
            context_parts.append(f"Memory: {memory_context}")
        
        # Add conversation history
        if include_conversation:
            conversation_context = self.get_conversation_context(last_n_turns=2)
            if conversation_context:
                context_parts.append(f"Conversation:\n{conversation_context}")
        
        # Combine with user input
        if context_parts:
            augmented_input = '\n\n'.join(context_parts) + f'\n\nUser: {user_input}'
        else:
            augmented_input = user_input
        
        return augmented_input
    
    def get_statistics(self) -> Dict:
        """
        Get memory bank statistics.
        
        Returns:
            Statistics dictionary
        """
        memory_stats = self.memory.get_statistics()
        
        return {
            'memory_stats': memory_stats,
            'conversation_turns': len(self.conversation_history),
            'max_chunks': self.memory.max_chunks,
            'retriever_config': {
                'top_k': self.retriever.top_k,
                'threshold': self.retriever.threshold,
                'keyword_weight': self.retriever.similarity_calculator.keyword_weight
            }
        }
    
    def clear_memory(self):
        """Clear all stored memory but keep conversation history."""
        self.memory.clear()
    
    def clear_conversation_history(self):
        """Clear conversation history but keep memory."""
        self.conversation_history.clear()
    
    def clear_all(self):
        """Clear all memory and conversation history."""
        self.clear_memory()
        self.clear_conversation_history()
    
    def save(self, filepath: str):
        """
        Save memory bank to file.
        
        Args:
            filepath: Path to save
        """
        import os
        import json
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'memory_id': self.memory.memory_id,
            'max_chunks': self.memory.max_chunks,
            'conversation_history': self.conversation_history,
            'retriever_config': {
                'top_k': self.retriever.top_k,
                'threshold': self.retriever.threshold,
                'keyword_weight': self.retriever.similarity_calculator.keyword_weight
            }
        }
        
        # Save memory chunks
        memory_filepath = filepath.replace('.json', '_memory.json')
        self.memory.save(memory_filepath)
        
        # Save bank metadata
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """
        Load memory bank from file.
        
        Args:
            filepath: Path to load from
        """
        import json
        
        # Load bank metadata
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load memory chunks
        memory_filepath = filepath.replace('.json', '_memory.json')
        self.memory.load(memory_filepath)
        
        # Load conversation history
        self.conversation_history = data.get('conversation_history', [])
        
        # Update retriever config
        retriever_config = data.get('retriever_config', {})
        if 'top_k' in retriever_config:
            self.retriever.set_top_k(retriever_config['top_k'])
        if 'threshold' in retriever_config:
            self.retriever.set_threshold(retriever_config['threshold'])
    
    def get_memory(self) -> Memory:
        """
        Get underlying memory instance.
        
        Returns:
            Memory instance
        """
        return self.memory
    
    def get_retriever(self) -> MemoryRetriever:
        """
        Get underlying retriever instance.
        
        Returns:
            MemoryRetriever instance
        """
        return self.retriever
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MemoryBank(id={self.memory.memory_id}, chunks={len(self.memory.chunks)}, turns={len(self.conversation_history)})"