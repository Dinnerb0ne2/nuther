"""
Memory storage implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements memory chunks and memory storage with keyword extraction.
"""

import re
import hashlib
import json
import time
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from datetime import datetime
import numpy as np

from src.config import config


class MemoryChunk:
    """
    A chunk of memory containing text content with metadata.
    Chunks are the basic units stored in memory for retrieval.
    """
    
    def __init__(self, content: str, chunk_id: Optional[str] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize memory chunk.
        
        Args:
            content: Text content of the chunk
            chunk_id: Unique identifier (auto-generated if None)
            metadata: Additional metadata dictionary
        """
        self.content = content
        self.chunk_id = chunk_id if chunk_id else self._generate_id()
        self.metadata = metadata if metadata else {}
        
        # Set creation timestamp
        self.metadata['created_at'] = datetime.now().isoformat()
        
        # Extract keywords
        self.keywords = self._extract_keywords(content)
        
        # Compute content hash for deduplication
        self.content_hash = self._compute_hash(content)
    
    def _generate_id(self) -> str:
        """
        Generate unique chunk ID.
        
        Returns:
            Unique chunk ID
        """
        timestamp = str(time.time()).encode('utf-8')
        return hashlib.md5(timestamp).hexdigest()[:16]
    
    def _compute_hash(self, content: str) -> str:
        """
        Compute hash of content for deduplication.
        
        Args:
            content: Text content
            
        Returns:
            Content hash
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> Dict[str, float]:
        """
        Extract keywords from content with weights.
        
        Args:
            content: Text content
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            Dictionary of keyword -> weight pairs
        """
        # Tokenize content
        tokens = self._tokenize(content)
        
        # Count token frequencies
        token_freq = Counter(tokens)
        
        # Filter out common stopwords and short tokens
        stopwords = self._get_stopwords()
        filtered_tokens = {
            token: freq for token, freq in token_freq.items()
            if token not in stopwords and len(token) > 1
        }
        
        # Sort by frequency
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: (-x[1], x[0]))
        
        # Take top keywords and normalize weights
        top_keywords = sorted_tokens[:max_keywords]
        
        if top_keywords:
            max_freq = max(freq for _, freq in top_keywords)
            keywords = {
                token: freq / max_freq
                for token, freq in top_keywords
            }
        else:
            keywords = {}
        
        return keywords
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:]', ' ', text)
        
        # Split into tokens
        tokens = re.findall(r'\b[\w\u4e00-\u9fff]+\b', text)
        
        return tokens
    
    def _get_stopwords(self) -> Set[str]:
        """
        Get set of stopwords.
        
        Returns:
            Set of stopwords
        """
        # Common English and Chinese stopwords
        english_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'you\'re', 'you\'ve', 'you\'ll', 'you\'d', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'she\'s', 'her',
            'hers', 'herself', 'it', 'it\'s', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
            'this', 'that', 'that\'ll', 'these', 'those', 'am', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'don\'t', 'should', 'should\'ve', 'now'
        }
        
        chinese_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '那', '个', '之', '与', '及', '等', '为', '以',
            '对', '于', '把', '被', '给', '让', '从', '向', '由', '或', '而', '且',
            '又', '亦', '乃', '其', '它', '咱', '咱们', '您', '你们', '他们', '她们'
        }
        
        return english_stopwords.union(chinese_stopwords)
    
    def get_text(self) -> str:
        """
        Get text content.
        
        Returns:
            Text content
        """
        return self.content
    
    def get_keywords(self) -> Dict[str, float]:
        """
        Get keywords with weights.
        
        Returns:
            Dictionary of keyword -> weight pairs
        """
        return self.keywords
    
    def update_metadata(self, key: str, value):
        """
        Update metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'keywords': self.keywords,
            'content_hash': self.content_hash,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryChunk':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            MemoryChunk instance
        """
        chunk = cls(
            content=data['content'],
            chunk_id=data['chunk_id'],
            metadata=data.get('metadata', {})
        )
        chunk.keywords = data.get('keywords', {})
        chunk.content_hash = data.get('content_hash', '')
        return chunk
    
    def __repr__(self) -> str:
        """String representation."""
        content_preview = self.content[:50] + '...' if len(self.content) > 50 else self.content
        return f"MemoryChunk(id={self.chunk_id}, content='{content_preview}', keywords={list(self.keywords.keys())})"


class Memory:
    """
    Memory storage for storing and managing memory chunks.
    Supports adding, retrieving, and managing chunks with deduplication.
    """
    
    def __init__(self, memory_id: Optional[str] = None, 
                 max_chunks: int = config.MEMORY_SIZE):
        """
        Initialize memory storage.
        
        Args:
            memory_id: Unique identifier (auto-generated if None)
            max_chunks: Maximum number of chunks to store
        """
        self.memory_id = memory_id if memory_id else self._generate_id()
        self.max_chunks = max_chunks
        
        # Storage for chunks
        self.chunks: Dict[str, MemoryChunk] = {}
        
        # Content hash to chunk ID mapping for deduplication
        self.content_hashes: Dict[str, str] = {}
        
        # Access history for chunk prioritization
        self.access_count: Dict[str, int] = {}
        self.last_access: Dict[str, float] = {}
    
    def _generate_id(self) -> str:
        """
        Generate unique memory ID.
        
        Returns:
            Unique memory ID
        """
        timestamp = str(time.time()).encode('utf-8')
        return hashlib.md5(timestamp).hexdigest()[:16]
    
    def add_chunk(self, content: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Add a new chunk to memory.
        
        Args:
            content: Text content
            metadata: Additional metadata
            
        Returns:
            Chunk ID if added, None if duplicate or memory full
        """
        # Create temporary chunk to compute hash
        temp_chunk = MemoryChunk(content, metadata=metadata)
        
        # Check for duplicates
        if temp_chunk.content_hash in self.content_hashes:
            return None
        
        # Check memory capacity
        if len(self.chunks) >= self.max_chunks:
            # Remove least recently used chunk
            self._evict_lru_chunk()
        
        # Add chunk
        self.chunks[temp_chunk.chunk_id] = temp_chunk
        self.content_hashes[temp_chunk.content_hash] = temp_chunk.chunk_id
        self.access_count[temp_chunk.chunk_id] = 0
        self.last_access[temp_chunk.chunk_id] = time.time()
        
        return temp_chunk.chunk_id
    
    def add_chunks_from_text(self, text: str, chunk_size: int = config.CHUNK_SIZE,
                             overlap: int = config.CHUNK_OVERLAP,
                             metadata: Optional[Dict] = None) -> List[str]:
        """
        Split text into chunks and add to memory.
        
        Args:
            text: Text to split into chunks
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks
            metadata: Base metadata for all chunks
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []
        tokens = self._tokenize_text(text)
        
        if not tokens:
            return chunk_ids
        
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = ' '.join(chunk_tokens)
            
            # Add metadata with chunk number
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_number'] = chunk_num
            chunk_metadata['start_token'] = start_idx
            chunk_metadata['end_token'] = end_idx
            
            chunk_id = self.add_chunk(chunk_text, chunk_metadata)
            if chunk_id:
                chunk_ids.append(chunk_id)
            
            # Move to next chunk with overlap
            start_idx = end_idx - overlap
            chunk_num += 1
            
            # Prevent infinite loop if overlap >= chunk_size
            if start_idx <= 0:
                break
        
        return chunk_ids
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:]', ' ', text)
        
        # Split into tokens
        tokens = re.findall(r'\b[\w\u4e00-\u9fff]+\b', text)
        
        return tokens
    
    def _evict_lru_chunk(self):
        """
        Evict least recently used chunk.
        """
        if not self.chunks:
            return
        
        # Find chunk with oldest last access time
        lru_chunk_id = min(self.last_access, key=self.last_access.get)
        
        # Remove chunk
        chunk = self.chunks[lru_chunk_id]
        del self.content_hashes[chunk.content_hash]
        del self.chunks[lru_chunk_id]
        del self.access_count[lru_chunk_id]
        del self.last_access[lru_chunk_id]
    
    def get_chunk(self, chunk_id: str) -> Optional[MemoryChunk]:
        """
        Get chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            MemoryChunk if found, None otherwise
        """
        if chunk_id in self.chunks:
            # Update access tracking
            self.access_count[chunk_id] += 1
            self.last_access[chunk_id] = time.time()
            return self.chunks[chunk_id]
        return None
    
    def get_all_chunks(self) -> List[MemoryChunk]:
        """
        Get all chunks.
        
        Returns:
            List of all chunks
        """
        return list(self.chunks.values())
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            True if removed, False if not found
        """
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            del self.content_hashes[chunk.content_hash]
            del self.chunks[chunk_id]
            del self.access_count[chunk_id]
            del self.last_access[chunk_id]
            return True
        return False
    
    def clear(self):
        """Clear all chunks."""
        self.chunks.clear()
        self.content_hashes.clear()
        self.access_count.clear()
        self.last_access.clear()
    
    def get_size(self) -> int:
        """
        Get number of chunks in memory.
        
        Returns:
            Number of chunks
        """
        return len(self.chunks)
    
    def is_full(self) -> bool:
        """
        Check if memory is full.
        
        Returns:
            True if memory is full
        """
        return len(self.chunks) >= self.max_chunks
    
    def get_statistics(self) -> Dict:
        """
        Get memory statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.chunks:
            return {
                'total_chunks': 0,
                'total_tokens': 0,
                'avg_chunk_size': 0,
                'most_accessed': None,
                'access_stats': {}
            }
        
        total_tokens = sum(len(chunk._tokenize(chunk.content)) for chunk in self.chunks.values())
        avg_chunk_size = total_tokens / len(self.chunks)
        
        most_accessed_id = max(self.access_count, key=self.access_count.get) if self.access_count else None
        
        return {
            'total_chunks': len(self.chunks),
            'total_tokens': total_tokens,
            'avg_chunk_size': avg_chunk_size,
            'most_accessed': most_accessed_id,
            'access_stats': dict(self.access_count)
        }
    
    def save(self, filepath: str):
        """
        Save memory to file.
        
        Args:
            filepath: Path to save memory
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'memory_id': self.memory_id,
            'max_chunks': self.max_chunks,
            'chunks': [chunk.to_dict() for chunk in self.chunks.values()],
            'access_count': self.access_count,
            'last_access': self.last_access
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """
        Load memory from file.
        
        Args:
            filepath: Path to load memory from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.memory_id = data['memory_id']
        self.max_chunks = data['max_chunks']
        
        # Load chunks
        self.chunks = {}
        self.content_hashes = {}
        for chunk_data in data['chunks']:
            chunk = MemoryChunk.from_dict(chunk_data)
            self.chunks[chunk.chunk_id] = chunk
            self.content_hashes[chunk.content_hash] = chunk.chunk_id
        
        # Load access tracking
        self.access_count = data.get('access_count', {})
        self.last_access = data.get('last_access', {})
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Memory(id={self.memory_id}, chunks={len(self.chunks)}/{self.max_chunks})"