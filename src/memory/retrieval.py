"""
Memory retrieval implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements memory retrieval with keyword-weighted chunk matching
for efficient and context-aware memory access.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

from src.config import config
from .memory import MemoryChunk


class SimilarityCalculator:
    """
    Calculator for computing similarity between queries and memory chunks.
    Supports multiple similarity metrics including keyword-weighted matching.
    """
    
    def __init__(self, keyword_weight: float = config.KEYWORD_WEIGHT):
        """
        Initialize similarity calculator.
        
        Args:
            keyword_weight: Weight for keyword matching in similarity computation
        """
        self.keyword_weight = keyword_weight
        self.semantic_weight = 1.0 - keyword_weight
    
    def compute_similarity(self, query: str, chunk: MemoryChunk) -> float:
        """
        Compute similarity between query and memory chunk.
        Combines keyword matching and semantic similarity.
        
        Args:
            query: Query text
            chunk: Memory chunk
            
        Returns:
            Similarity score between 0 and 1
        """
        # Compute keyword similarity
        keyword_sim = self._compute_keyword_similarity(query, chunk.keywords)
        
        # Compute semantic similarity (token overlap)
        semantic_sim = self._compute_semantic_similarity(query, chunk.content)
        
        # Combine with weights
        similarity = self.keyword_weight * keyword_sim + self.semantic_weight * semantic_sim
        
        return similarity
    
    def _compute_keyword_similarity(self, query: str, chunk_keywords: Dict[str, float]) -> float:
        """
        Compute keyword-based similarity.
        
        Args:
            query: Query text
            chunk_keywords: Dictionary of chunk keywords with weights
            
        Returns:
            Keyword similarity score
        """
        if not chunk_keywords:
            return 0.0
        
        # Extract query keywords
        query_keywords = self._extract_query_keywords(query)
        
        if not query_keywords:
            return 0.0
        
        # Compute overlap
        matched_keywords = set(query_keywords.keys()) & set(chunk_keywords.keys())
        
        if not matched_keywords:
            return 0.0
        
        # Weighted similarity based on keyword importance
        similarity = 0.0
        for keyword in matched_keywords:
            # Combine query and chunk keyword weights
            keyword_sim = query_keywords[keyword] * chunk_keywords[keyword]
            similarity += keyword_sim
        
        # Normalize by number of matched keywords
        similarity = similarity / len(matched_keywords)
        
        return similarity
    
    def _extract_query_keywords(self, query: str, max_keywords: int = 10) -> Dict[str, float]:
        """
        Extract keywords from query with weights.
        
        Args:
            query: Query text
            max_keywords: Maximum number of keywords
            
        Returns:
            Dictionary of keyword -> weight pairs
        """
        # Tokenize query
        tokens = self._tokenize(query)
        
        # Count frequencies
        from collections import Counter
        token_freq = Counter(tokens)
        
        # Filter stopwords and short tokens
        stopwords = self._get_stopwords()
        filtered_tokens = {
            token: freq for token, freq in token_freq.items()
            if token not in stopwords and len(token) > 1
        }
        
        # Sort by frequency
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: (-x[1], x[0]))
        
        # Take top keywords and normalize
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
    
    def _compute_semantic_similarity(self, query: str, content: str) -> float:
        """
        Compute semantic similarity using token overlap (Jaccard similarity).
        
        Args:
            query: Query text
            content: Content text
            
        Returns:
            Semantic similarity score
        """
        query_tokens = set(self._tokenize(query))
        content_tokens = set(self._tokenize(content))
        
        if not query_tokens or not content_tokens:
            return 0.0
        
        # Jaccard similarity
        intersection = query_tokens & content_tokens
        union = query_tokens | content_tokens
        
        if not union:
            return 0.0
        
        jaccard_sim = len(intersection) / len(union)
        
        return jaccard_sim
    
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
    
    def compute_embedding_similarity(self, query_embedding: np.ndarray, 
                                     chunk_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            query_embedding: Query embedding vector
            chunk_embedding: Chunk embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        query_norm = np.linalg.norm(query_embedding)
        chunk_norm = np.linalg.norm(chunk_embedding)
        
        if query_norm == 0 or chunk_norm == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(query_embedding, chunk_embedding) / (query_norm * chunk_norm)
        
        return float(similarity)


class MemoryRetriever:
    """
    Memory retriever for finding relevant chunks based on queries.
    Uses keyword-weighted matching for efficient retrieval.
    """
    
    def __init__(self, similarity_calculator: Optional[SimilarityCalculator] = None,
                 top_k: int = config.RETRIEVAL_TOP_K,
                 threshold: float = config.SIMILARITY_THRESHOLD):
        """
        Initialize memory retriever.
        
        Args:
            similarity_calculator: Similarity calculator instance
            top_k: Number of top chunks to retrieve
            threshold: Minimum similarity threshold
        """
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()
        self.top_k = top_k
        self.threshold = threshold
    
    def retrieve(self, query: str, chunks: List[MemoryChunk]) -> List[Tuple[MemoryChunk, float]]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: Query text
            chunks: List of memory chunks to search
            
        Returns:
            List of (chunk, similarity) tuples sorted by similarity
        """
        if not chunks:
            return []
        
        # Compute similarities
        chunk_similarities = []
        for chunk in chunks:
            similarity = self.similarity_calculator.compute_similarity(query, chunk)
            if similarity >= self.threshold:
                chunk_similarities.append((chunk, similarity))
        
        # Sort by similarity (descending)
        chunk_similarities.sort(key=lambda x: -x[1])
        
        # Return top-k
        return chunk_similarities[:self.top_k]
    
    def retrieve_by_keywords(self, query: str, chunks: List[MemoryChunk],
                             min_keyword_matches: int = 1) -> List[Tuple[MemoryChunk, float]]:
        """
        Retrieve chunks that match at least a minimum number of keywords.
        
        Args:
            query: Query text
            chunks: List of memory chunks to search
            min_keyword_matches: Minimum number of keyword matches required
            
        Returns:
            List of (chunk, similarity) tuples sorted by similarity
        """
        if not chunks:
            return []
        
        # Extract query keywords
        query_keywords = self.similarity_calculator._extract_query_keywords(query)
        
        if not query_keywords:
            return []
        
        # Filter chunks by keyword matches
        filtered_chunks = []
        for chunk in chunks:
            matched_keywords = set(query_keywords.keys()) & set(chunk.keywords.keys())
            if len(matched_keywords) >= min_keyword_matches:
                similarity = self.similarity_calculator.compute_similarity(query, chunk)
                if similarity >= self.threshold:
                    filtered_chunks.append((chunk, similarity))
        
        # Sort by similarity (descending)
        filtered_chunks.sort(key=lambda x: -x[1])
        
        # Return top-k
        return filtered_chunks[:self.top_k]
    
    def retrieve_with_explanation(self, query: str, 
                                   chunks: List[MemoryChunk]) -> List[Tuple[MemoryChunk, float, Dict]]:
        """
        Retrieve chunks with explanation of why they were selected.
        
        Args:
            query: Query text
            chunks: List of memory chunks to search
            
        Returns:
            List of (chunk, similarity, explanation) tuples
        """
        if not chunks:
            return []
        
        results = []
        
        # Extract query keywords
        query_keywords = self.similarity_calculator._extract_query_keywords(query)
        
        for chunk in chunks:
            similarity = self.similarity_calculator.compute_similarity(query, chunk)
            
            if similarity >= self.threshold:
                # Compute explanation
                matched_keywords = set(query_keywords.keys()) & set(chunk.keywords.keys())
                
                explanation = {
                    'matched_keywords': list(matched_keywords),
                    'keyword_weight': sum(chunk.keywords[k] for k in matched_keywords),
                    'query_keywords': list(query_keywords.keys()),
                    'chunk_keywords': list(chunk.keywords.keys())
                }
                
                results.append((chunk, similarity, explanation))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: -x[1])
        
        # Return top-k
        return results[:self.top_k]
    
    def batch_retrieve(self, queries: List[str], 
                       chunks: List[MemoryChunk]) -> List[List[Tuple[MemoryChunk, float]]]:
        """
        Retrieve relevant chunks for multiple queries.
        
        Args:
            queries: List of query texts
            chunks: List of memory chunks to search
            
        Returns:
            List of retrieval results for each query
        """
        results = []
        for query in queries:
            result = self.retrieve(query, chunks)
            results.append(result)
        return results
    
    def set_top_k(self, top_k: int):
        """
        Set top-k parameter.
        
        Args:
            top_k: Number of top chunks to retrieve
        """
        self.top_k = top_k
    
    def set_threshold(self, threshold: float):
        """
        Set similarity threshold.
        
        Args:
            threshold: Minimum similarity threshold
        """
        self.threshold = threshold
    
    def get_statistics(self, queries: List[str], 
                       chunks: List[MemoryChunk]) -> Dict:
        """
        Get retrieval statistics.
        
        Args:
            queries: List of query texts
            chunks: List of memory chunks to search
            
        Returns:
            Statistics dictionary
        """
        total_queries = len(queries)
        total_retrieved = 0
        avg_similarity = 0.0
        
        for query in queries:
            results = self.retrieve(query, chunks)
            total_retrieved += len(results)
            if results:
                avg_similarity += sum(sim for _, sim in results) / len(results)
        
        avg_similarity = avg_similarity / total_queries if total_queries > 0 else 0.0
        
        return {
            'total_queries': total_queries,
            'total_chunks': len(chunks),
            'avg_retrieved_per_query': total_retrieved / total_queries if total_queries > 0 else 0,
            'avg_similarity': avg_similarity,
            'top_k': self.top_k,
            'threshold': self.threshold
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MemoryRetriever(top_k={self.top_k}, threshold={self.threshold})"
