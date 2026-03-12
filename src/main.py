# -*- coding: utf-8 -*-
# Copyright 2026(C) Dinnerb0ne<tomma_2022@outlook.com>
#
#    Copyright 2026 [Dinnerb0ne]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# date: 2026-03-12
# version: 0.1
# description: Retro Memory LSTM (Nuther)
# LICENSE: Apache-2.0


"""
Main entry point for Nuther (Retro Memory LSTM) neural network framework.
This module serves as the administrator that coordinates and controls all modules.
"""

import sys
import os
import argparse
from typing import Optional, Dict, List

# Handle both module and script execution
if __name__ == '__main__' or __package__ is None:
    # Running as script: add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    from src.config import config
    from src.vocab import Vocabulary
    from src.crawler import CrawlerPipeline, create_sample_knowledge_base
    from src.model import NutherModel
    from src.chat import ChatBot
else:
    # Running as module: use relative imports
    from .config import config
    from .vocab import Vocabulary
    from .crawler import CrawlerPipeline, create_sample_knowledge_base
    from .model import NutherModel
    from .chat import ChatBot


class NutherFramework:
    """
    Main framework administrator that coordinates all modules.
    Provides unified interface for managing the complete Nuther system.
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 vocab_path: Optional[str] = None):
        """
        Initialize Nuther framework.
        
        Args:
            model_path: Path to saved model (None for new model)
            vocab_path: Path to saved vocabulary (None for new vocab)
        """
        self.model: Optional[NutherModel] = None
        self.vocab: Optional[Vocabulary] = None
        self.chat_bot: Optional[ChatBot] = None
        self.crawler: Optional[CrawlerPipeline] = None
        
        # Initialize from saved model or create new
        if model_path and vocab_path:
            self.load(model_path, vocab_path)
        else:
            self.initialize_new()
    
    def initialize_new(self):
        """Initialize new model and vocabulary."""
        print("Initializing new Nuther model...")
        
        # Create vocabulary
        self.vocab = Vocabulary(vocab_size=config.VOCAB_SIZE)
        
        # Create model
        self.model = NutherModel(
            vocab=self.vocab,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            use_memory=True
        )
        
        # Create chat bot
        self.chat_bot = ChatBot(self.model)
        
        # Create crawler
        self.crawler = CrawlerPipeline()
        
        print(f"Model initialized with {self.model.get_parameter_count()['total']:,} parameters")
    
    def load(self, model_path: str, vocab_path: str):
        """
        Load saved model and vocabulary.
        
        Args:
            model_path: Path to model directory
            vocab_path: Path to vocabulary file
        """
        print("Loading Nuther model...")
        
        # Load vocabulary
        self.vocab = Vocabulary()
        self.vocab.load(vocab_path)
        
        # Load model
        self.model = NutherModel(vocab=self.vocab)
        self.model.load(model_path)
        
        # Create chat bot
        self.chat_bot = ChatBot(self.model)
        
        # Create crawler
        self.crawler = CrawlerPipeline()
        
        print("Model loaded successfully")
    
    def save(self, save_dir: str):
        """
        Save model and vocabulary.
        
        Args:
            save_dir: Directory to save to
        """
        print(f"Saving model to {save_dir}...")
        self.model.save(save_dir)
        print("Model saved successfully")
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        print("Building vocabulary...")
        self.vocab.build_vocab(texts)
        print(f"Vocabulary built with {self.vocab.get_vocab_size()} tokens")
    
    def load_knowledge_base(self, file_path: str):
        """
        Load knowledge from file into memory.
        
        Args:
            file_path: Path to knowledge file
        """
        print(f"Loading knowledge from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.model.store_knowledge(text)
        print("Knowledge loaded successfully")
    
    def crawl_and_store(self, urls: List[str], max_pages: int = config.CRAWLER_MAX_PAGES):
        """
        Crawl websites and store knowledge.
        
        Args:
            urls: List of URLs to crawl
            max_pages: Maximum pages to crawl
        """
        print(f"Crawling {len(urls)} URLs...")
        stored_count = self.crawler.crawl_and_store(urls, max_pages)
        
        # Store crawled content in model memory
        kb = self.crawler.get_knowledge_bank()
        corpus = kb.get_text_corpus()
        
        for text in corpus:
            self.model.store_knowledge(text)
        
        print(f"Crawled and stored {stored_count} documents")
    
    def chat(self, max_length: Optional[int] = None,
             temperature: Optional[float] = None):
        """
        Start interactive chat.
        
        Args:
            max_length: Maximum output length
            temperature: Sampling temperature
        """
        print("\nStarting interactive chat...")
        self.chat_bot.interactive_chat(max_length=max_length, temperature=temperature)
    
    def generate(self, input_text: str, max_length: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """
        Generate response for input text.
        
        Args:
            input_text: Input text
            max_length: Maximum output length
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        return self.model.generate(input_text, max_length, temperature)
    
    def get_statistics(self) -> Dict:
        """
        Get framework statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'model_params': self.model.get_parameter_count(),
            'vocab_size': self.vocab.get_vocab_size()
        }
        
        if self.model.get_memory_bank():
            memory_stats = self.model.get_memory_bank().get_statistics()
            stats['memory'] = memory_stats
        
        if self.chat_bot:
            stats['chat_sessions'] = len(self.chat_bot.get_all_sessions())
            stats['current_session'] = self.chat_bot.current_session_id
        
        return stats
    
    def print_statistics(self):
        """Print framework statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("  Nuther Framework Statistics")
        print("="*60)
        print(f"Vocabulary Size: {stats['vocab_size']:,}")
        print(f"Model Parameters:")
        print(f"  - Encoder: {stats['model_params']['encoder']:,}")
        print(f"  - Decoder: {stats['model_params']['decoder']:,}")
        print(f"  - Total: {stats['model_params']['total']:,}")
        
        if 'memory' in stats:
            print(f"Memory:")
            print(f"  - Chunks: {stats['memory']['memory_stats']['total_chunks']}")
            print(f"  - Tokens: {stats['memory']['memory_stats']['total_tokens']}")
            print(f"  - Conversation Turns: {stats['memory']['conversation_turns']}")
        
        if 'chat_sessions' in stats:
            print(f"Chat:")
            print(f"  - Sessions: {stats['chat_sessions']}")
            print(f"  - Current: {stats['current_session']}")
        
        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Nuther (Retro Memory LSTM) - Dialogue Neural Network Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive chat with sample knowledge
  python -m src.main chat --sample-knowledge
  
  # Crawl websites and chat
  python -m src.main chat --crawl https://example.com
  
  # Load saved model and chat
  python -m src.main chat --model data/model --vocab data/vocab.txt
  
  # Generate single response
  python -m src.main generate --input "Hello" --sample-knowledge
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat')
    chat_parser.add_argument('--model', type=str, help='Path to saved model')
    chat_parser.add_argument('--vocab', type=str, help='Path to vocabulary file')
    chat_parser.add_argument('--sample-knowledge', action='store_true',
                           help='Load sample knowledge base')
    chat_parser.add_argument('--crawl', type=str, nargs='+',
                           help='URLs to crawl for knowledge')
    chat_parser.add_argument('--max-length', type=int, default=config.MAX_SEQ_LENGTH,
                           help='Maximum output length')
    chat_parser.add_argument('--temperature', type=float, default=0.8,
                           help='Sampling temperature')
    chat_parser.add_argument('--stats', action='store_true',
                           help='Show statistics before chat')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate single response')
    gen_parser.add_argument('--input', type=str, required=True,
                          help='Input text')
    gen_parser.add_argument('--model', type=str, help='Path to saved model')
    gen_parser.add_argument('--vocab', type=str, help='Path to vocabulary file')
    gen_parser.add_argument('--sample-knowledge', action='store_true',
                           help='Load sample knowledge base')
    gen_parser.add_argument('--max-length', type=int, default=config.MAX_SEQ_LENGTH,
                           help='Maximum output length')
    gen_parser.add_argument('--temperature', type=float, default=0.8,
                           help='Sampling temperature')
    
    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Crawl websites for knowledge')
    crawl_parser.add_argument('urls', type=str, nargs='+', help='URLs to crawl')
    crawl_parser.add_argument('--max-pages', type=int, default=config.CRAWLER_MAX_PAGES,
                            help='Maximum pages to crawl')
    crawl_parser.add_argument('--output', type=str, default='data/crawled',
                            help='Output directory')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show framework statistics')
    stats_parser.add_argument('--model', type=str, help='Path to saved model')
    stats_parser.add_argument('--vocab', type=str, help='Path to vocabulary file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize framework
    if args.command in ['chat', 'generate', 'stats']:
        if args.model and args.vocab:
            framework = NutherFramework(model_path=args.model, vocab_path=args.vocab)
        else:
            framework = NutherFramework()
    else:
        framework = NutherFramework()
    
    # Execute command
    if args.command == 'chat':
        # Load sample knowledge if requested
        if args.sample_knowledge:
            print("Loading sample knowledge base...")
            create_sample_knowledge_base()
            framework.load_knowledge_base('data/knowledge_base/indexed/index.json')
        
        # Crawl if requested
        if args.crawl:
            framework.crawl_and_store(args.crawl)
        
        # Show statistics if requested
        if args.stats:
            framework.print_statistics()
        
        # Start chat
        framework.chat(max_length=args.max_length, temperature=args.temperature)
    
    elif args.command == 'generate':
        # Load sample knowledge if requested
        if args.sample_knowledge:
            create_sample_knowledge_base()
            framework.load_knowledge_base('data/knowledge_base/indexed/index.json')
        
        # Generate response
        response = framework.generate(
            args.input, max_length=args.max_length, temperature=args.temperature
        )
        print(f"Input: {args.input}")
        print(f"Response: {response}")
    
    elif args.command == 'crawl':
        # Crawl websites
        framework.crawl_and_store(args.urls, max_pages=args.max_pages)
    
    elif args.command == 'stats':
        # Show statistics
        framework.print_statistics()


if __name__ == '__main__':
    main()