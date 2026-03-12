"""
Web crawler module for Nuther (Retro Memory LSTM) neural network framework.
This module handles web crawling, text cleaning, and knowledge base management.
"""

import os
import re
import time
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Set
from urllib.parse import urljoin, urlparse
from collections import deque
import requests
from bs4 import BeautifulSoup

from src.config import config


class TextCleaner:
    """Text cleaner for preprocessing crawled text data."""
    
    def __init__(self):
        """Initialize text cleaner."""
        # Common stopwords for English and Chinese
        self.english_stopwords = {
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
            'can', 'will', 'just', 'don', 'don\'t', 'should', 'should\'ve', 'now',
            'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'aren\'t',
            'couldn', 'couldn\'t', 'didn', 'didn\'t', 'doesn', 'doesn\'t', 'hadn',
            'hadn\'t', 'hasn', 'hasn\'t', 'haven', 'haven\'t', 'isn', 'isn\'t',
            'ma', 'mightn', 'mightn\'t', 'mustn', 'mustn\'t', 'needn', 'needn\'t',
            'shan', 'shan\'t', 'shouldn', 'shouldn\'t', 'wasn', 'wasn\'t', 'weren',
            'weren\'t', 'won', 'won\'t', 'wouldn', 'wouldn\'t'
        }
        
        self.chinese_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '那', '个', '之', '与', '及', '等', '为', '以',
            '对', '于', '把', '被', '给', '让', '从', '向', '由', '或', '而', '且',
            '又', '亦', '乃', '其', '它', '咱', '咱们', '您', '你们', '他们', '她们',
            '它们', '大家', '各位', '谁', '什么', '哪里', '哪个', '多少', '怎样', '如何'
        }
        
        self.all_stopwords = self.english_stopwords.union(self.chinese_stopwords)
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text containing HTML tags
            
        Returns:
            Text with HTML tags removed
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        return text
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text containing URLs
            
        Returns:
            Text with URLs removed
        """
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        return text
    
    def remove_email_addresses(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Args:
            text: Input text containing email addresses
            
        Returns:
            Text with email addresses removed
        """
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters but keep basic punctuation.
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
        """
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff.,!?;:()\'"-]', ' ', text)
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing whitespace
        text = text.strip()
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token.lower() not in self.all_stopwords]
        return ' '.join(filtered_tokens)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by applying all cleaning steps.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned and normalized text
        """
        # Apply cleaning steps in order
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_email_addresses(text)
        text = self.remove_special_characters(text)
        text = self.remove_extra_whitespace(text)
        text = self.remove_stopwords(text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split by sentence delimiters
        sentences = re.split(r'[.!?。！？]+', text)
        # Clean and filter empty sentences
        sentences = [self.remove_extra_whitespace(s) for s in sentences if s.strip()]
        return sentences
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from text.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        # Split by newlines
        paragraphs = re.split(r'\n+', text)
        # Clean and filter empty paragraphs
        paragraphs = [self.remove_extra_whitespace(p) for p in paragraphs if p.strip()]
        return paragraphs


class WebCrawler:
    """Web crawler for scraping text content from web pages."""
    
    def __init__(self, timeout: int = config.CRAWLER_TIMEOUT, 
                 delay: float = config.CRAWLER_DELAY):
        """
        Initialize web crawler.
        
        Args:
            timeout: Request timeout in seconds
            delay: Delay between requests in seconds
        """
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.visited_urls: Set[str] = set()
        self.text_cleaner = TextCleaner()
    
    def crawl_page(self, url: str) -> Optional[Dict[str, str]]:
        """
        Crawl a single web page and extract text content.
        
        Args:
            url: URL to crawl
            
        Returns:
            Dictionary containing 'url', 'title', 'content', 'cleaned_content'
        """
        try:
            # Avoid crawling the same URL twice
            if url in self.visited_urls:
                return None
            
            self.visited_urls.add(url)
            
            # Make request
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.title.string.strip() if soup.title else ''
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                script.decompose()
            
            # Extract text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean the text
            cleaned_text = self.text_cleaner.normalize_text(text)
            
            # Extract links for further crawling
            links = self._extract_links(soup, url)
            
            result = {
                'url': url,
                'title': title,
                'content': text,
                'cleaned_content': cleaned_text,
                'links': links
            }
            
            # Add delay to avoid being blocked
            time.sleep(self.delay)
            
            return result
            
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None
    
    def crawl_multiple_pages(self, urls: List[str], max_pages: int = config.CRAWLER_MAX_PAGES) -> List[Dict[str, str]]:
        """
        Crawl multiple web pages.
        
        Args:
            urls: List of URLs to crawl
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of crawled page data
        """
        results = []
        url_queue = deque(urls)
        crawled_count = 0
        
        while url_queue and crawled_count < max_pages:
            url = url_queue.popleft()
            
            result = self.crawl_page(url)
            if result:
                results.append(result)
                crawled_count += 1
                
                # Add discovered links to queue
                for link in result['links']:
                    if link not in self.visited_urls:
                        url_queue.append(link)
        
        return results
    
    def crawl_site(self, base_url: str, max_pages: int = config.CRAWLER_MAX_PAGES,
                   restrict_to_domain: bool = True) -> List[Dict[str, str]]:
        """
        Crawl an entire website starting from base URL.
        
        Args:
            base_url: Base URL to start crawling
            max_pages: Maximum number of pages to crawl
            restrict_to_domain: Whether to restrict crawling to the same domain
            
        Returns:
            List of crawled page data
        """
        # Parse base URL
        parsed = urlparse(base_url)
        base_domain = parsed.netloc
        
        results = []
        url_queue = deque([base_url])
        crawled_count = 0
        
        while url_queue and crawled_count < max_pages:
            url = url_queue.popleft()
            
            result = self.crawl_page(url)
            if result:
                results.append(result)
                crawled_count += 1
                
                # Add discovered links to queue
                for link in result['links']:
                    if link not in self.visited_urls:
                        if restrict_to_domain:
                            link_parsed = urlparse(link)
                            if link_parsed.netloc == base_domain:
                                url_queue.append(link)
                        else:
                            url_queue.append(link)
        
        return results
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract all links from a parsed page.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of absolute URLs
        """
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Convert relative URLs to absolute URLs
            absolute_url = urljoin(base_url, href)
            
            # Filter out non-HTTP links and fragments
            parsed = urlparse(absolute_url)
            if parsed.scheme in ['http', 'https'] and not parsed.fragment:
                links.append(absolute_url)
        
        return links
    
    def close(self):
        """Close the crawler session."""
        self.session.close()


class KnowledgeBase:
    """Knowledge base for storing and managing crawled text data."""
    
    def __init__(self, base_dir: str = config.KNOWLEDGE_BASE_DIR):
        """
        Initialize knowledge base.
        
        Args:
            base_dir: Base directory for storing knowledge base files
        """
        self.base_dir = base_dir
        self.text_cleaner = TextCleaner()
        
        # Create directories if they don't exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'cleaned'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'indexed'), exist_ok=True)
    
    def add_document(self, url: str, title: str, content: str, 
                     cleaned_content: str) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            url: Document URL
            title: Document title
            content: Raw content
            cleaned_content: Cleaned content
            
        Returns:
            Document ID
        """
        # Generate document ID from URL hash
        doc_id = hashlib.md5(url.encode('utf-8')).hexdigest()
        
        # Save raw document
        raw_data = {
            'id': doc_id,
            'url': url,
            'title': title,
            'content': content
        }
        raw_path = os.path.join(self.base_dir, 'raw', f'{doc_id}.json')
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        
        # Save cleaned document
        cleaned_data = {
            'id': doc_id,
            'url': url,
            'title': title,
            'content': cleaned_content,
            'sentences': self.text_cleaner.extract_sentences(cleaned_content),
            'paragraphs': self.text_cleaner.extract_paragraphs(cleaned_content)
        }
        cleaned_path = os.path.join(self.base_dir, 'cleaned', f'{doc_id}.json')
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        # Update index
        self._update_index(doc_id, url, title, cleaned_content)
        
        return doc_id
    
    def _update_index(self, doc_id: str, url: str, title: str, content: str):
        """
        Update the knowledge base index.
        
        Args:
            doc_id: Document ID
            url: Document URL
            title: Document title
            content: Document content
        """
        index_path = os.path.join(self.base_dir, 'indexed', 'index.json')
        
        # Load existing index or create new one
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {'documents': {}}
        
        # Add document to index
        index['documents'][doc_id] = {
            'url': url,
            'title': title,
            'word_count': len(content.split()),
            'sentence_count': len(self.text_cleaner.extract_sentences(content))
        }
        
        # Save index
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def get_document(self, doc_id: str, cleaned: bool = True) -> Optional[Dict]:
        """
        Get a document from the knowledge base.
        
        Args:
            doc_id: Document ID
            cleaned: Whether to get cleaned version
            
        Returns:
            Document data or None if not found
        """
        if cleaned:
            doc_path = os.path.join(self.base_dir, 'cleaned', f'{doc_id}.json')
        else:
            doc_path = os.path.join(self.base_dir, 'raw', f'{doc_id}.json')
        
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_all_documents(self, cleaned: bool = True) -> List[Dict]:
        """
        Get all documents from the knowledge base.
        
        Args:
            cleaned: Whether to get cleaned versions
            
        Returns:
            List of document data
        """
        doc_dir = os.path.join(self.base_dir, 'cleaned' if cleaned else 'raw')
        documents = []
        
        if os.path.exists(doc_dir):
            for filename in os.listdir(doc_dir):
                if filename.endswith('.json'):
                    doc_path = os.path.join(doc_dir, filename)
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        documents.append(json.load(f))
        
        return documents
    
    def get_index(self) -> Dict:
        """
        Get the knowledge base index.
        
        Returns:
            Index dictionary
        """
        index_path = os.path.join(self.base_dir, 'indexed', 'index.json')
        
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'documents': {}}
    
    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """
        Search documents by keyword.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of matching documents
        """
        results = []
        documents = self.get_all_documents(cleaned=True)
        
        keyword_lower = keyword.lower()
        
        for doc in documents:
            # Search in title and content
            if (keyword_lower in doc['title'].lower() or 
                keyword_lower in doc['content'].lower()):
                results.append(doc)
        
        return results
    
    def get_text_corpus(self) -> List[str]:
        """
        Get all text content as a corpus for vocabulary building.
        
        Returns:
            List of text strings
        """
        documents = self.get_all_documents(cleaned=True)
        corpus = [doc['content'] for doc in documents]
        return corpus
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Statistics dictionary
        """
        index = self.get_index()
        documents = self.get_all_documents(cleaned=True)
        
        total_words = sum(doc['content'].count(' ') + 1 for doc in documents)
        total_sentences = sum(len(doc.get('sentences', [])) for doc in documents)
        total_paragraphs = sum(len(doc.get('paragraphs', [])) for doc in documents)
        
        return {
            'total_documents': len(documents),
            'total_words': total_words,
            'total_sentences': total_sentences,
            'total_paragraphs': total_paragraphs,
            'average_words_per_document': total_words / len(documents) if documents else 0
        }


class CrawlerPipeline:
    """Pipeline for orchestrating the crawling and knowledge base building process."""
    
    def __init__(self, base_dir: str = config.KNOWLEDGE_BASE_DIR):
        """
        Initialize crawler pipeline.
        
        Args:
            base_dir: Base directory for knowledge base
        """
        self.crawler = WebCrawler()
        self.knowledge_base = KnowledgeBase(base_dir)
    
    def crawl_and_store(self, urls: List[str], max_pages: int = config.CRAWLER_MAX_PAGES) -> int:
        """
        Crawl URLs and store results in knowledge base.
        
        Args:
            urls: List of URLs to crawl
            max_pages: Maximum number of pages to crawl
            
        Returns:
            Number of documents stored
        """
        # Crawl pages
        crawled_pages = self.crawler.crawl_multiple_pages(urls, max_pages)
        
        # Store in knowledge base
        stored_count = 0
        for page in crawled_pages:
            self.knowledge_base.add_document(
                url=page['url'],
                title=page['title'],
                content=page['content'],
                cleaned_content=page['cleaned_content']
            )
            stored_count += 1
        
        return stored_count
    
    def crawl_site_and_store(self, base_url: str, max_pages: int = config.CRAWLER_MAX_PAGES) -> int:
        """
        Crawl an entire website and store results in knowledge base.
        
        Args:
            base_url: Base URL of the site
            max_pages: Maximum number of pages to crawl
            
        Returns:
            Number of documents stored
        """
        # Crawl site
        crawled_pages = self.crawler.crawl_site(base_url, max_pages)
        
        # Store in knowledge base
        stored_count = 0
        for page in crawled_pages:
            self.knowledge_base.add_document(
                url=page['url'],
                title=page['title'],
                content=page['content'],
                cleaned_content=page['cleaned_content']
            )
            stored_count += 1
        
        return stored_count
    
    def get_knowledge_base(self) -> KnowledgeBase:
        """
        Get the knowledge base instance.
        
        Returns:
            KnowledgeBase instance
        """
        return self.knowledge_base
    
    def close(self):
        """Close the crawler pipeline."""
        self.crawler.close()


def create_sample_knowledge_base():
    """
    Create a sample knowledge base with some predefined content for testing.
    This is useful for initial testing without crawling real websites.
    """
    pipeline = CrawlerPipeline()
    kb = pipeline.knowledge_base
    
    # Sample documents for testing
    sample_docs = [
        {
            'url': 'sample://artificial_intelligence',
            'title': 'Artificial Intelligence',
            'content': 'Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.'
        },
        {
            'url': 'sample://machine_learning',
            'title': 'Machine Learning',
            'content': 'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Common machine learning algorithms include neural networks, decision trees, support vector machines, and clustering algorithms.'
        },
        {
            'url': 'sample://deep_learning',
            'title': 'Deep Learning',
            'content': 'Deep learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to model complex patterns in large datasets. It has revolutionized fields such as computer vision, natural language processing, and speech recognition. Popular deep learning architectures include Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers.'
        },
        {
            'url': 'sample://natural_language_processing',
            'title': 'Natural Language Processing',
            'content': 'Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves tasks such as text classification, sentiment analysis, machine translation, question answering, and text generation. Modern NLP systems often use transformer-based models like BERT, GPT, and T5.'
        },
        {
            'url': 'sample://neural_networks',
            'title': 'Neural Networks',
            'content': 'Neural networks are computing systems inspired by biological neural networks in the human brain. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight that is adjusted during training. Neural networks can learn complex non-linear relationships from data and are widely used in pattern recognition, classification, and regression tasks.'
        },
        {
            'url': 'sample://recurrent_neural_networks',
            'title': 'Recurrent Neural Networks',
            'content': 'Recurrent Neural Networks (RNNs) are a type of neural network designed to process sequential data. They maintain an internal state (hidden state) that allows them to remember previous inputs, making them suitable for tasks like language modeling, speech recognition, and time series prediction. However, vanilla RNNs suffer from vanishing and exploding gradient problems.'
        },
        {
            'url': 'sample://lstm',
            'title': 'Long Short-Term Memory',
            'content': 'Long Short-Term Memory (LSTM) is a special kind of RNN capable of learning long-term dependencies. It introduces a memory cell and gating mechanisms (input gate, forget gate, output gate) to control the flow of information. LSTMs address the vanishing gradient problem and are widely used in sequence-to-sequence tasks, speech recognition, and language modeling.'
        },
        {
            'url': 'sample://attention_mechanism',
            'title': 'Attention Mechanism',
            'content': 'Attention mechanism allows neural networks to focus on relevant parts of the input when producing each part of the output. It was introduced to improve sequence-to-sequence models by enabling them to attend to different positions in the source sequence. Self-attention, where each position attends to all other positions, is a key component of transformer models.'
        },
        {
            'url': 'sample://transformers',
            'title': 'Transformers',
            'content': 'Transformers are a neural network architecture that relies entirely on self-attention mechanisms, without recurrence or convolution. They can process entire sequences in parallel, making them highly efficient for training. Transformers have achieved state-of-the-art results in NLP tasks and are the foundation of models like BERT, GPT, and T5.'
        },
        {
            'url': 'sample://reinforcement_learning',
            'title': 'Reinforcement Learning',
            'content': 'Reinforcement learning is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative reward. It has been successfully applied to game playing, robotics, and autonomous systems. Key algorithms include Q-learning, policy gradients, and actor-critic methods.'
        }
    ]
    
    # Add sample documents to knowledge base
    for doc in sample_docs:
        cleaned_content = kb.text_cleaner.normalize_text(doc['content'])
        kb.add_document(
            url=doc['url'],
            title=doc['title'],
            content=doc['content'],
            cleaned_content=cleaned_content
        )
    
    print(f"Created sample knowledge base with {len(sample_docs)} documents")
    print(f"Statistics: {kb.get_statistics()}")
    
    pipeline.close()
    
    return kb


if __name__ == '__main__':
    # Create sample knowledge base for testing
    kb = create_sample_knowledge_base()
    
    # Test knowledge base operations
    print("\n--- Knowledge Base Test ---")
    index = kb.get_index()
    print(f"Index: {index}")
    
    # Search by keyword
    print("\n--- Search Test ---")
    results = kb.search_by_keyword('neural')
    print(f"Found {len(results)} documents matching 'neural'")
    for result in results[:3]:
        print(f"- {result['title']}: {result['content'][:100]}...")
    
    # Get text corpus
    print("\n--- Corpus Test ---")
    corpus = kb.get_text_corpus()
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Total words: {sum(len(doc.split()) for doc in corpus)}")