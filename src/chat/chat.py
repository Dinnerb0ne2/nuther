# -*- coding: utf-8 -*-
"""
Chat implementation for Nuther (Retro Memory LSTM) neural network framework.
This module implements interactive dialogue functionality with input encoding,
model inference, and output decoding.
"""

import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import config
from src.vocab import Vocabulary
from src.model import NutherModel


class ChatSession:
    """
    Chat session for managing a single conversation.
    Tracks dialogue history and context for the current session.
    """
    
    def __init__(self, session_id: str):
        """
        Initialize chat session.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.start_time = time.time()
        self.message_count = 0
        self.dialogue_history: List[Dict] = []
        self.metadata: Dict = {}
    
    def add_message(self, role: str, content: str, 
                    extra_info: Optional[Dict] = None):
        """
        Add a message to the dialogue history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            extra_info: Additional information
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': time.time(),
            'message_id': self.message_count
        }
        
        if extra_info:
            message.update(extra_info)
        
        self.dialogue_history.append(message)
        self.message_count += 1
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get dialogue history.
        
        Args:
            last_n: Number of recent messages to return (None for all)
            
        Returns:
            List of messages
        """
        if last_n is None:
            return self.dialogue_history.copy()
        else:
            return self.dialogue_history[-last_n:]
    
    def get_duration(self) -> float:
        """
        Get session duration in seconds.
        
        Returns:
            Session duration
        """
        return time.time() - self.start_time
    
    def get_statistics(self) -> Dict:
        """
        Get session statistics.
        
        Returns:
            Statistics dictionary
        """
        user_messages = [m for m in self.dialogue_history if m['role'] == 'user']
        assistant_messages = [m for m in self.dialogue_history if m['role'] == 'assistant']
        
        return {
            'session_id': self.session_id,
            'duration': self.get_duration(),
            'total_messages': self.message_count,
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'turns': len(user_messages)
        }
    
    def clear_history(self):
        """Clear dialogue history."""
        self.dialogue_history.clear()
        self.message_count = 0
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"ChatSession(id={self.session_id}, turns={stats['turns']}, duration={stats['duration']:.1f}s)"


class ChatBot:
    """
    Chat bot for interactive dialogue using Nuther model.
    Provides user-friendly interface for conversations.
    """
    
    def __init__(self, model: NutherModel, max_history: int = 50):
        """
        Initialize chat bot.
        
        Args:
            model: Nuther model instance
            max_history: Maximum number of turns to keep in history
        """
        self.model = model
        self.max_history = max_history
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        self.default_params = {
            'max_length': config.MAX_SEQ_LENGTH,
            'temperature': 0.8
        }
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new chat session.
        
        Args:
            session_id: Session ID (auto-generated if None)
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        session = ChatSession(session_id)
        self.sessions[session_id] = session
        self.current_session_id = session_id
        
        return session_id
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[ChatSession]:
        """
        Get chat session.
        
        Args:
            session_id: Session ID (uses current if None)
            
        Returns:
            ChatSession or None if not found
        """
        if session_id is None:
            session_id = self.current_session_id
        
        return self.sessions.get(session_id)
    
    def switch_session(self, session_id: str) -> bool:
        """
        Switch to a different session.
        
        Args:
            session_id: Session ID to switch to
            
        Returns:
            True if successful, False if session not found
        """
        if session_id in self.sessions:
            self.current_session_id = session_id
            return True
        return False
    
    def chat(self, user_input: str, 
             session_id: Optional[str] = None,
             max_length: Optional[int] = None,
             temperature: Optional[float] = None,
             use_memory: bool = True) -> Dict:
        """
        Process user input and generate response.
        
        Args:
            user_input: User input text
            session_id: Session ID (uses current if None)
            max_length: Maximum output length
            temperature: Sampling temperature
            use_memory: Whether to use memory retrieval
            
        Returns:
            Response dictionary
        """
        # Get or create session
        session = self.get_session(session_id)
        if session is None:
            session_id = self.create_session(session_id)
            session = self.sessions[session_id]
        
        # Set parameters
        max_length = max_length or self.default_params['max_length']
        temperature = temperature or self.default_params['temperature']
        
        # Generate response
        if use_memory:
            response_text, memory_context = self.model.generate_with_memory(
                user_input, max_length=max_length, temperature=temperature
            )
        else:
            response_text = self.model.generate(
                user_input, max_length=max_length, temperature=temperature
            )
            memory_context = None
        
        # Get memory stats
        memory_stats = None
        if self.model.get_memory_bank():
            memory_stats = self.model.get_memory_bank().get_statistics()
        
        # Add to dialogue history
        session.add_message('user', user_input)
        session.add_message('assistant', response_text, {
            'memory_context': memory_context,
            'memory_stats': memory_stats
        })
        
        # Prepare response
        result = {
            'session_id': session.session_id,
            'user_input': user_input,
            'response': response_text,
            'turn_number': session.message_count // 2,
            'timestamp': time.time()
        }
        
        if memory_context:
            result['memory_context'] = memory_context
        
        if memory_stats:
            result['memory_stats'] = memory_stats
        
        return result
    
    def batch_chat(self, inputs: List[str],
                   session_id: Optional[str] = None,
                   max_length: Optional[int] = None,
                   temperature: Optional[float] = None) -> List[Dict]:
        """
        Process multiple inputs in batch.
        
        Args:
            inputs: List of user inputs
            session_id: Session ID
            max_length: Maximum output length
            temperature: Sampling temperature
            
        Returns:
            List of response dictionaries
        """
        responses = []
        for user_input in inputs:
            response = self.chat(
                user_input, session_id, max_length, temperature
            )
            responses.append(response)
        return responses
    
    def continue_conversation(self, max_turns: int = 10,
                             max_length: Optional[int] = None,
                             temperature: Optional[float] = None) -> List[Dict]:
        """
        Continue conversation for multiple turns.
        
        Args:
            max_turns: Maximum number of turns
            max_length: Maximum output length
            temperature: Sampling temperature
            
        Returns:
            List of conversation turns
        """
        turns = []
        
        for _ in range(max_turns):
            try:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                
                response = self.chat(user_input, max_length=max_length, temperature=temperature)
                turns.append(response)
                
                print(f"Bot: {response['response']}")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        return turns
    
    def interactive_chat(self, session_id: Optional[str] = None,
                        max_length: Optional[int] = None,
                        temperature: Optional[float] = None):
        """
        Start interactive chat session.
        
        Args:
            session_id: Session ID
            max_length: Maximum output length
            temperature: Sampling temperature
        """
        print("="*60)
        print("  NUTHER Chat Bot")
        print("  Type 'exit', 'quit', or 'bye' to end the conversation")
        print("  Type 'history' to see conversation history")
        print("  Type 'stats' to see session statistics")
        print("  Type 'clear' to clear conversation history")
        print("="*60)
        
        # Create or get session
        session = self.get_session(session_id)
        if session is None:
            session_id = self.create_session(session_id)
            print(f"\nStarted new session: {session_id}")
        else:
            print(f"\nContinuing session: {session_id}")
        
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'history':
                    self._show_history(session)
                    continue
                
                if user_input.lower() == 'stats':
                    self._show_stats(session)
                    continue
                
                if user_input.lower() == 'clear':
                    session.clear_history()
                    print("Conversation history cleared.")
                    continue
                
                # Process user input
                response = self.chat(
                    user_input, session_id=session_id,
                    max_length=max_length, temperature=temperature
                )
                
                print(f"Bot: {response['response']}")
                
                # Show memory info if available
                if 'memory_context' in response and response['memory_context']:
                    print(f"[Memory: {len(response['memory_context'])} chars retrieved]")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
    
    def _show_history(self, session: ChatSession):
        """
        Show conversation history.
        
        Args:
            session: Chat session
        """
        history = session.get_history()
        
        if not history:
            print("No conversation history yet.")
            return
        
        print("\n" + "="*60)
        print("  Conversation History")
        print("="*60)
        
        for message in history:
            role = message['role'].upper()
            content = message['content']
            print(f"\n{role}: {content}")
        
        print("\n" + "="*60 + "\n")
    
    def _show_stats(self, session: ChatSession):
        """
        Show session statistics.
        
        Args:
            session: Chat session
        """
        stats = session.get_statistics()
        
        print("\n" + "="*60)
        print("  Session Statistics")
        print("="*60)
        print(f"Session ID: {stats['session_id']}")
        print(f"Duration: {stats['duration']:.1f} seconds")
        print(f"Total Messages: {stats['total_messages']}")
        print(f"User Messages: {stats['user_messages']}")
        print(f"Assistant Messages: {stats['assistant_messages']}")
        print(f"Dialogue Turns: {stats['turns']}")
        print("="*60 + "\n")
    
    def get_all_sessions(self) -> List[str]:
        """
        Get all session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session_id == session_id:
                self.current_session_id = None
            return True
        return False
    
    def clear_all_sessions(self):
        """Clear all sessions."""
        self.sessions.clear()
        self.current_session_id = None
    
    def set_default_params(self, max_length: Optional[int] = None,
                          temperature: Optional[float] = None):
        """
        Set default parameters.
        
        Args:
            max_length: Maximum output length
            temperature: Sampling temperature
        """
        if max_length is not None:
            self.default_params['max_length'] = max_length
        if temperature is not None:
            self.default_params['temperature'] = temperature
    
    def get_model(self) -> NutherModel:
        """
        Get the underlying model.
        
        Returns:
            Nuther model instance
        """
        return self.model
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ChatBot(sessions={len(self.sessions)}, current={self.current_session_id})"
