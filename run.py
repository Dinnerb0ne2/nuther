# -*- coding: utf-8 -*-
"""
Quick start script for Nuther (Retro Memory LSTM) framework.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import main

if __name__ == '__main__':
    main()