import os
import sys
import pytest

# Ensure the parent directory (code/) is in the path so tests can find layers/models/retrieval
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
