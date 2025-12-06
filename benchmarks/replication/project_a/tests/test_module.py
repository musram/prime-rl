"""
Tests for module.py
"""

import pytest
from src.module import calculate_sum, multiply


def test_calculate_sum():
    """Test calculate_sum function."""
    assert calculate_sum([1, 2, 3]) == 6
    assert calculate_sum([10, 20, 30]) == 60
    assert calculate_sum([]) == 0
    assert calculate_sum([-1, 1]) == 0


def test_multiply():
    """Test multiply function."""
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0
    assert multiply(-2, 3) == -6

