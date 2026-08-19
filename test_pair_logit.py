#!/usr/bin/env python3
"""Test script for pair logit loss implementation."""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from bin.tabpack.tabpack import form_pairs


def test_form_pairs_basic():
    """Test basic pair formation with contiguous keys."""
    # Example: key with indices [8, 9, 10] and labels [0, 1, 0]
    # Should form pairs (8,9) and (9,10)
    # Pair (8,9): pos=9, neg=8
    # Pair (9,10): pos=9, neg=10
    
    keys = np.array([1, 1, 1, 2, 2, 3, 3, 3])
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    
    pos_indices, neg_indices = form_pairs(keys, labels)
    
    print("Test 1: Basic pair formation")
    print(f"Keys: {keys}")
    print(f"Labels: {labels}")
    print(f"Positive indices: {pos_indices}")
    print(f"Negative indices: {neg_indices}")
    
    # Expected pairs:
    # Key 1: (0,1) -> pos=1, neg=0; (1,2) -> pos=1, neg=2
    # Key 2: (3,4) -> pos=3, neg=4
    # Key 3: (5,6) -> pos=6, neg=5; (6,7) -> pos=6, neg=7
    
    expected_pos = np.array([1, 1, 3, 6, 6])
    expected_neg = np.array([0, 2, 4, 5, 7])
    
    assert np.array_equal(pos_indices, expected_pos), f"Expected {expected_pos}, got {pos_indices}"
    assert np.array_equal(neg_indices, expected_neg), f"Expected {expected_neg}, got {neg_indices}"
    print("✓ Test 1 passed\n")


def test_form_pairs_single_object():
    """Test that keys with single object are skipped."""
    keys = np.array([1, 2, 2, 3])
    labels = np.array([1, 0, 1, 0])
    
    pos_indices, neg_indices = form_pairs(keys, labels)
    
    print("Test 2: Single object keys skipped")
    print(f"Keys: {keys}")
    print(f"Labels: {labels}")
    print(f"Positive indices: {pos_indices}")
    print(f"Negative indices: {neg_indices}")
    
    # Only key 2 should form a pair: (1,2) -> pos=2, neg=1
    expected_pos = np.array([2])
    expected_neg = np.array([1])
    
    assert np.array_equal(pos_indices, expected_pos), f"Expected {expected_pos}, got {pos_indices}"
    assert np.array_equal(neg_indices, expected_neg), f"Expected {expected_neg}, got {neg_indices}"
    print("✓ Test 2 passed\n")


def test_form_pairs_same_labels():
    """Test that pairs with same labels are skipped."""
    keys = np.array([1, 1, 2, 2])
    labels = np.array([1, 1, 0, 0])
    
    pos_indices, neg_indices = form_pairs(keys, labels)
    
    print("Test 3: Same label pairs skipped")
    print(f"Keys: {keys}")
    print(f"Labels: {labels}")
    print(f"Positive indices: {pos_indices}")
    print(f"Negative indices: {neg_indices}")
    
    # No valid pairs should be formed
    assert len(pos_indices) == 0, f"Expected empty array, got {pos_indices}"
    assert len(neg_indices) == 0, f"Expected empty array, got {neg_indices}"
    print("✓ Test 3 passed\n")


def test_form_pairs_large_dataset():
    """Test performance on large dataset."""
    print("Test 4: Large dataset performance")
    
    # Create a large dataset with 1M objects
    n_objects = 1_000_000
    n_keys = 500_000  # Average 2 objects per key
    
    # Generate keys (contiguous)
    keys = np.repeat(np.arange(n_keys), 2)
    
    # Generate labels (alternating 0, 1 for each key)
    labels = np.tile([0, 1], n_keys)
    
    print(f"Dataset size: {n_objects} objects, {n_keys} keys")
    
    import time
    start = time.time()
    pos_indices, neg_indices = form_pairs(keys, labels)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.3f} seconds")
    print(f"Pairs formed: {len(pos_indices)}")
    
    # Each key should form 1 pair
    assert len(pos_indices) == n_keys, f"Expected {n_keys} pairs, got {len(pos_indices)}"
    print("✓ Test 4 passed\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing form_pairs() function")
    print("=" * 60 + "\n")
    
    test_form_pairs_basic()
    test_form_pairs_single_object()
    test_form_pairs_same_labels()
    test_form_pairs_large_dataset()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
