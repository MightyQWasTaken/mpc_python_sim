#!/usr/bin/env python3
"""
Universal data viewer script to load and display various file formats.
Supports: .npy, .pt, .npz
Usage: python view_K_matrix.py <filepath>
"""

import sys
import numpy as np
import torch
import os

def load_npy(filepath):
    """Load and return a .npy file."""
    return np.load(filepath)

def load_pt(filepath):
    """Load and return a .pt file (PyTorch tensor/checkpoint)."""
    data = torch.load(filepath, map_location='cpu')
    return data

def load_npz(filepath):
    """Load and return a .npz file (zipped numpy arrays)."""
    data = np.load(filepath)
    return data

def print_data(data, name="Data"):
    """Pretty print data based on its type."""
    if isinstance(data, dict):
        print(f"{name} is a dictionary with keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"\n  {key}:")
            print_array_info(value, indent=4)
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        print(f"{name}:")
        print_array_info(data)
    elif isinstance(data, (int, float, str)):
        print(f"{name}: {data}")
    else:
        print(f"{name}: {type(data)}")
        print(data)

def print_array_info(arr, indent=0):
    """Print information about an array/tensor."""
    prefix = " " * indent
    
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().detach().numpy()
    
    if isinstance(arr, np.ndarray):
        print(f"{prefix}Shape: {arr.shape}")
        print(f"{prefix}Dtype: {arr.dtype}")
        print(f"{prefix}Min: {arr.min():.6f}, Max: {arr.max():.6f}, Mean: {arr.mean():.6f}")
        print(f"{prefix}Values:\n{arr}")
    else:
        print(f"{prefix}{arr}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_K_matrix.py <filepath>")
        print("Supported formats: .npy, .pt, .npz")
        print("\nExample:")
        print("  python view_K_matrix.py data/model/K_matrix.npy")
        print("  python view_K_matrix.py data/x_train.pt")
        exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        exit(1)
    
    _, ext = os.path.splitext(filepath)
    
    try:
        if ext == '.npy':
            data = load_npy(filepath)
        elif ext == '.pt':
            data = load_pt(filepath)
        elif ext == '.npz':
            data = load_npz(filepath)
        else:
            print(f"Error: Unsupported file format: {ext}")
            print("Supported formats: .npy, .pt, .npz")
            exit(1)
        
        print_data(data, os.path.basename(filepath))
    
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)

if __name__ == "__main__":
    main()
