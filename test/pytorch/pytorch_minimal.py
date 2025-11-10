#!/usr/bin/env python3
"""
Minimal PyTorch script demonstrating basic tensor operations and GPU usage.
"""

import torch
import torch.nn as nn
import time


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    # Create tensors
    print("\n--- Tensor Operations ---")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    # Matrix multiplication
    start = time.time()
    z = torch.matmul(x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Matrix multiplication (1000x1000): {elapsed*1000:.2f}ms")

    # Simple neural network
    print("\n--- Neural Network ---")
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    # Forward pass
    batch = torch.randn(128, 1000, device=device)
    start = time.time()
    output = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Forward pass (batch=128): {elapsed*1000:.2f}ms")
    print(f"Output shape: {output.shape}")

    print("\n--- Complete ---")


if __name__ == "__main__":
    main()
