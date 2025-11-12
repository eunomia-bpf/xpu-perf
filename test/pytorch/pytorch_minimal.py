#!/usr/bin/env python3
"""
Minimal PyTorch script demonstrating various tensor operations and GPU usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def matrix_operations(device):
    """Test various matrix operations"""
    print("\n--- Matrix Operations ---")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    # Matrix multiplication
    start = time.time()
    z = torch.matmul(x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Matrix multiplication (1000x1000): {elapsed*1000:.2f}ms")

    # Element-wise operations
    start = time.time()
    result = x * y + torch.sin(x) - torch.cos(y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Element-wise ops (mul, sin, cos): {elapsed*1000:.2f}ms")

    # Reduction operations
    start = time.time()
    mean = x.mean()
    std = x.std()
    max_val = x.max()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Reduction ops (mean, std, max): {elapsed*1000:.2f}ms")


def convolution_operations(device):
    """Test convolution operations"""
    print("\n--- Convolution Operations ---")

    # 2D convolution
    input_tensor = torch.randn(8, 3, 224, 224, device=device)
    conv_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)

    start = time.time()
    output = conv_layer(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Conv2d (8x3x224x224 -> 8x64x224x224): {elapsed*1000:.2f}ms")

    # Batch normalization
    bn_layer = nn.BatchNorm2d(64).to(device)
    start = time.time()
    output = bn_layer(output)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"BatchNorm2d (8x64x224x224): {elapsed*1000:.2f}ms")

    # Max pooling
    start = time.time()
    output = F.max_pool2d(output, kernel_size=2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"MaxPool2d (8x64x224x224 -> 8x64x112x112): {elapsed*1000:.2f}ms")


def attention_operations(device):
    """Test attention mechanism operations"""
    print("\n--- Attention Operations ---")

    batch_size = 32
    seq_len = 128
    hidden_dim = 512

    # Multi-head attention inputs
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    value = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Scaled dot-product attention
    start = time.time()
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_dim ** 0.5)
    attention_weights = F.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Scaled dot-product attention: {elapsed*1000:.2f}ms")


def neural_network_forward(device):
    """Test neural network forward pass"""
    print("\n--- Neural Network Forward Pass ---")

    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)
    ).to(device)

    batch = torch.randn(128, 1000, device=device)
    start = time.time()
    output = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Forward pass (batch=128): {elapsed*1000:.2f}ms")
    print(f"Output shape: {output.shape}")


def backward_pass(device):
    """Test backward pass and gradient computation"""
    print("\n--- Backward Pass ---")

    model = nn.Linear(512, 256).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    x = torch.randn(64, 512, device=device)
    target = torch.randn(64, 256, device=device)

    start = time.time()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Forward + Backward + Update: {elapsed*1000:.2f}ms")
    print(f"Loss: {loss.item():.4f}")


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    # Run various operations
    matrix_operations(device)
    convolution_operations(device)
    attention_operations(device)
    neural_network_forward(device)
    backward_pass(device)

    print("\n--- Complete ---")


if __name__ == "__main__":
    main()
