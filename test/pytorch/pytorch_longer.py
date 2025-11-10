#!/usr/bin/env python3
"""
Longer-running PyTorch script for profiling testing.
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
    print("\n--- Running for 10 seconds ---")

    # Simple neural network
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    start_time = time.time()
    iteration = 0

    while time.time() - start_time < 10.0:
        # Forward pass
        batch = torch.randn(128, 1000, device=device)
        output = model(batch)

        if iteration % 100 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Iteration {iteration}")

        iteration += 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    print(f"\n--- Complete ---")
    print(f"Total iterations: {iteration}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per iteration: {total_time*1000/iteration:.2f}ms")


if __name__ == "__main__":
    main()
