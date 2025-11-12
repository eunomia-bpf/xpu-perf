#!/usr/bin/env python3
"""
ResNet training script for profiling testing.
Trains a ResNet-18 model on synthetic data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
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

    print("\n--- Training ResNet-18 for 10 seconds ---")

    # Create ResNet-18 model
    model = models.resnet18(pretrained=False, num_classes=1000)
    model = model.to(device)
    model.train()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training parameters
    batch_size = 32
    input_size = (3, 224, 224)  # ResNet standard input size

    start_time = time.time()
    iteration = 0

    while time.time() - start_time < 10.0:
        # Generate synthetic batch
        images = torch.randn(batch_size, *input_size, device=device)
        labels = torch.randint(0, 1000, (batch_size,), device=device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Iteration {iteration}, Loss: {loss.item():.4f}")

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
