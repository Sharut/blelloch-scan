# BlellochScan

A fast, gradient-friendly PyTorch module for parallel prefix-scan operations using the Blelloch algorithm. Whether you need cumulative sums, running maximums, or any associative reduction, BlellochScan delivers efficient, backpropagation-compatible scans on GPU or CPU.

## Background

Prefix scans (or parallel scans) form the backbone of many parallel algorithms—powering tasks in graphics, scientific computing, and neural network kernels. The Blelloch algorithm arranges data into a binary tree, performing an **upsweep** (pairwise reduction) to accumulate partial results, then a **downsweep** (distribution) to compute the final scan in O(n) work and O(log n) depth.

## Key Features

- **Inclusive & Exclusive Modes**: Compute both prefix sums and shifted-prefix scans.
- **Custom Combine Function**: Supply any associative operation (sum, max, min, neural merge, etc.).
- **Automatic Power-of-Two Padding**: Transparently pads sequences to the next power of two.
- **Full Autograd Support**: Seamlessly integrates with PyTorch’s backward pass.
- **Minimal Dependencies**: Pure PyTorch implementation with no external requirements.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-org/blelloch-scan.git
cd blelloch-scan
```

Import directly in your project:

```python
from blelloch_scan import BlellochScan
```

## Quick Start

```python
import torch
from blelloch_scan import BlellochScan

# Create a batch of 1D data: shape (B, L, D, N)
X = torch.arange(1, 9).view(1, 8, 1, 1).float().requires_grad_()

# Define an additive combiner
def comb(a, b):
    return a + b

# Inclusive scan
scan = BlellochScan(comb, inclusive=True)
Y = scan(X)
print(Y.flatten())  # [ 1,  3,  6, 10, 15, 21, 28, 36]

# Backward test
test_loss = Y.sum()
test_loss.backward()
print(X.grad.flatten())  # [8, 7, 6, 5, 4, 3, 2, 1]
```
