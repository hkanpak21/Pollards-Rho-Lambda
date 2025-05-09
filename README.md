# DLP Algorithm Demonstrator

This project implements and demonstrates three algorithms for solving the Discrete Logarithm Problem (DLP):

1. **Baby-Step Giant-Step (BSGS)** - A deterministic algorithm with space complexity O(√N)
2. **Pollard's Rho** - A probabilistic algorithm with constant space complexity
3. **Pollard's Lambda (Kangaroo)** - A probabilistic algorithm designed for bounded discrete logarithms

## Purpose

The program serves as an educational tool to compare and visualize how these three algorithms approach the same DLP problem. It highlights their operational differences, computational steps, and resource usage (especially space complexity).

## Usage

Run the program with:

```
python dlp_algorithms.py
```

You'll be prompted to enter:
- Prime modulus p (the size of the group ℤₚ*)
- Generator g
- Target element h (to find k where g^k ≡ h (mod p))
- Order N of the generator (optional, defaults to p-1)

The program will demonstrate each algorithm's step-by-step execution and provide a comparison of their performance metrics.

## Example

For a simple example, try:
- p = 101 (a small prime)
- g = 2 (a primitive root modulo 101)
- h = 30 (to find k where 2^k ≡ 30 (mod 101))

## Prerequisites

- Python 3.x
- Standard libraries only (no external dependencies)