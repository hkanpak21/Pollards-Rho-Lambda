# Elliptic Curve Visualizations

This directory contains visualizations of elliptic curves and discrete logarithm algorithms running on them.

Each algorithm visualization shows a different approach to solving the discrete logarithm problem (DLP) on the elliptic curve y² = x³ + 3x + 7 over the finite field F₂₉. The problem is to find k such that k × Generator = Target.

## 1. Baby-Step Giant-Step Algorithm

![Baby-Step Giant-Step on Elliptic Curve y² = x³ + 3x + 7 over F₂₉](bsgs_a3_b7_p29.png)

### Elements in the visualization:

- **Light gray dots**: All points on the elliptic curve
- **Green dot**: Generator point (0, 6)
- **Purple dot**: Target point (23, 18), which equals 5 × (0, 6)
- **Blue dots**: Baby steps (g, g², g³, ...) - precomputed points
- **Red dots**: Giant steps (h·(g⁻ᵐ)⁰, h·(g⁻ᵐ)¹, ...) - search points
- **Yellow dot with green arrow**: Match point where a giant step equals a baby step

### How Baby-Step Giant-Step Works:

1. The algorithm uses a space-time tradeoff with O(√n) memory and time complexity.
2. Baby steps: Precompute and store g⁰, g¹, g², ..., gᵐ⁻¹ where m = ⌈√n⌉.
3. Giant steps: Compute h·(g⁻ᵐ)ⁱ for i = 0, 1, 2, ... and check for matches with baby steps.
4. When a match is found: k = i·m + j where i is the giant step index and j is the baby step index.

## 2. Pollard's Rho Algorithm

![Pollard's Rho Algorithm on Elliptic Curve y² = x³ + 3x + 7 over F₂₉](pollards_rho_a3_b7_p29.png)

### Elements in the visualization:

- **Light gray dots**: All points on the elliptic curve
- **Green dot**: Generator point (0, 6)
- **Red dot**: Target point (23, 18), which equals 5 × (0, 6)
- **Blue line with circles**: Tortoise path (moves at normal speed)
- **Red line with X marks**: Hare path (moves at double speed)
- **Purple dot**: Collision point where tortoise and hare meet

### How Pollard's Rho Works:

1. The algorithm uses a pseudorandom walk function to traverse the elliptic curve points.
2. Two pointers (tortoise and hare) move through the points at different speeds.
3. When they collide, we can solve for the discrete logarithm k where k × Generator = Target.
4. This is a space-efficient algorithm with O(1) memory requirements and O(√n) expected time complexity.

## 3. Pollard's Lambda (Kangaroo) Algorithm

![Pollard's Lambda Algorithm on Elliptic Curve y² = x³ + 3x + 7 over F₂₉](lambda_a3_b7_p29.png)

### Elements in the visualization:

- **Light gray dots**: All points on the elliptic curve
- **Green dot**: Generator point (0, 6)
- **Red dot**: Target point (23, 18), which equals 5 × (0, 6)
- **Blue line with circles**: Tame kangaroo path (starts from known position)
- **Red line with X marks**: Wild kangaroo path (starts from target point)
- **Blue stars**: Distinguished points found by the tame kangaroo
- **Purple dot**: Collision point where the kangaroos meet at a distinguished point

### How Pollard's Lambda Works:

1. The algorithm is designed for bounded discrete logarithms where k is known to be in range [a,b].
2. Two kangaroos (tame and wild) jump through the curve using the same pseudorandom jump function.
3. The tame kangaroo starts at g^b and keeps track of distinguished points it encounters.
4. The wild kangaroo starts at h = g^k and follows the same jump pattern.
5. When the wild kangaroo hits a distinguished point previously found by the tame kangaroo, we can determine k.
6. This algorithm has O(√(b-a)) expected time complexity, making it efficient for bounded ranges.

## Generating Visualizations

You can generate these visualizations using the command:

```
python ec_visualizer.py [options]
```

For example, to generate all three algorithm visualizations:

```
python ec_visualizer.py --pollard-rho --bsgs --lambda
```

Or to run a specific algorithm on a different curve:

```
python ec_visualizer.py --a 5 --b 9 --p 47 --bsgs
```

To save images without displaying interactive plots:

```
python ec_visualizer.py --all --save-only
``` 