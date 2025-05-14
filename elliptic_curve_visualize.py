import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from elliptic_curve_plot import EllipticCurve
from typing import List, Tuple, Dict, Optional, Set
import random
import colorsys
import os


def get_random_colors(n: int) -> List[Tuple[float, float, float]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        # Use HSV color space to get evenly spaced colors
        hue = i / n
        saturation = 0.5 + random.uniform(0, 0.5)
        value = 0.5 + random.uniform(0, 0.5)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


def visualize_subgroups(curve: EllipticCurve):
    """
    Visualizes the subgroups of an elliptic curve. Each subgroup is colored differently.
    """
    points = curve.find_points()
    
    # Find generators and their orders
    generators_and_orders = []
    processed_points = set()
    
    # For each point, find the subgroup it generates
    for P in points:
        if P not in processed_points:
            # Find the order of this point
            subgroup = []
            current = P
            order = 1
            
            # Add P, 2P, 3P, ... until we reach the identity
            while current is not None:
                subgroup.append(current)
                processed_points.add(current)
                current = curve.add_points(current, P)
                order += 1
                if current in subgroup:  # We've cycled back
                    break
            
            # Adjust the order since we added one extra for the None check
            order = len(subgroup)
            generators_and_orders.append((P, order, subgroup))
    
    # Sort by order (descending)
    generators_and_orders.sort(key=lambda x: x[1], reverse=True)
    
    # Color each subgroup differently
    colors = get_random_colors(len(generators_and_orders))
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot each subgroup with its own color
    for i, (generator, order, subgroup) in enumerate(generators_and_orders):
        x_coords = [p[0] for p in subgroup]
        y_coords = [p[1] for p in subgroup]
        
        # Label the generator point
        plt.scatter(x_coords, y_coords, color=colors[i], label=f'Generator ({generator[0]}, {generator[1]}) - Order {order}', alpha=0.7)
    
    # Plot settings
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Set plot title and labels
    plt.title(f"Subgroups of Elliptic Curve $y^2 = x^3 + {curve.a}x + {curve.b}$ over $\mathbb{{F}}_{{{curve.p}}}$")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Show legend
    plt.legend(loc='upper right')
    
    # Show the plot
    plt.show()


def visualize_scalar_multiplication(curve: EllipticCurve, base_point: Tuple[int, int], max_scalar: int = 10):
    """
    Visualizes the scalar multiplication sequence kP for k = 1 to max_scalar.
    """
    if not curve.is_on_curve(base_point):
        raise ValueError("Base point must be on the curve")
    
    # Calculate points kP for k = 1 to max_scalar
    points = [base_point]
    for k in range(2, max_scalar + 1):
        next_point = curve.scalar_multiply(k, base_point)
        if next_point is None:
            break
        points.append(next_point)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot all points on the curve as small dots
    all_curve_points = curve.find_points()
    all_x = [p[0] for p in all_curve_points]
    all_y = [p[1] for p in all_curve_points]
    plt.scatter(all_x, all_y, color='lightgray', alpha=0.3, s=30)
    
    # Plot the scalar multiplication sequence
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], color='red', alpha=0.8, s=100)
        plt.text(point[0] + 0.5, point[1] + 0.5, f"{i+1}P", fontsize=12)
    
    # Connect the points with arrows to show the sequence
    for i in range(len(points) - 1):
        plt.arrow(points[i][0], points[i][1], 
                  points[i+1][0] - points[i][0], points[i+1][1] - points[i][1],
                  head_width=0.5, head_length=0.7, fc='blue', ec='blue', alpha=0.6)
    
    # Plot settings
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Set plot title and labels
    plt.title(f"Scalar Multiplication Sequence for P = {base_point} on $y^2 = x^3 + {curve.a}x + {curve.b}$ over $\mathbb{{F}}_{{{curve.p}}}$")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Set equal aspect ratio and limits
    plt.axis('equal')
    
    # Show the plot
    plt.show()


def visualize_pollard_rho(curve: EllipticCurve, generator: Tuple[int, int], target: Tuple[int, int], 
                          max_iterations: int = 50, save_image: bool = True, show_plot: bool = True):
    """
    Visualizes the Pollard's Rho algorithm on an elliptic curve.
    
    Args:
        curve: The elliptic curve
        generator: The base point generator
        target: The target point (we want to find k such that k*generator = target)
        max_iterations: Maximum number of iterations for the algorithm
        save_image: Whether to save the plot as a PNG file
        show_plot: Whether to display the plot interactively
    """
    if not curve.is_on_curve(generator) or not curve.is_on_curve(target):
        raise ValueError("Both generator and target points must be on the curve")
    
    # Define partition function (similar to the one in dlp_algorithms.py)
    def partition(point: Tuple[int, int]) -> int:
        # Use the x-coordinate mod 3 to determine the partition
        return point[0] % 3
    
    # Define the iteration function
    def f(point: Tuple[int, int], a: int, b: int) -> Tuple[Tuple[int, int], int, int]:
        part = partition(point)
        
        if part == 0:  # S0
            # x' = x+x, a' = 2a, b' = 2b
            new_point = curve.add_points(point, point)
            new_a = (2 * a) % curve.p
            new_b = (2 * b) % curve.p
        elif part == 1:  # S1
            # x' = x+g, a' = a+1, b' = b
            new_point = curve.add_points(point, generator)
            new_a = (a + 1) % curve.p
            new_b = b
        else:  # S2
            # x' = x+h, a' = a, b' = b+1
            new_point = curve.add_points(point, target)
            new_a = a
            new_b = (b + 1) % curve.p
            
        return new_point, new_a, new_b
    
    # Initialize
    x_t, a_t, b_t = generator, 1, 0  # Tortoise
    x_h, a_h, b_h = generator, 1, 0  # Hare
    
    tortoise_path = [(x_t, a_t, b_t)]
    hare_path = [(x_h, a_h, b_h)]
    
    # Run Pollard's Rho algorithm
    for _ in range(max_iterations):
        # Tortoise moves one step
        x_t, a_t, b_t = f(x_t, a_t, b_t)
        tortoise_path.append((x_t, a_t, b_t))
        
        # Hare moves two steps
        x_h, a_h, b_h = f(x_h, a_h, b_h)
        x_h, a_h, b_h = f(x_h, a_h, b_h)
        hare_path.append((x_h, a_h, b_h))
        
        # Check for collision
        if x_t == x_h:
            break
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot all points on the curve as small dots
    all_curve_points = curve.find_points()
    all_x = [p[0] for p in all_curve_points]
    all_y = [p[1] for p in all_curve_points]
    plt.scatter(all_x, all_y, color='lightgray', alpha=0.3, s=30)
    
    # Plot the generator and target points
    plt.scatter(generator[0], generator[1], color='green', alpha=1.0, s=100, label='Generator')
    plt.scatter(target[0], target[1], color='red', alpha=1.0, s=100, label='Target')
    
    # Plot the tortoise path
    tortoise_x = [p[0][0] for p in tortoise_path]
    tortoise_y = [p[0][1] for p in tortoise_path]
    plt.plot(tortoise_x, tortoise_y, 'b-', alpha=0.5, marker='o', markersize=5, label='Tortoise Path')
    
    # Plot the hare path
    hare_x = [p[0][0] for p in hare_path]
    hare_y = [p[0][1] for p in hare_path]
    plt.plot(hare_x, hare_y, 'r-', alpha=0.5, marker='x', markersize=5, label='Hare Path')
    
    # Mark the collision point if found
    if x_t == x_h:
        plt.scatter(x_t[0], x_t[1], color='purple', alpha=1.0, s=150, label='Collision')
    
    # Plot settings
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Set plot title and labels
    plt.title(f"Pollard's Rho Algorithm on Elliptic Curve $y^2 = x^3 + {curve.a}x + {curve.b}$ over $\mathbb{{F}}_{{{curve.p}}}$")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Show legend
    plt.legend(loc='upper right')
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Save the plot if requested
    if save_image:
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        # Save plot as PNG file
        filename = f'images/pollards_rho_a{curve.a}_b{curve.b}_p{curve.p}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved Pollard's Rho plot to {filename}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_baby_step_giant_step(curve: EllipticCurve, generator: Tuple[int, int], target: Tuple[int, int], 
                                  save_image: bool = True, show_plot: bool = True):
    """
    Visualizes the Baby-Step Giant-Step algorithm on an elliptic curve.
    
    Args:
        curve: The elliptic curve
        generator: The base point generator
        target: The target point (we want to find k such that k*generator = target)
        save_image: Whether to save the plot as a PNG file
        show_plot: Whether to display the plot interactively
    """
    if not curve.is_on_curve(generator) or not curve.is_on_curve(target):
        raise ValueError("Both generator and target points must be on the curve")
    
    # Compute order of the generator (in this case, we'll use a small curve so we can compute it directly)
    order = 1
    current = generator
    while current is not None:
        current = curve.add_points(current, generator)
        if current == generator:
            break
        order += 1
    
    # Calculate m = ceil(sqrt(N))
    m = int(np.ceil(np.sqrt(order)))
    
    # Baby steps: compute g^j for j = 0, 1, ..., m-1
    baby_steps = {}
    baby_points = [None]  # Point at infinity (for j=0)
    current = generator
    for j in range(1, m):
        baby_steps[current] = j
        baby_points.append(current)
        current = curve.add_points(current, generator)
    
    # Compute g^(-m)
    g_inv_m = None
    temp = generator
    for _ in range(m):
        # Add (n-1) copies of generator to get -generator
        # Since g^n = identity, g^(n-1) = g^(-1)
        g_inv_m = curve.add_points(g_inv_m, temp)
    
    # Giant steps: compute h * (g^(-m))^i for i = 0, 1, ..., m-1
    giant_steps = []
    current = target
    for i in range(m):
        giant_steps.append(current)
        if current in baby_steps:
            j = baby_steps[current]
            k = (i * m + j) % order
            break
        current = curve.add_points(current, g_inv_m)
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Plot all points on the curve as small dots
    all_curve_points = curve.find_points()
    all_x = [p[0] for p in all_curve_points]
    all_y = [p[1] for p in all_curve_points]
    plt.scatter(all_x, all_y, color='lightgray', alpha=0.3, s=30)
    
    # Plot baby steps
    for j, point in enumerate(baby_points):
        if point is not None:  # Skip point at infinity
            plt.scatter(point[0], point[1], color='blue', alpha=0.7, s=80)
            plt.text(point[0] + 0.3, point[1] + 0.3, f"g^{j}", fontsize=10)
    
    # Plot giant steps
    for i, point in enumerate(giant_steps):
        plt.scatter(point[0], point[1], color='red', alpha=0.7, s=80)
        plt.text(point[0] - 1.5, point[1] - 0.5, f"h·(g^-m)^{i}", fontsize=10)
    
    # Highlight the generator and target points
    plt.scatter(generator[0], generator[1], color='green', alpha=1.0, s=120, label='Generator g')
    plt.scatter(target[0], target[1], color='purple', alpha=1.0, s=120, label='Target h')
    
    # Draw an arrow from the matching giant step to the matching baby step
    if 'j' in locals():
        match_giant = giant_steps[i]
        match_baby = baby_points[j]
        plt.arrow(match_giant[0], match_giant[1], 
                 match_baby[0] - match_giant[0], match_baby[1] - match_giant[1],
                 head_width=0.5, head_length=0.7, fc='green', ec='green', alpha=0.9)
        
        plt.scatter(match_giant[0], match_giant[1], color='yellow', alpha=1.0, s=150, 
                   label=f'Match: k = {i}·{m} + {j} = {k}')
    
    # Plot settings
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Set plot title and labels
    plt.title(f"Baby-Step Giant-Step on Elliptic Curve $y^2 = x^3 + {curve.a}x + {curve.b}$ over $\mathbb{{F}}_{{{curve.p}}}$")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Show legend
    plt.legend(loc='upper right')
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Save the plot if requested
    if save_image:
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        # Save plot as PNG file
        filename = f'images/bsgs_a{curve.a}_b{curve.b}_p{curve.p}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved Baby-Step Giant-Step plot to {filename}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_pollards_lambda(curve: EllipticCurve, generator: Tuple[int, int], target: Tuple[int, int], 
                             a: int = 0, b: int = 20, save_image: bool = True, show_plot: bool = True):
    """
    Visualizes the Pollard's Lambda (Kangaroo) algorithm on an elliptic curve.
    
    Args:
        curve: The elliptic curve
        generator: The base point generator
        target: The target point (we want to find k such that k*generator = target)
        a: Lower bound of the discrete log (default: 0)
        b: Upper bound of the discrete log (default: 20)
        save_image: Whether to save the plot as a PNG file
        show_plot: Whether to display the plot interactively
    """
    if not curve.is_on_curve(generator) or not curve.is_on_curve(target):
        raise ValueError("Both generator and target points must be on the curve")
    
    # Define a distance function
    def distance(point: Tuple[int, int]) -> int:
        """Simple distance function based on the least significant bits of x-coordinate"""
        if point is None:
            return 0
        return 1 << (point[0] % 4)  # 2^(x mod 4) - small jumps of sizes 1, 2, 4, 8
    
    # Define a function to check if a point is distinguished
    def is_distinguished(point: Tuple[int, int]) -> bool:
        """A point is distinguished if its x-coordinate mod 8 is 0"""
        if point is None:
            return False
        return point[0] % 8 == 0
    
    # Create tame kangaroo path
    tame_path = []
    tame_pos = curve.scalar_multiply(b, generator)  # Start at upper bound
    tame_jumps = 0
    tame_distinguished = set()
    
    # Run the tame kangaroo for a while to collect distinguished points
    max_tame_steps = 2 * (b - a)  # Reasonable number of steps
    for _ in range(max_tame_steps):
        if tame_pos is None:
            break
            
        tame_path.append((tame_pos, b - tame_jumps))
        
        if is_distinguished(tame_pos):
            tame_distinguished.add((tame_pos, b - tame_jumps))
            
        # Take a jump
        d = distance(tame_pos)
        jump_point = curve.scalar_multiply(d, generator)
        tame_pos = curve.add_points(tame_pos, jump_point)
        tame_jumps += d
    
    # Create wild kangaroo path
    wild_path = []
    wild_pos = target  # Start at the target point
    wild_jumps = 0
    
    # Run the wild kangaroo until it finds a distinguished point or exceeds max steps
    max_wild_steps = 3 * (b - a)  # More steps for wild kangaroo
    found_match = False
    
    for _ in range(max_wild_steps):
        if wild_pos is None:
            break
            
        wild_path.append((wild_pos, wild_jumps))
        
        # Check if this is a distinguished point seen by tame kangaroo
        if is_distinguished(wild_pos):
            for dp, tj in tame_distinguished:
                if dp == wild_pos:
                    # Found a match of distinguished points
                    k = tj - wild_jumps
                    if a <= k <= b:  # Verify it's in the expected range
                        found_match = True
                        match_point = wild_pos
                        solution = k
                        break
            
        if found_match:
            break
            
        # Take a jump
        d = distance(wild_pos)
        jump_point = curve.scalar_multiply(d, generator)
        wild_pos = curve.add_points(wild_pos, jump_point)
        wild_jumps += d
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Plot all points on the curve as small dots
    all_curve_points = curve.find_points()
    all_x = [p[0] for p in all_curve_points]
    all_y = [p[1] for p in all_curve_points]
    plt.scatter(all_x, all_y, color='lightgray', alpha=0.3, s=30)
    
    # Plot the generator and target points
    plt.scatter(generator[0], generator[1], color='green', alpha=1.0, s=100, label='Generator')
    plt.scatter(target[0], target[1], color='red', alpha=1.0, s=100, label='Target')
    
    # Plot the tame kangaroo path
    tame_x = [p[0][0] for p in tame_path]
    tame_y = [p[0][1] for p in tame_path]
    plt.plot(tame_x, tame_y, 'b-', alpha=0.5, marker='o', markersize=5, label='Tame Kangaroo Path')
    
    # Plot the wild kangaroo path
    wild_x = [p[0][0] for p in wild_path]
    wild_y = [p[0][1] for p in wild_path]
    plt.plot(wild_x, wild_y, 'r-', alpha=0.5, marker='x', markersize=5, label='Wild Kangaroo Path')
    
    # Mark distinguished points
    for point, _ in tame_distinguished:
        plt.scatter(point[0], point[1], color='blue', alpha=1.0, s=150, marker='*')
    
    # Mark the collision point if found
    if found_match:
        plt.scatter(match_point[0], match_point[1], color='purple', alpha=1.0, s=200, 
                   label=f'Match: k = {solution}')
    
    # Plot settings
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Set plot title and labels
    plt.title(f"Pollard's Lambda Algorithm on Elliptic Curve $y^2 = x^3 + {curve.a}x + {curve.b}$ over $\mathbb{{F}}_{{{curve.p}}}$")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Add a note about distinguished points
    plt.figtext(0.5, 0.01, "⭐ Distinguished points (x mod 8 = 0)", ha="center", fontsize=10)
    
    # Show legend
    plt.legend(loc='upper right')
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Save the plot if requested
    if save_image:
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        # Save plot as PNG file
        filename = f'images/lambda_a{curve.a}_b{curve.b}_p{curve.p}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved Pollard's Lambda plot to {filename}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Use small prime to have a manageable number of points
    p = 17
    a = 2
    b = 2
    curve = EllipticCurve(a, b, p)
    
    # Find points on the curve
    points = curve.find_points()
    print(f"Found {len(points)} points on the curve y^2 = x^3 + {a}x + {b} (mod {p})")
    
    # Visualize the subgroups
    visualize_subgroups(curve)
    
    # Choose a base point and visualize scalar multiplication
    if len(points) > 0:
        # Find a point with high order (preferably a generator)
        base_point = points[0]
        for point in points:
            # Compute the order of the point
            current = point
            order = 1
            while True:
                current = curve.add_points(current, point)
                if current is None or current == point:
                    break
                order += 1
            if order > 4:  # Choose a point with reasonably high order
                base_point = point
                break
                
        print(f"Using base point {base_point} for scalar multiplication visualization")
        visualize_scalar_multiplication(curve, base_point, max_scalar=10)
        
        # Choose a target point for algorithm visualizations
        # We'll use a scalar multiple of the base point
        k = 5  # The discrete log we're pretending to find
        target_point = curve.scalar_multiply(k, base_point)
        
        if target_point is not None:
            print(f"Using target point {target_point} for algorithm visualizations (k={k})")
            
            # Visualize Pollard's Rho
            visualize_pollard_rho(curve, base_point, target_point, max_iterations=20)
            
            # Visualize Baby-Step Giant-Step
            visualize_baby_step_giant_step(curve, base_point, target_point)
            
            # Visualize Pollard's Lambda, assuming k is between 0 and 10
            visualize_pollards_lambda(curve, base_point, target_point, a=0, b=10) 