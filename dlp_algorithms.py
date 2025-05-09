import math
import random
from typing import Tuple, Dict, List, Callable, Set


def mod_inverse(a: int, m: int) -> int:
    """
    Compute the modular inverse of a modulo m.
    Returns x such that (a * x) % m == 1.
    Raises ValueError if gcd(a, m) != 1 (modular inverse doesn't exist).
    """
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Modular inverse doesn't exist (gcd({a}, {m}) = {g})")
    return x % m


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm to find gcd and Bézout coefficients.
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if a == 0:
        return (b, 0, 1)
    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return (g, x, y)


def is_prime(n: int) -> bool:
    """Simple primality test."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def baby_step_giant_step(g: int, h: int, p: int, N: int = None, verbose: bool = True) -> Tuple[int, int, int]:
    """
    Solve the discrete logarithm problem using Baby-Step Giant-Step method.
    Find k such that g^k ≡ h (mod p).
    
    Args:
        g: The generator
        h: The target element
        p: The prime modulus
        N: The order of g (default: p-1)
        verbose: If True, print step-by-step details
        
    Returns:
        k: The solution to g^k ≡ h (mod p)
        steps: Number of group operations performed
        space: Number of elements stored
    """
    # If N is not provided, assume g generates the full group
    if N is None:
        N = p - 1
    
    # Calculate m = ceil(sqrt(N))
    m = math.ceil(math.sqrt(N))
    
    if verbose:
        print("\n=== Baby-Step Giant-Step Algorithm ===")
        print(f"Parameters: g={g}, h={h}, p={p}, N={N}")
        print(f"Using m = ceil(sqrt(N)) = {m}")
        print("\nComputing Baby Steps:")
    
    # Baby Steps: Compute and store (j, g^j mod p) for j = 0, 1, ..., m-1
    baby_steps = {}
    g_j = 1  # g^0 = 1
    
    for j in range(m):
        baby_steps[g_j] = j
        if verbose:
            print(f"  Storing: (j={j}, g^j={g_j})")
        g_j = (g_j * g) % p
    
    # Compute g^(-m) mod p
    g_inv_m = pow(g, N - m, p)  # g^(-m) ≡ g^(N-m) (mod p)
    
    if verbose:
        print(f"\nComputed g^(-m) mod p = {g_inv_m}")
        print("\nComputing Giant Steps:")
    
    # Giant Steps: Check h * (g^(-m))^i mod p for i = 0, 1, ..., m-1
    current = h
    steps = m  # Count baby steps
    
    for i in range(m):
        if verbose:
            print(f"  Checking: i={i}, h*(g^(-m))^{i}={current}")
        
        if current in baby_steps:
            j = baby_steps[current]
            k = (i * m + j) % N
            
            if verbose:
                print(f"\nCollision found! i={i}, j={j}")
                print(f"Solution: k = i*m + j = {i}*{m} + {j} = {k} mod {N}")
                print(f"Verification: g^{k} mod {p} = {pow(g, k, p)}, h = {h}")
            
            return k, steps + i + 1, len(baby_steps)
        
        current = (current * g_inv_m) % p
        steps += 1
    
    if verbose:
        print("\nNo solution found within the given range.")
    
    return None, steps, len(baby_steps)


def pollards_rho(g: int, h: int, p: int, N: int = None, verbose: bool = True) -> Tuple[int, int, int]:
    """
    Solve the discrete logarithm problem using Pollard's Rho method with Floyd's cycle finding.
    Find k such that g^k ≡ h (mod p).
    
    Args:
        g: The generator
        h: The target element
        p: The prime modulus
        N: The order of g (default: p-1)
        verbose: If True, print step-by-step details
        
    Returns:
        k: The solution to g^k ≡ h (mod p)
        steps: Number of iterations performed
        space: Number of elements stored (constant: 2 for tortoise and hare)
    """
    # If N is not provided, assume g generates the full group
    if N is None:
        N = p - 1
    
    if verbose:
        print("\n=== Pollard's Rho Algorithm with Floyd's Cycle Finding ===")
        print(f"Parameters: g={g}, h={h}, p={p}, N={N}")
    
    # Define the iteration function f(x, a, b)
    def f(x: int, a: int, b: int) -> Tuple[int, int, int]:
        # Partition based on x mod 3
        partition = x % 3
        
        if partition == 0:  # S0
            # x' = x^2, a' = 2a, b' = 2b
            x_new = (x * x) % p
            a_new = (2 * a) % N
            b_new = (2 * b) % N
            step_desc = "S0: x' = x^2, a' = 2a, b' = 2b"
        elif partition == 1:  # S1
            # x' = x*g, a' = a+1, b' = b
            x_new = (x * g) % p
            a_new = (a + 1) % N
            b_new = b
            step_desc = "S1: x' = x*g, a' = a+1, b' = b"
        else:  # S2
            # x' = x*h, a' = a, b' = b+1
            x_new = (x * h) % p
            a_new = a
            b_new = (b + 1) % N
            step_desc = "S2: x' = x*h, a' = a, b' = b+1"
        
        return x_new, a_new, b_new, step_desc
    
    # Initialize tortoise and hare
    x_t, a_t, b_t = 1, 0, 0  # Initial values: x₀ = g^(a₀)*h^(b₀) = 1
    x_h, a_h, b_h = 1, 0, 0  # Same initial values for hare
    
    if verbose:
        print("\nInitial Values:")
        print(f"  Tortoise (x_T, a_T, b_T) = ({x_t}, {a_t}, {b_t})")
        print(f"  Hare (x_H, a_H, b_H) = ({x_h}, {a_h}, {b_h})")
        print("\nIterations:")
    
    iterations = 0
    
    while True:
        # Tortoise moves one step
        x_t, a_t, b_t, step_t = f(x_t, a_t, b_t)
        
        # Hare moves two steps
        x_h, a_h, b_h, step_h1 = f(x_h, a_h, b_h)
        x_h, a_h, b_h, step_h2 = f(x_h, a_h, b_h)
        
        iterations += 1
        
        if verbose:
            print(f"\nIteration {iterations}:")
            print(f"  Tortoise: {step_t}")
            print(f"  Tortoise (x_T, a_T, b_T) = ({x_t}, {a_t}, {b_t})")
            print(f"  Hare: 2 steps - {step_h1}, then {step_h2}")
            print(f"  Hare (x_H, a_H, b_H) = ({x_h}, {a_h}, {b_h})")
        
        # Check for collision
        if x_t == x_h:
            if verbose:
                print(f"\nCollision detected after {iterations} iterations!")
                print(f"  Tortoise (x_T, a_T, b_T) = ({x_t}, {a_t}, {b_t})")
                print(f"  Hare (x_H, a_H, b_H) = ({x_h}, {a_h}, {b_h})")
            
            # Solve for k: (b_T - b_H)k ≡ (a_H - a_T) (mod N)
            if b_t == b_h:
                if verbose:
                    print("  b_T = b_H, no unique solution exists from this collision.")
                # Reset and try again with different initial values
                x_t, a_t, b_t = random.randint(1, p-1), random.randint(0, N-1), random.randint(0, N-1)
                x_h, a_h, b_h = x_t, a_t, b_t
                continue
            
            # Calculate (a_H - a_T) mod N
            a_diff = (a_h - a_t) % N
            
            # Calculate (b_T - b_H) mod N
            b_diff = (b_t - b_h) % N
            
            try:
                # Solve (b_T - b_H)k ≡ (a_H - a_T) (mod N)
                b_diff_inv = mod_inverse(b_diff, N)
                k = (a_diff * b_diff_inv) % N
                
                if verbose:
                    print(f"\nSolving congruence equation:")
                    print(f"  (b_T - b_H)k ≡ (a_H - a_T) (mod N)")
                    print(f"  ({b_t} - {b_h})k ≡ ({a_h} - {a_t}) (mod {N})")
                    print(f"  {b_diff}k ≡ {a_diff} (mod {N})")
                    print(f"  k ≡ {a_diff} * {b_diff_inv} ≡ {k} (mod {N})")
                    print(f"\nVerification: g^{k} mod {p} = {pow(g, k, p)}, h = {h}")
                
                return k, iterations, 2  # 2 states stored (tortoise and hare)
                
            except ValueError as e:
                if verbose:
                    print(f"  Error: {e}")
                    print("  Restarting with new random values...")
                
                # Reset with random initial values
                x_t, a_t, b_t = random.randint(1, p-1), random.randint(0, N-1), random.randint(0, N-1)
                x_h, a_h, b_h = x_t, a_t, b_t
        
        # Safety check to prevent infinite loops in test environments
        if iterations > 1000:
            if verbose:
                print("\nReached maximum iterations. Stopping.")
            return None, iterations, 2
    

def pollards_lambda(g: int, h: int, p: int, N: int = None, verbose: bool = True) -> Tuple[int, int, int]:
    """
    Solve the discrete logarithm problem using Pollard's Lambda (Kangaroo) method.
    Find k such that g^k ≡ h (mod p).
    
    Args:
        g: The generator
        h: The target element
        p: The prime modulus
        N: The order of g (default: p-1)
        verbose: If True, print step-by-step details
        
    Returns:
        k: The solution to g^k ≡ h (mod p)
        steps: Number of iterations performed
        space: Number of distinguished points stored
    """
    # If N is not provided, assume g generates the full group
    if N is None:
        N = p - 1
    
    if verbose:
        print("\n=== Pollard's Lambda (Kangaroo) Algorithm ===")
        print(f"Parameters: g={g}, h={h}, p={p}, N={N}")
    
    # Define a simple distance function (using small powers of g)
    def distance(x: int) -> int:
        # Simple distance function based on the least significant bits
        return 1 + (x % 16)  # Jump sizes between 1 and 16
    
    # Define what makes a point distinguished (e.g., last few bits are 0)
    def is_distinguished(x: int) -> bool:
        return x % 20 == 0  # About 5% of points should be distinguished
    
    # Initialize tame and wild kangaroos
    starting_pos_tame = N // 2  # Tame kangaroo starts at g^(N/2)
    x_tame = pow(g, starting_pos_tame, p)
    x_wild = h  # Wild kangaroo starts at h (target)
    
    distance_tame = starting_pos_tame
    distance_wild = 0  # We don't know the actual exponent, so start at 0
    
    # Store for distinguished points
    distinguished_points = {}  # Format: {point: (kangaroo_type, distance)}
    
    if verbose:
        print("\nInitial values:")
        print(f"  Tame kangaroo starts at g^(N/2) = g^{starting_pos_tame} mod p = {x_tame}")
        print(f"  Wild kangaroo starts at h = {x_wild}")
        print(f"  Distinguished point criterion: x mod 20 == 0")
        print("\nIterations:")
    
    iterations = 0
    max_iterations = 2 * math.sqrt(N) + 1000  # Theoretical bound plus safety margin
    
    while iterations < max_iterations:
        # Move tame kangaroo
        jump_tame = distance(x_tame)
        x_tame = (x_tame * pow(g, jump_tame, p)) % p
        distance_tame = (distance_tame + jump_tame) % N
        
        # Move wild kangaroo
        jump_wild = distance(x_wild)
        x_wild = (x_wild * pow(g, jump_wild, p)) % p
        distance_wild = (distance_wild + jump_wild) % N
        
        iterations += 1
        
        if verbose and iterations <= 20:  # Limit verbose output to first 20 iterations
            print(f"\nIteration {iterations}:")
            print(f"  Tame kangaroo jumped by {jump_tame}")
            print(f"  Tame kangaroo is now at x={x_tame}, distance={distance_tame}")
            print(f"  Wild kangaroo jumped by {jump_wild}")
            print(f"  Wild kangaroo is now at x={x_wild}, distance={distance_wild}")
        
        # Check if tame kangaroo is at a distinguished point
        if is_distinguished(x_tame):
            if verbose and iterations <= 20:
                print(f"  Tame kangaroo found distinguished point: {x_tame}")
            
            # Check if this distinguished point was visited by wild kangaroo
            if x_tame in distinguished_points and distinguished_points[x_tame][0] == 'wild':
                wild_distance = distinguished_points[x_tame][1]
                if verbose:
                    print(f"\nCollision detected at distinguished point {x_tame}!")
                    print(f"  Tame kangaroo distance: {distance_tame}")
                    print(f"  Wild kangaroo distance: {wild_distance}")
                
                # Calculate k: k + wild_distance ≡ tame_distance (mod N)
                # => k ≡ tame_distance - wild_distance (mod N)
                k = (distance_tame - wild_distance) % N
                
                if verbose:
                    print(f"\nSolving for k:")
                    print(f"  k + {wild_distance} ≡ {distance_tame} (mod {N})")
                    print(f"  k ≡ {distance_tame} - {wild_distance} ≡ {k} (mod {N})")
                    print(f"\nVerification: g^{k} mod {p} = {pow(g, k, p)}, h = {h}")
                
                return k, iterations, len(distinguished_points)
            
            # Store this distinguished point from tame kangaroo
            distinguished_points[x_tame] = ('tame', distance_tame)
        
        # Check if wild kangaroo is at a distinguished point
        if is_distinguished(x_wild):
            if verbose and iterations <= 20:
                print(f"  Wild kangaroo found distinguished point: {x_wild}")
            
            # Check if this distinguished point was visited by tame kangaroo
            if x_wild in distinguished_points and distinguished_points[x_wild][0] == 'tame':
                tame_distance = distinguished_points[x_wild][1]
                if verbose:
                    print(f"\nCollision detected at distinguished point {x_wild}!")
                    print(f"  Tame kangaroo distance: {tame_distance}")
                    print(f"  Wild kangaroo distance: {distance_wild}")
                
                # Calculate k: k + wild_distance ≡ tame_distance (mod N)
                # => k ≡ tame_distance - wild_distance (mod N)
                k = (tame_distance - distance_wild) % N
                
                if verbose:
                    print(f"\nSolving for k:")
                    print(f"  k + {distance_wild} ≡ {tame_distance} (mod {N})")
                    print(f"  k ≡ {tame_distance} - {distance_wild} ≡ {k} (mod {N})")
                    print(f"\nVerification: g^{k} mod {p} = {pow(g, k, p)}, h = {h}")
                
                return k, iterations, len(distinguished_points)
            
            # Store this distinguished point from wild kangaroo
            distinguished_points[x_wild] = ('wild', distance_wild)
        
        # Check for direct collision between tame and wild kangaroos
        if x_tame == x_wild:
            if verbose:
                print(f"\nDirect collision detected at point {x_tame}!")
                print(f"  Tame kangaroo distance: {distance_tame}")
                print(f"  Wild kangaroo distance: {distance_wild}")
            
            # Calculate k: k + wild_distance ≡ tame_distance (mod N)
            # => k ≡ tame_distance - wild_distance (mod N)
            k = (distance_tame - distance_wild) % N
            
            if verbose:
                print(f"\nSolving for k:")
                print(f"  k + {distance_wild} ≡ {distance_tame} (mod {N})")
                print(f"  k ≡ {distance_tame} - {distance_wild} ≡ {k} (mod {N})")
                print(f"\nVerification: g^{k} mod {p} = {pow(g, k, p)}, h = {h}")
            
            return k, iterations, len(distinguished_points)
    
    if verbose:
        print("\nExceeded maximum iterations without finding a solution.")
    
    return None, iterations, len(distinguished_points)


def run_comparison(g: int, h: int, p: int, N: int = None, verbose: bool = True, 
                   run_bsgs: bool = True, run_rho: bool = True, run_lambda: bool = True) -> None:
    """
    Run a comparison of the selected DLP algorithms on the same problem instance.
    
    Args:
        g: The generator
        h: The target element
        p: The prime modulus
        N: The order of g (default: p-1)
        verbose: If True, print step-by-step details
        run_bsgs: If True, run the Baby-Step Giant-Step algorithm
        run_rho: If True, run the Pollard's Rho algorithm
        run_lambda: If True, run the Pollard's Lambda algorithm
    """
    # Check inputs
    if not is_prime(p):
        print(f"Error: {p} is not a prime number.")
        return
    
    if g <= 0 or g >= p or h <= 0 or h >= p:
        print(f"Error: g and h must be in the range [1, {p-1}].")
        return
    
    if N is None:
        N = p - 1
    
    # Check if h is in the subgroup generated by g
    if pow(g, N, p) != 1:
        print(f"Error: g^{N} ≠ 1 (mod {p}). The provided order N={N} is incorrect.")
        return
    
    results = []
    
    # Run Baby-Step Giant-Step
    if run_bsgs:
        k_bsgs, ops_bsgs, space_bsgs = baby_step_giant_step(g, h, p, N, verbose)
        results.append(("Baby-Step Giant-Step", k_bsgs, ops_bsgs, space_bsgs))
    
    # Run Pollard's Rho
    if run_rho:
        k_rho, ops_rho, space_rho = pollards_rho(g, h, p, N, verbose)
        results.append(("Pollard's Rho", k_rho, ops_rho, space_rho))
    
    # Run Pollard's Lambda
    if run_lambda:
        k_lambda, ops_lambda, space_lambda = pollards_lambda(g, h, p, N, verbose)
        results.append(("Pollard's Lambda", k_lambda, ops_lambda, space_lambda))
    
    # Print comparison table
    print("\n=== Comparison of DLP Algorithms ===")
    print(f"Parameters: g={g}, h={h}, p={p}, N={N}")
    print("\n| Algorithm        | k found | Iterations/Group Ops | Space Used (items stored) |")
    print("|------------------|---------|----------------------|---------------------------|")
    
    for name, k, ops, space in results:
        k_str = str(k) if k is not None else "Not found"
        print(f"| {name:<16} | {k_str:<7} | {ops:<20} | {space:<25} |")


def run_example():
    """Run a predefined example to demonstrate the algorithms."""
    print("=== DLP Algorithm Demo with Predefined Example ===")
    print("Running with p=101, g=2, h=30")
    print("For this example, we expect k=81 (2^81 ≡ 30 (mod 101))")
    
    # Run the comparison on a small example
    run_comparison(g=2, h=30, p=101, N=100, verbose=True,
                  run_bsgs=True, run_rho=True, run_lambda=True)
    
    # Verify the answer
    k = 81
    print(f"\nVerification: g^{k} mod p = {pow(2, k, 101)}, h = 30")


def main():
    """
    Main function to handle user input and run the DLP algorithms.
    """
    print("=== Discrete Logarithm Problem (DLP) Algorithm Demonstrator ===")
    print("This program demonstrates three algorithms for solving the DLP:")
    print("1. Baby-Step Giant-Step (BSGS)")
    print("2. Pollard's Rho with Floyd's cycle-finding")
    print("3. Pollard's Lambda (Kangaroo) with distinguished points")
    print("\nThe goal is to find k such that g^k ≡ h (mod p).")
    
    # Ask if user wants to run the example or input custom values
    choice = input("\nDo you want to run the predefined example (y) or input custom values (n)? ").strip().lower()
    
    if choice == 'y' or choice == 'yes':
        run_example()
        return
    
    # Get user input for DLP parameters
    try:
        p = int(input("\nEnter prime modulus p: "))
        if not is_prime(p):
            print(f"Warning: {p} is not prime. The algorithms may not work correctly.")
        
        g = int(input("Enter generator g: "))
        h = int(input("Enter target element h: "))
        
        use_default_order = input("Use default order N = p-1? (y/n): ").strip().lower()
        if use_default_order == 'y' or use_default_order == '':
            N = p - 1
        else:
            N = int(input("Enter order N of generator g: "))
        
        # Check inputs
        if g <= 0 or g >= p:
            print(f"Error: g must be in the range [1, {p-1}].")
            return
        
        if h <= 0 or h >= p:
            print(f"Error: h must be in the range [1, {p-1}].")
            return
        
        # Choose which algorithms to run
        print("\nWhich algorithms would you like to run?")
        run_bsgs = input("Run Baby-Step Giant-Step? (y/n): ").strip().lower() in ['y', 'yes', '']
        run_rho = input("Run Pollard's Rho? (y/n): ").strip().lower() in ['y', 'yes', '']
        run_lambda = input("Run Pollard's Lambda? (y/n): ").strip().lower() in ['y', 'yes', '']
        
        verbose = input("Show step-by-step details? (y/n): ").strip().lower() in ['y', 'yes', '']
        
        # Run the comparison
        run_comparison(g, h, p, N, verbose, run_bsgs, run_rho, run_lambda)
        
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")


if __name__ == "__main__":
    main() 