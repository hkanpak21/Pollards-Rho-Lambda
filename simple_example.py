from dlp_algorithms import baby_step_giant_step

def run_simple_bsgs():
    """Run a simple BSGS example with minimal output"""
    print("Testing Baby-Step Giant-Step algorithm...")
    
    # Parameters for a small example
    p = 101  # Prime modulus
    g = 2    # Generator
    h = 30   # Target element (we're looking for k where g^k = h mod p)
    N = 100  # Order of the generator
    
    # Run BSGS with minimal output
    k, steps, space = baby_step_giant_step(g, h, p, N, verbose=False)
    
    # Display results
    print(f"Problem: Find k where {g}^k ≡ {h} (mod {p})")
    print(f"Solution: k = {k}")
    print(f"Operations performed: {steps}")
    print(f"Space used: {space} elements stored")
    print(f"Verification: {g}^{k} mod {p} = {pow(g, k, p)}")

if __name__ == "__main__":
    run_simple_bsgs() 