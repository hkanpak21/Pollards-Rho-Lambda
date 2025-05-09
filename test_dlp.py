from dlp_algorithms import run_comparison

# Define a small test case
# We'll use p=101 (prime), g=2 (generator), and h=30 (target)
# For this example, we know that 2^81 ≡ 30 (mod 101)

print("=== DLP Algorithm Test ===")
print("Testing with p=101, g=2, h=30, N=100")
print("Expected answer: k=81 (2^81 ≡ 30 (mod 101))")
print("\nRunning algorithms...")

# Run with less verbose output for cleaner test results
run_comparison(g=5, h=30, p=101, N=100, verbose=False, 
               run_bsgs=True, run_rho=True, run_lambda=True)

# Verify the answer
print("\nVerification:")
print(f"2^81 mod 101 = {pow(2, 81, 101)}")
print(f"Expected h = 30") 