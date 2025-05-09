from dlp_algorithms import baby_step_giant_step

# Define a small test case
# We'll use p=101 (prime), g=2 (generator), and h=30 (target)
# For this example, we know that 2^81 ≡ 30 (mod 101)

p = 101
g = 2
h = 30
N = 100  # Order of g in Z_p*

print(f"Testing Baby-Step Giant-Step with p={p}, g={g}, h={h}")
k, steps, space = baby_step_giant_step(g, h, p, N, verbose=False)
print(f"Solution found: k={k}")
print(f"Steps performed: {steps}")
print(f"Space used: {space} elements stored")
print(f"Verification: g^k mod p = {pow(g, k, p)}, h = {h}") 