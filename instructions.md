---

**Product Requirements Document: DLP Algorithm Demonstrator**

**1. Introduction**

*   **Product Name:** DLP Algorithm Demonstrator & Comparator
*   **Purpose:** To create a command-line or simple interactive Python program that implements and visually demonstrates the execution of three discrete logarithm problem (DLP) algorithms: Baby-Step Giant-Step (BSGS), Pollard's Rho, and Pollard's Lambda. The primary goal is to illustrate their differing approaches, computational steps, and resource usage (especially space complexity) on small, "toy" examples.
*   **Target User of this PRD:** AI Code Generation Agent
*   **Target User of the Final Program:** Students, educators, and individuals learning about cryptographic algorithms and their practical differences.

**2. Goals & Objectives**

*   **Primary Goal:** Clearly demonstrate the operational differences between BSGS, Pollard's Rho, and Pollard's Lambda when solving the same small DLP instance.
*   **Objectives:**
    *   Implement correct, functional versions of BSGS, Pollard's Rho (with Floyd's cycle-finding), and Pollard's Lambda (with distinguished points).
    *   Allow the user to specify parameters for a toy DLP instance (e.g., in $\mathbb{Z}_p^*$).
    *   Provide step-by-step textual output illustrating the key operations of each algorithm.
    *   Highlight differences in space complexity (e.g., items stored).
    *   Compare the number of group operations or iterations required.
    *   Ensure the code is modular and understandable, allowing for potential future extensions.

**3. User Stories**

*   As a student, I want to see the list of "baby steps" BSGS stores so I can understand its space requirement.
*   As a student, I want to watch Pollard's Rho generate its sequence and detect a cycle with the "tortoise and hare" to see how it avoids large storage.
*   As a student, I want to observe how Pollard's Lambda uses multiple "walkers" and "distinguished points" to find a collision.
*   As an educator, I want to run the same DLP instance through all three algorithms and get a comparative summary of their performance (steps, storage).
*   As a learner, I want to input a small prime $p$, a generator $g$, and a target value $h$ to see $k$ (where $g^k \equiv h \pmod p$) being found by each method.

**4. Functional Requirements**

*   **4.1. DLP Instance Setup:**
    *   The program must work with the Discrete Logarithm Problem in a finite cyclic group. For simplicity and "toy" demonstration, primarily target $\mathbb{Z}_p^*$ (the multiplicative group of integers modulo a prime $p$).
    *   User Inputs:
        *   Prime modulus $p$ (small, e.g., < 1000 for quick toy attacks).
        *   Generator $g$ of $\mathbb{Z}_p^*$.
        *   Target element $h \in \mathbb{Z}_p^*$ (such that $h = g^k \pmod p$ for some $k$).
        *   (Optional) The order $N$ of the generator $g$ (if not $p-1$). If not provided, it can be assumed $g$ generates the full group of order $p-1$ for simplicity, or $N$ can be pre-calculated.

*   **4.2. Baby-Step Giant-Step (BSGS) Implementation:**
    *   Calculate $m = \lceil \sqrt{N} \rceil$.
    *   Baby Steps: Compute and store pairs $(j, g^j \pmod p)$ for $j = 0, 1, \ldots, m-1$.
        *   **Output:** Display each baby step being computed and stored (e.g., "Storing (j, value)"). Show the final list/dictionary of stored baby steps.
    *   Giant Steps: Compute $g^{-m} \pmod p$. Then, for $i = 0, 1, \ldots, m-1$, compute $h \cdot (g^{-m})^i \pmod p$.
        *   **Output:** Display each giant step value being computed and checked against the stored baby steps.
    *   Collision: Identify when a giant step value matches a stored baby step value.
        *   **Output:** Announce the collision and the $i, j$ values.
    *   Solution: Calculate $k = i \cdot m + j \pmod N$.
        *   **Output:** Display the found $k$.
    *   Metrics: Report total baby steps stored, total giant steps computed.

*   **4.3. Pollard's Rho Implementation:**
    *   Function $f(x, a, b)$: Define a "random-looking" iteration function. A common choice for $\mathbb{Z}_p^*$ is to partition the group into $S_0, S_1, S_2$ and update $(x_i, a_i, b_i)$ based on which partition $x_i$ falls into:
        *   If $x_i \in S_0$: $x_{i+1} = x_i^2 \pmod p$, $a_{i+1} = 2a_i \pmod N$, $b_{i+1} = 2b_i \pmod N$.
        *   If $x_i \in S_1$: $x_{i+1} = x_i \cdot g \pmod p$, $a_{i+1} = a_i+1 \pmod N$, $b_{i+1} = b_i \pmod N$.
        *   If $x_i \in S_2$: $x_{i+1} = x_i \cdot h \pmod p$, $a_{i+1} = a_i \pmod N$, $b_{i+1} = b_i+1 \pmod N$.
        *   (Partitioning can be simple, e.g., $x \pmod 3$).
    *   Initial values: $x_0 = g^{a_0}h^{b_0}$ (e.g., $x_0=1, a_0=0, b_0=0$ or random $a_0, b_0$).
    *   Floyd's Cycle-Finding (Tortoise and Hare):
        *   Tortoise: $(x_T, a_T, b_T)$, advances one step using $f$.
        *   Hare: $(x_H, a_H, b_H)$, advances two steps using $f$.
        *   **Output:** In each iteration, display the current $(x_T, a_T, b_T)$ and $(x_H, a_H, b_H)$.
    *   Collision: Detect when $x_T = x_H$.
        *   **Output:** Announce the collision and the values $(a_T, b_T)$ and $(a_H, b_H)$.
    *   Solution: Solve $a_T + k \cdot b_T \equiv a_H + k \cdot b_H \pmod N$ for $k$. This simplifies to $(b_T - b_H)k \equiv (a_H - a_T) \pmod N$.
        *   **Output:** Show the congruence and the steps to solve for $k$ (including modular inverse). Display the found $k$.
    *   Metrics: Report total iterations until collision. Highlight that only a few states (tortoise, hare) are stored.

*   **4.4. Pollard's Lambda (Kangaroo) Implementation:**
    *   Two "kangaroos" (walkers): Tame (T) and Wild (W).
    *   Distance function $d(x)$: A simple function to define jump sizes (e.g., $d(x) = g^{x \pmod s}$ for small $s$, or just a few predefined jump sizes $g^{d_j}$).
    *   Tame Kangaroo: Starts at $x_T = g^{N/2}$ (known exponent), total distance covered $D_T = N/2$.
    *   Wild Kangaroo: Starts at $x_W = h$ (unknown exponent $k$), total distance covered $D_W = 0$.
    *   Iteration: In each step, both kangaroos "jump":
        *   $D_T \leftarrow D_T + \text{distance_of_jump_from_}x_T \pmod N$
        *   $x_T \leftarrow x_T \cdot \text{jump_from_}x_T \pmod p$
        *   $D_W \leftarrow D_W + \text{distance_of_jump_from_}x_W \pmod N$
        *   $x_W \leftarrow x_W \cdot \text{jump_from_}x_W \pmod p$
        *   **Output:** Display current $(x_T, D_T)$ and $(x_W, D_W)$ for each step.
    *   Distinguished Points: Define a simple property (e.g., $x \pmod{dp\_mod} == 0$).
        *   When a kangaroo lands on a distinguished point, store $(x, D)$.
        *   **Output:** Announce when a distinguished point is hit and stored.
    *   Collision:
        *   If $x_T = x_W$: Collision! Solve $D_T \equiv k + D_W \pmod N \implies k \equiv D_T - D_W \pmod N$.
        *   If Tame kangaroo lands on a distinguished point $(x_{dp}, D_{dp})$ previously visited by Wild: Collision! Solve $D_{dp\_Tame} \equiv k + D_{dp\_Wild} \pmod N$.
        *   If Wild kangaroo lands on a distinguished point $(x_{dp}, D_{dp})$ previously visited by Tame: Collision!
        *   **Output:** Announce the collision type and the values used.
    *   Solution: Calculate $k$.
        *   **Output:** Display the found $k$.
    *   Metrics: Report total iterations, number of distinguished points stored.

*   **4.5. Comparative Output:**
    *   After running all selected algorithms on the same DLP instance:
        *   Display a summary table:
            | Algorithm        | $k$ found | Iterations/Group Ops | Space Used (items stored) |
            |------------------|-----------|----------------------|---------------------------|
            | BSGS             | ...       | ...                  | (e.g., $m$ baby steps)    |
            | Pollard's Rho    | ...       | ...                  | (e.g., 2 states)          |
            | Pollard's Lambda | ...       | ...                  | (e.g., \#dist. points)    |

*   **4.6. User Interface (CLI):**
    *   Prompt user for $p, g, h, N$.
    *   Allow user to select which algorithm(s) to run.
    *   Option for verbose output (step-by-step) or summary output.

**5. Non-Functional Requirements**

*   **5.1. Correctness:** Algorithms must be implemented correctly and find the correct $k$.
*   **5.2. Clarity of Output:** The step-by-step demonstration should be easy to follow.
*   **5.3. Modularity:** Code should be organized into functions/classes for each algorithm to promote reusability and readability.
*   **5.4. Simplicity:** Prioritize clear demonstration over optimal performance for very large numbers (it's a "toy").
*   **5.5. Error Handling:** Basic error handling for invalid inputs (e.g., $h$ not in group generated by $g$, non-prime $p$).

**6. Technical Considerations**

*   **Programming Language:** Python 3.x
*   **Libraries:** Standard Python libraries. No complex external dependencies are strictly necessary for the core logic, but `math` module will be used.
*   **Group Order $N$:** For simplicity, for $\mathbb{Z}_p^*$, $N$ can be $p-1$ if $g$ is a primitive root. If $g$ has a smaller order $N$, this must be provided or calculated. The Pollard's Rho/Lambda equations require arithmetic modulo $N$.
*   **Modular Inverse:** A function for `modInverse(a, m)` will be needed.

**7. Output and Visualization**

*   Primarily text-based output.
*   Clear separation between the execution logs of different algorithms.
*   Emphasis on showing stored items for BSGS versus minimal storage for Rho/Lambda.
*   For Lambda, show the list of stored distinguished points.

**8. Scope**

*   **In Scope:**
    *   Implementation of BSGS, Pollard's Rho (Floyd's), Pollard's Lambda (2 kangaroos, distinguished points) for $\mathbb{Z}_p^*$.
    *   User input for DLP parameters for $\mathbb{Z}_p^*$.
    *   Detailed textual step-by-step execution trace.
    *   Comparative summary of key metrics.
*   **Out of Scope (for initial version):**
    *   Graphical User Interface (GUI).
    *   Support for other groups (e.g., Elliptic Curves) unless trivial to add.
    *   Highly optimized implementations for speed on large numbers.
    *   Advanced variants of Rho/Lambda (e.g., parallel Rho, multi-kangaroo Lambda beyond 2).
    *   Automatic calculation of group/element order for complex cases.

**9. Success Metrics**

*   The program correctly solves DLP instances for given toy parameters using all three methods.
*   The output clearly illustrates the fundamental differences in how each algorithm approaches the problem and manages resources (especially storage).
*   A user unfamiliar with the intricacies of these algorithms can gain a better understanding by observing the program's execution.

**10. Notes for AI Agent (Implementation Hints)**

*   Define a common interface or structure for how each algorithm is called and returns its results (e.g., `(k_found, iterations, space_metric)`).
*   For Pollard's Rho partitioning, $x \pmod 3$ is a simple way to divide elements into $S_0, S_1, S_2$.
*   For Pollard's Lambda distinguished points, $x \pmod k == 0$ for some small $k$ (e.g., $k=10$ or $k=16$) is a simple criterion.
*   Ensure modular arithmetic is handled correctly, especially the modulus $N$ for exponents and $p$ for group elements.
*   Be mindful of edge cases, e.g., if $b_T - b_H$ in Pollard's Rho is not invertible mod $N$. This means $(b_T - b_H)$ and $N$ share a common factor. The algorithm might need to be restarted with different parameters or the congruence solved using methods for $ax \equiv b \pmod n$ where $\gcd(a,n) > 1$. For toy examples, aim for cases where it's invertible.
*   The "space used" metric can be:
    *   BSGS: Number of elements in the baby-step dictionary.
    *   Rho: Constant (e.g., 2 for tortoise and hare states).
    *   Lambda: Number of distinguished points stored.

---

This PRD should provide a solid foundation for the AI agent to generate the desired Python program.