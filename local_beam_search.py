import random

def local_beam_search(initial_states, k, max_iters, objective_fn):
    """
    initial_states: list of k randomly generated starting positions
    k: the beam width (number of states to keep)
    objective_fn: the function we are trying to maximize
    """
    current_states = initial_states

    for iteration in range(max_iters):
        all_successors = []   # Will hold tuples of (state, fitness)

        # --- STEP 1, 2 & 3: Generate, Evaluate, and Pool Neighbors ---
        for state in current_states:
            # Generate 5 random neighbors for this specific state
            for _ in range(5):
                neighbor = state + random.uniform(-10, 10)
                fit = objective_fn(neighbor)
                all_successors.append((neighbor, fit))

        # --- STEP 4: Prune the Beam (The Cut) ---
        all_successors.sort(key=lambda x: x[1], reverse=True)
        current_states = [item[0] for item in all_successors[:k]]

    return max(current_states, key=objective_fn)


# =====================================================================
# LOCAL TEST RUNNER - DO NOT MODIFY BELOW THIS LINE
# =====================================================================
if __name__ == "__main__":
    # Fix the random seed for reproducible testing
    random.seed(42)

    # NOTE: stand-in test (the lab sheet's runner is cut off after random.seed(42)).
    # Parabola with a single global maximum of 0 at x = 50.
    objective = lambda x: -(x - 50) ** 2
    k = 3
    starts = [random.uniform(-500, 500) for _ in range(k)]
    best = local_beam_search(starts, k=k, max_iters=100, objective_fn=objective)

    print(f"Initial states: {[round(s, 2) for s in starts]}")
    print(f"Best state found: x = {best:.4f}, f(x) = {objective(best):.4f}")
    if abs(best - 50) < 5:
        print("SUCCESS: Local Beam Search converged to the global maximum (x ~ 50).")
    else:
        print("FAILED: Beam did not converge to the global maximum.")
