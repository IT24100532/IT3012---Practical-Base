import argparse
import csv
import random
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# ---------------------------------------------------------------- dataset
def load_instance(path):
    """Read one Pisinger instance: header lines (name, n, c, z, time) then 'id,value,weight,x' rows."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    name = lines[0]
    header = dict(line.split(" ", 1) for line in lines[1:5])
    v, w = [], []
    for row in lines[5:]:
        _, value, weight, _ = row.split(",")
        v.append(int(value))
        w.append(int(weight))
    return name, v, w, int(header["c"]), int(header["z"])


# ---------------------------------------------------------------- GA design
def make_fitness(v, w, C, M):
    """Hard penalty: value(x) - M * max(0, weight(x) - C)."""
    v, w = np.array(v), np.array(w)

    def fitness(ind):
        x = np.array(ind)
        overweight = max(0, int(x @ w) - C)
        return (int(x @ v) - M * overweight,)
    return fitness


def sel_roulette(pop, k):
    """Fitness-proportional selection. Hard-penalty fitness can be negative, so shift it to be positive first."""
    fits = [ind.fitness.values[0] for ind in pop]
    low = min(fits)
    weights = [f - low + 1 for f in fits]
    return random.choices(pop, weights=weights, k=k)


def sel_rank(pop, k):
    """Linear rank selection: worst gets weight 1, best gets weight len(pop)."""
    ranked = sorted(pop, key=lambda ind: ind.fitness.values[0])
    return random.choices(ranked, weights=range(1, len(ranked) + 1), k=k)


SELECTIONS = {
    "tournament": lambda k: (tools.selTournament, {"tournsize": k}),
    "roulette": lambda k: (sel_roulette, {}),
    "rank": lambda k: (sel_rank, {}),
}
CROSSOVERS = {
    "1-point": (tools.cxOnePoint, {}),
    "2-point": (tools.cxTwoPoint, {}),
    "uniform": (tools.cxUniform, {"indpb": 0.5}),
}


def run_ga(v, w, C, seed, selection="tournament", tourn_k=3, crossover="2-point",
           pc=0.9, pm=0.02, elitism=2, gens=200, pop_size=150):
    """One GA run. Returns per-generation best/avg fitness and the best individual found."""
    random.seed(seed)
    n = len(v)
    select_fn, select_args = SELECTIONS[selection](tourn_k)
    mate_fn, mate_args = CROSSOVERS[crossover]

    tb = base.Toolbox()
    tb.register("attr_bool", random.randint, 0, 1)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_bool, n=n)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", make_fitness(v, w, C, M=10 * max(v)))
    tb.register("select", select_fn, **select_args)
    tb.register("mate", mate_fn, **mate_args)
    tb.register("mutate", tools.mutFlipBit, indpb=pm)

    pop = tb.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)

    best_hist, avg_hist = [], []
    for _ in range(gens + 1):
        fits = [ind.fitness.values[0] for ind in pop]
        best_hist.append(max(fits))
        avg_hist.append(sum(fits) / len(fits))
        if len(best_hist) > gens:
            break

        # Elitism: copy the top E unchanged, fill the rest with selected, crossed and mutated children
        elites = [tb.clone(ind) for ind in tools.selBest(pop, elitism)]
        offspring = [tb.clone(ind) for ind in tb.select(pop, pop_size - elitism)]
        for a, b in zip(offspring[::2], offspring[1::2]):
            if random.random() < pc:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for ind in offspring:
            tb.mutate(ind)  # every gene flips with probability pm
            del ind.fitness.values
        for ind in offspring:
            ind.fitness.values = tb.evaluate(ind)
        pop = elites + offspring

    best = tools.selBest(pop, 1)[0]
    return best_hist, avg_hist, best


# ---------------------------------------------------------------- experiments
BASELINE = dict(selection="tournament", tourn_k=3, crossover="2-point", pc=0.9, pm=0.02, elitism=2,
                gens=200, pop_size=150)

VARIANTS = [
    # (label, group, changes from baseline, note)
    ("Baseline (k=3, 2pt, pm=0.02)", "baseline", {}, "reference"),
    ("Tournament k=2", "selection", {"tourn_k": 2}, "less pressure"),
    ("Tournament k=4", "selection", {"tourn_k": 4}, "more pressure"),
    ("Roulette", "selection", {"selection": "roulette"}, "fitness-proportional"),
    ("Rank", "selection", {"selection": "rank"}, "rank-proportional"),
    ("1-point crossover", "crossover", {"crossover": "1-point"}, "fewer cut points"),
    ("Uniform crossover", "crossover", {"crossover": "uniform"}, "diversity trade-off"),
    ("pm = 0.01", "mutation", {"pm": 0.01}, "less mutation"),
    ("pm = 0.05", "mutation", {"pm": 0.05}, "more mutation"),
    ("pm = 0.10", "mutation", {"pm": 0.10}, "risk of randomization"),
]


def gens_to_reach(best_hist, target):
    return next((g for g, f in enumerate(best_hist) if f >= target), None)


def summarize(label, runs, target, C, w, optimum, note):
    finals = [r["best_hist"][-1] for r in runs]
    reached = [g for g in (gens_to_reach(r["best_hist"], target) for r in runs) if g is not None]
    feasible = sum(sum(g * wi for g, wi in zip(r["best"], w)) <= C for r in runs)
    # Fitness >= 0 means the best solution fits in the knapsack (no penalty)
    first_valid = [g for g in (gens_to_reach(r["best_hist"], 0) for r in runs) if g is not None]
    return {
        "variant": label,
        "best_mean": round(statistics.mean(finals), 1),
        "best_sd": round(statistics.stdev(finals), 1) if len(finals) > 1 else 0.0,
        "best_max": max(finals),
        "gap_to_optimum_pct": round(100 * (optimum - statistics.mean(finals)) / optimum, 1),
        "gens_to_baseline_median": statistics.median(reached) if reached else None,
        "runs_reaching": f"{len(reached)}/{len(runs)}",
        "first_valid_gen": statistics.median(first_valid) if first_valid else None,
        "feasible_runs": f"{feasible}/{len(runs)}",
        "note": note,
    }


def run_variant(v, w, C, seeds, **settings):
    runs = []
    for seed in seeds:
        best_hist, avg_hist, best = run_ga(v, w, C, seed=seed, **settings)
        runs.append({"best_hist": best_hist, "avg_hist": avg_hist, "best": best})
    return runs


def median_curve(runs, key):
    return np.median(np.array([r[key] for r in runs]), axis=0)


def plot_convergence(runs, title, path, optimum):
    best, avg = median_curve(runs, "best_hist"), median_curve(runs, "avg_hist")
    plt.figure(figsize=(7, 4))
    plt.plot(best, label="best (median of runs)")
    plt.plot(avg, label="average (median of runs)")
    plt.axhline(optimum, color="grey", linestyle="--", linewidth=1, label=f"known optimum ({optimum})")
    # Overweight solutions get huge negative penalties, so use a symlog scale to show both signs
    plt.yscale("symlog", linthresh=1000)
    plt.xlabel("Generation"); plt.ylabel("Fitness (symlog scale)"); plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_group(results, labels, title, path, optimum):
    plt.figure(figsize=(7, 4))
    for label in labels:
        plt.plot(median_curve(results[label], "best_hist"), label=label)
    plt.axhline(optimum, color="grey", linestyle="--", linewidth=1, label="known optimum")
    plt.ylim(0, optimum * 1.05)  # below 0 = best solution is still overweight
    plt.xlabel("Generation"); plt.ylabel("Best fitness (median of runs)"); plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def write_table(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'Variant':30} {'Best fitness (mean ± sd)':>26} {'Gens to baseline median':>24} "
          f"{'First valid gen':>16} {'Feasible':>9}  Note")
    for r in rows:
        gens = "not reached" if r["gens_to_baseline_median"] is None else f"{r['gens_to_baseline_median']:g} ({r['runs_reaching']})"
        valid = "never" if r["first_valid_gen"] is None else f"{r['first_valid_gen']:g}"
        print(f"{r['variant']:30} {r['best_mean']:>17.1f} ± {r['best_sd']:<6.1f} {gens:>24} {valid:>16} "
              f"{r['feasible_runs']:>9}  {r['note']}")


def main():
    parser = argparse.ArgumentParser(description="GA for 0/1 knapsack (Practical 06)")
    parser.add_argument("--runs", type=int, default=10, help="independent runs (seeds 0..runs-1) per variant")
    args = parser.parse_args()
    seeds = list(range(args.runs))
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # A + B: baseline and operator comparison on the 100-item instance
    name, v, w, C, z = load_instance(DATA / "knapPI_1_100_1000_1.csv")
    print(f"Instance {name}: n={len(v)}, C={C}, known optimum z={z}, seeds={seeds}")
    results = {label: run_variant(v, w, C, seeds, **{**BASELINE, **changes}) for label, _, changes, _ in VARIANTS}

    baseline_label = VARIANTS[0][0]
    target = statistics.median(r["best_hist"][-1] for r in results[baseline_label])
    print(f"Baseline median final best fitness (convergence target): {target:g}")
    rows = [summarize(label, results[label], target, C, w, z, note) for label, _, _, note in VARIANTS]
    write_table(rows, RESULTS / "operator_comparison.csv")
    print_table(rows, "A + B: baseline and operator comparison (100 items)")

    plot_convergence(results[baseline_label], "Baseline: best and average fitness (100 items)",
                     FIGURES / "baseline_convergence.png", z)
    for group, title in (("selection", "Selection methods"), ("crossover", "Crossover types"),
                         ("mutation", "Mutation rates")):
        labels = [baseline_label] + [label for label, g, _, _ in VARIANTS if g == group]
        plot_group(results, labels, f"{title} (100 items)", FIGURES / f"compare_{group}.png", z)

    # C: larger instance with baseline settings, and with pm scaled to ~1 flip per child (1/n)
    name, v, w, C, z = load_instance(DATA / "knapPI_1_500_1000_1.csv")
    print(f"\nInstance {name}: n={len(v)}, C={C}, known optimum z={z}")
    large = {
        "Baseline on 500 items": ("reference", run_variant(v, w, C, seeds, **BASELINE)),
        "pm = 1/n = 0.002": ("mutation scaled to size", run_variant(v, w, C, seeds, **{**BASELINE, "pm": 1 / len(v)})),
    }
    target = statistics.median(r["best_hist"][-1] for r in large["Baseline on 500 items"][1])
    rows = [summarize(label, runs, target, C, w, z, note) for label, (note, runs) in large.items()]
    write_table(rows, RESULTS / "large_instance.csv")
    print_table(rows, "C: larger instance (500 items)")
    plot_group({k: runs for k, (_, runs) in large.items()}, list(large), "500 items: baseline vs pm = 1/n",
               FIGURES / "large_instance.png", z)
    print(f"\nTables saved to {RESULTS.name}/, plots saved to {FIGURES.name}/")


if __name__ == "__main__":
    main()
