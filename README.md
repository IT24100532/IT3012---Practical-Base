# Practical 06: Genetic Algorithm for 0/1 Knapsack

A GA built with [DEAP](https://deap.readthedocs.io/) that solves the 0/1 knapsack problem and compares
selection, crossover and mutation settings.

## Files

| Path | What it is |
|---|---|
| `knapsack_ga.py` | The single script that runs every experiment |
| `data/knapPI_1_100_1000_1.csv` | Main instance: 100 items, capacity 995, known optimum 9147 |
| `data/knapPI_1_500_1000_1.csv` | Larger instance (part C): 500 items, capacity 2543, known optimum 28857 |
| `results/` | Result tables (CSV) written by the script |
| `figures/` | Convergence plots written by the script |
| `report.typ` / `report.pdf` | The report (Typst source and compiled PDF) |

The instances are the first instance of `knapPI_1_100_1000.csv` and `knapPI_1_500_1000.csv` from
Pisinger's `smallcoeff_pisinger.tgz` (<http://hjemmesider.diku.dk/~pisinger/codes.html>).
Each file has a header (`n`, capacity `c`, optimum `z`) followed by one `id,value,weight,x` row per item.

## Reproduce the results

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python knapsack_ga.py            # about 1.5 minutes
```

`python knapsack_ga.py --runs 3` does a faster check with fewer runs per variant.

## Settings

- **Encoding:** a list of `n` bits, where 1 means the item is taken.
- **Fitness (hard penalty):** `value(x) - M * max(0, weight(x) - C)` with `M = 10 * max(v)`.
- **Baseline:** tournament selection (k = 3), 2-point crossover (pc = 0.9), bit-flip mutation (pm = 0.02 per gene),
  elitism E = 2, 200 generations, population 150.
- **Fair comparison:** every variant changes one setting from the baseline and runs 10 times with seeds 0–9.
- **Convergence speed:** the first generation whose best fitness reaches the baseline's median final best fitness (8267.5).

DEAP's `eaSimple` has no elitism, so the script uses a short generation loop with DEAP's operators instead.
DEAP has no rank selection, and its roulette needs positive fitness, so both are implemented in the script
(roulette shifts fitness so the worst individual has weight 1).
