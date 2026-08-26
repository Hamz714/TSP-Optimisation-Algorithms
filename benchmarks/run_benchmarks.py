"""Benchmark harness.

Produces every number quoted in the README. Four stages, each independently runnable:

``bounds``
    Held-Karp lower bounds for the instances with no published optimum, so their gaps have a
    reference. Written to ``results/bounds.json``.

``main``
    The headline grid: every configuration in :data:`tsp.algorithms.MAIN`, over every
    instance, repeated across seeds. Written to ``results/results.csv``.

``ablation``
    Each enhancement removed one at a time, on a representative subset. Written to
    ``results/ablation.csv``.

``stagnation``
    Stagnation recovery on and off at a long budget on small instances, which is the only
    regime where the colony converges far enough for the mechanism to engage at all. Written
    to ``results/stagnation.csv``.

Runs are distributed across cores with :mod:`multiprocessing`. Each job is a pure function of
``(instance, algorithm, seed, budget)``, so the work parallelises without any shared state.
The time budget is wall clock per run, so results depend on the machine; the README records
which one produced the published table.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsp.algorithms import ABLATION, ALGORITHMS, MAIN, resolve  # noqa: E402
from tsp.bounds import held_karp_bound  # noqa: E402
from tsp.instance import all_instances, load_instance  # noqa: E402
from tsp.tour import assert_valid_tour, tour_length  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

#: Instances used for the ablation study: a spread of sizes and of structure.
ABLATION_INSTANCES = ["eil76", "kroA100", "ch150", "lin318", "tsp180", "tsp535"]

SMALL_BUDGET = 10.0
LARGE_BUDGET = 60.0
LARGE_THRESHOLD = 200

_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class Job:
    instance_path: str
    algorithm: str
    seed: int
    time_limit: float


def budget_for(n: int) -> float:
    """Larger instances get a longer budget, because construction alone costs O(n) per ant."""
    return LARGE_BUDGET if n > LARGE_THRESHOLD else SMALL_BUDGET


def _instance(path: str):
    """Load an instance once per worker process and reuse it across that worker's jobs."""
    if path not in _CACHE:
        _CACHE[path] = load_instance(path)
    return _CACHE[path]


def run_job(job: Job) -> dict:
    instance = _instance(job.instance_path)
    runner = resolve(job.algorithm)

    started = time.perf_counter()
    result = runner(instance, job.time_limit, job.seed)
    wall = time.perf_counter() - started

    # Every published number is validated before it is written, not after.
    assert_valid_tour(result.tour, instance.n, job.algorithm)
    recomputed = tour_length(instance.dist, result.tour)
    if recomputed != result.length:
        raise AssertionError(
            f"{job.algorithm} on {instance.name}: reported {result.length}, measured {recomputed}"
        )

    return {
        "instance": instance.name,
        "n": instance.n,
        "algorithm": job.algorithm,
        "seed": job.seed,
        "length": result.length,
        "optimum": instance.optimum or "",
        "iterations": result.iterations,
        "time_limit": job.time_limit,
        "seconds": round(wall, 3),
        "time_to_best": round(result.time_to_best, 3),
    }


def bound_job(path: str) -> tuple[str, dict]:
    instance = _instance(path)
    result = held_karp_bound(instance, time_limit=60.0 if instance.n > LARGE_THRESHOLD else 30.0)
    return instance.name, {
        "n": instance.n,
        "bound": result.bound,
        "upper_bound": result.upper_bound,
        "iterations": result.iterations,
        "seconds": round(result.seconds, 2),
        "exact": result.optimal,
        "published_optimum": instance.optimum,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _execute(jobs: list[Job], workers: int, label: str) -> list[dict]:
    print(f"{label}: {len(jobs)} runs on {workers} workers", flush=True)
    estimate = sum(job.time_limit for job in jobs) / workers / 60
    print(f"{label}: roughly {estimate:.0f} minutes of wall clock", flush=True)

    rows: list[dict] = []
    started = time.perf_counter()
    with multiprocessing.Pool(workers) as pool:
        for index, row in enumerate(pool.imap_unordered(run_job, jobs), start=1):
            rows.append(row)
            if index % 10 == 0 or index == len(jobs):
                elapsed = (time.perf_counter() - started) / 60
                print(f"  {index}/{len(jobs)} done, {elapsed:.1f} min elapsed", flush=True)
    rows.sort(key=lambda row: (row["n"], row["instance"], row["algorithm"], row["seed"]))
    return rows


def stage_bounds(workers: int) -> None:
    paths = [str(path) for path in all_instances()]
    print(f"bounds: {len(paths)} instances", flush=True)
    with multiprocessing.Pool(workers) as pool:
        entries = dict(pool.map(bound_job, paths))

    invalid = [
        name
        for name, entry in entries.items()
        if entry["published_optimum"] and entry["bound"] > entry["published_optimum"]
    ]
    if invalid:
        raise AssertionError(f"lower bound exceeded a published optimum for: {invalid}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bounds.json").write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    print(f"bounds: written to {RESULTS / 'bounds.json'}", flush=True)


def stage_main(workers: int, seeds: int) -> None:
    jobs = []
    for path in all_instances():
        n = load_instance(path).n
        for algorithm in MAIN:
            for seed in range(seeds):
                jobs.append(Job(str(path), algorithm, seed, budget_for(n)))

    rows = _execute(jobs, workers, "main")
    _write_csv(RESULTS / "results.csv", rows)
    print(f"main: written to {RESULTS / 'results.csv'}", flush=True)


def stage_ablation(workers: int, seeds: int) -> None:
    paths = {path.stem: path for path in all_instances()}
    jobs = []
    for name in ABLATION_INSTANCES:
        path = paths[name]
        for algorithm in ABLATION:
            for seed in range(seeds):
                jobs.append(Job(str(path), algorithm, seed, SMALL_BUDGET))

    rows = _execute(jobs, workers, "ablation")
    _write_csv(RESULTS / "ablation.csv", rows)
    print(f"ablation: written to {RESULTS / 'ablation.csv'}", flush=True)


#: Small instances, where a long budget buys thousands of iterations and the colony has time
#: to actually converge. Stagnation recovery cannot be measured on anything else: at a
#: 10 second budget the similarity threshold is crossed on about 1 percent of iterations at
#: n = 100 and never at all beyond n = 300, so the mechanism is simply not reached.
STAGNATION_INSTANCES = ["tsp042", "tsp048", "berlin52", "eil76"]
STAGNATION_BUDGET = 60.0


def stage_stagnation(workers: int, seeds: int) -> None:
    paths = {path.stem: path for path in all_instances()}
    jobs = []
    for name in STAGNATION_INSTANCES:
        for algorithm in ("aco", "aco -stagnation"):
            for seed in range(seeds):
                jobs.append(Job(str(paths[name]), algorithm, seed, STAGNATION_BUDGET))

    rows = _execute(jobs, workers, "stagnation")
    _write_csv(RESULTS / "stagnation.csv", rows)
    print(f"stagnation: written to {RESULTS / 'stagnation.csv'}", flush=True)


def stage_environment() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cores": multiprocessing.cpu_count(),
                "small_budget_seconds": SMALL_BUDGET,
                "large_budget_seconds": LARGE_BUDGET,
                "large_threshold_cities": LARGE_THRESHOLD,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the published benchmarks.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "bounds", "main", "ablation", "stagnation"],
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--ablation-seeds", type=int, default=8)
    parser.add_argument("--stagnation-seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

    stage_environment()
    if args.stage in ("all", "bounds"):
        stage_bounds(args.workers)
    if args.stage in ("all", "main"):
        stage_main(args.workers, args.seeds)
    if args.stage in ("all", "ablation"):
        stage_ablation(args.workers, args.ablation_seeds)
    if args.stage in ("all", "stagnation"):
        stage_stagnation(args.workers, args.stagnation_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
