"""One registry of every runnable configuration.

The CLI and the benchmark harness both resolve names through here, so a configuration can
never drift between what is published in the results table and what a reader gets when they
run the same name from the command line.

The ``*-baseline`` entries are what make the ablation table meaningful: they are the same
code paths with the enhancements switched off, not separate implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tsp.aco import ACO_BASELINE, AcoConfig, run_aco
from tsp.construct import best_nearest_neighbour, greedy_edge, nearest_neighbour
from tsp.instance import Instance
from tsp.local_search import local_search
from tsp.pso import PSO_BASELINE, PSO_SWAP, PsoConfig, run_pso
from tsp.result import SearchResult

Runner = Callable[[Instance, float, int], SearchResult]


def _constructive(name: str, build) -> Runner:
    def run(instance: Instance, time_limit: float = 0.0, seed: int = 0) -> SearchResult:
        from tsp.budget import Budget

        budget = Budget(time_limit=max(time_limit, 1.0)).start()
        tour, length = build(instance, seed)
        budget.record(length)
        return SearchResult(
            tour=tour,
            length=length,
            algorithm=name,
            instance=instance.name,
            seed=seed,
            iterations=1,
            seconds=budget.elapsed,
            time_to_best=budget.time_to_best,
            trace=budget.trace,
        )

    return run


def _nn(instance: Instance, seed: int):
    return nearest_neighbour(instance, seed % instance.n)


def _nn_2opt(instance: Instance, seed: int):
    tour, length = nearest_neighbour(instance, seed % instance.n)
    return local_search(instance, tour, length)


def _greedy(instance: Instance, seed: int):
    return greedy_edge(instance)


def _greedy_2opt(instance: Instance, seed: int):
    tour, length = greedy_edge(instance)
    return local_search(instance, tour, length)


def _aco(config: AcoConfig) -> Runner:
    def run(instance: Instance, time_limit: float = 10.0, seed: int = 0) -> SearchResult:
        return run_aco(instance, time_limit=time_limit, seed=seed, config=config)

    return run


def _pso(config: PsoConfig, label: str) -> Runner:
    def run(instance: Instance, time_limit: float = 10.0, seed: int = 0) -> SearchResult:
        result = run_pso(instance, time_limit=time_limit, seed=seed, config=config)
        result.algorithm = label
        return result

    return run


def _labelled_aco(config: AcoConfig, label: str) -> Runner:
    inner = _aco(config)

    def run(instance: Instance, time_limit: float = 10.0, seed: int = 0) -> SearchResult:
        result = inner(instance, time_limit, seed)
        result.algorithm = label
        return result

    return run


ALGORITHMS: dict[str, Runner] = {
    "nn": _constructive("nn", _nn),
    "nn+2opt": _constructive("nn+2opt", _nn_2opt),
    "greedy": _constructive("greedy", _greedy),
    "greedy+2opt": _constructive("greedy+2opt", _greedy_2opt),
    "aco": _labelled_aco(AcoConfig(), "aco"),
    "aco-baseline": _labelled_aco(ACO_BASELINE, "aco-baseline"),
    "pso": _pso(PsoConfig(), "pso"),
    "pso-baseline": _pso(PSO_BASELINE, "pso-baseline"),
    "pso-swap": _pso(PSO_SWAP, "pso-swap"),
}

#: Configurations used for the headline results table.
MAIN = ["nn", "greedy+2opt", "aco-baseline", "aco", "pso-baseline", "pso"]

#: One enhancement removed at a time, for the ablation table.
ABLATION: dict[str, Runner] = {
    "aco": ALGORITHMS["aco"],
    "aco -local-search": _labelled_aco(
        replace(AcoConfig(), use_local_search=False), "aco -local-search"
    ),
    "aco -candidates": _labelled_aco(
        replace(AcoConfig(), use_candidates=False), "aco -candidates"
    ),
    "aco -stagnation": _labelled_aco(
        replace(AcoConfig(), use_stagnation_recovery=False), "aco -stagnation"
    ),
    "pso": ALGORITHMS["pso"],
    "pso -local-search": _pso(
        replace(PsoConfig(), local_search_rate=0.0), "pso -local-search"
    ),
    "pso -crossover": _pso(replace(PsoConfig(), crossover_rate=0.0), "pso -crossover"),
    "pso -dynamic-topology": _pso(
        replace(PsoConfig(), dynamic_topology=False), "pso -dynamic-topology"
    ),
    "pso -varying-inertia": _pso(
        replace(PsoConfig(), time_varying_inertia=False), "pso -varying-inertia"
    ),
}


def resolve(name: str) -> Runner:
    """Look up a runner by name, with a helpful error listing the valid names."""
    if name in ALGORITHMS:
        return ALGORITHMS[name]
    if name in ABLATION:
        return ABLATION[name]
    known = ", ".join(sorted(set(ALGORITHMS) | set(ABLATION)))
    raise KeyError(f"unknown algorithm {name!r}; choose one of: {known}")
