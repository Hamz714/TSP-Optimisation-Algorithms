"""End-to-end algorithm behaviour: validity, reproducibility, and that enhancements help."""

from __future__ import annotations

import pytest

from conftest import random_instance
from tsp.aco import ACO_BASELINE, AcoConfig, run_aco
from tsp.algorithms import ALGORITHMS, resolve
from tsp.construct import best_nearest_neighbour, greedy_edge, nearest_neighbour
from tsp.pso import PSO_BASELINE, PSO_SWAP, PsoConfig, run_pso
from tsp.tour import is_valid_tour, read_tour, tour_length, write_tour


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_every_algorithm_returns_a_valid_tour_with_an_honest_length(name, berlin52):
    result = resolve(name)(berlin52, 1.0, 0)
    assert is_valid_tour(result.tour, berlin52.n)
    assert result.length == tour_length(berlin52.dist, result.tour)
    assert result.algorithm == name


@pytest.mark.parametrize("name", ["nn", "greedy", "nn+2opt", "greedy+2opt"])
def test_constructive_heuristics_are_valid(name, eil76):
    result = resolve(name)(eil76, 0.0, 0)
    assert is_valid_tour(result.tour, eil76.n)
    assert result.length == tour_length(eil76.dist, result.tour)


def test_resolve_rejects_unknown_names(berlin52):
    with pytest.raises(KeyError, match="unknown algorithm"):
        resolve("no-such-algorithm")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aco_is_reproducible_when_bounded_by_iterations(seed, eil76):
    """Time-budgeted runs cannot be bit-reproducible, because the iteration count depends on
    machine load. Bounding by iterations instead removes the wall clock from the result and
    the same seed must then give exactly the same tour."""
    kwargs = dict(time_limit=60.0, max_iterations=5, config=AcoConfig(), seed=seed)
    first = run_aco(eil76, **kwargs)
    second = run_aco(eil76, **kwargs)
    assert first.tour == second.tour
    assert first.length == second.length


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pso_is_reproducible_when_bounded_by_iterations(seed, eil76):
    kwargs = dict(time_limit=60.0, max_iterations=10, config=PsoConfig(), seed=seed)
    first = run_pso(eil76, **kwargs)
    second = run_pso(eil76, **kwargs)
    assert first.tour == second.tour
    assert first.length == second.length


def test_different_seeds_explore_differently(eil76):
    lengths = {
        run_aco(eil76, time_limit=60.0, max_iterations=3, seed=seed).length
        for seed in range(6)
    }
    assert len(lengths) > 1


@pytest.mark.parametrize("encoding", ["random_key", "swap"])
def test_pso_encodings_both_produce_valid_tours(encoding, berlin52):
    config = PSO_SWAP if encoding == "swap" else PSO_BASELINE
    result = run_pso(berlin52, time_limit=1.5, seed=0, config=config)
    assert is_valid_tour(result.tour, berlin52.n)
    assert result.length == tour_length(berlin52.dist, result.tour)


def test_enhanced_aco_beats_the_baseline(kroA100):
    """Measured on kroA100 rather than berlin52: at 52 cities both configurations reach the
    optimum within the budget, so the instance cannot separate them."""
    enhanced = run_aco(kroA100, time_limit=3.0, seed=0, config=AcoConfig())
    baseline = run_aco(kroA100, time_limit=3.0, seed=0, config=ACO_BASELINE)
    assert enhanced.length < baseline.length


def test_enhanced_pso_beats_the_baseline(kroA100):
    enhanced = run_pso(kroA100, time_limit=3.0, seed=0, config=PsoConfig())
    baseline = run_pso(kroA100, time_limit=3.0, seed=0, config=PSO_BASELINE)
    assert enhanced.length < baseline.length


def test_aco_finds_the_optimum_on_berlin52(berlin52):
    result = run_aco(berlin52, time_limit=5.0, seed=0)
    assert result.length == berlin52.optimum


@pytest.mark.parametrize("seed", range(3))
def test_algorithms_solve_a_tiny_instance_optimally(seed):
    """On six cities every configuration should reach the exact optimum."""
    import itertools

    instance = random_instance(6, seed=seed)
    optimum = min(
        tour_length(instance.dist, [0, *order])
        for order in itertools.permutations(range(1, 6))
    )
    assert run_aco(instance, time_limit=1.0, seed=seed).length == optimum
    assert run_pso(instance, time_limit=1.0, seed=seed).length == optimum


def test_trace_is_monotonically_improving(berlin52):
    result = run_aco(berlin52, time_limit=2.0, seed=0)
    lengths = [length for _, length in result.trace]
    assert lengths == sorted(lengths, reverse=True)
    assert lengths[-1] == result.length


def test_time_budget_is_respected(berlin52):
    result = run_aco(berlin52, time_limit=1.0, seed=0)
    assert result.seconds < 3.0


def test_best_nearest_neighbour_is_at_least_as_good_as_a_single_start(berlin52):
    _, single = nearest_neighbour(berlin52, 0)
    _, best = best_nearest_neighbour(berlin52, starts=berlin52.n)
    assert best <= single


def test_greedy_edge_produces_a_single_cycle(eil76):
    tour, length = greedy_edge(eil76)
    assert is_valid_tour(tour, eil76.n)
    assert length == tour_length(eil76.dist, tour)


def test_tour_round_trips_through_disk(tmp_path, berlin52):
    result = run_aco(berlin52, time_limit=1.0, seed=0)
    path = write_tour(tmp_path / "t.tour", berlin52, result.tour, result.length, {"algorithm": "aco"})
    tour, meta = read_tour(path)
    assert tour == result.tour
    assert int(meta["TOUR LENGTH"]) == result.length
    assert meta["ALGORITHM"] == "aco"
