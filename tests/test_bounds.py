"""Held-Karp lower bound.

A lower bound that is too high is worse than no bound at all, because it silently understates
every reported gap. The decisive test is that the bound never exceeds a published optimum.
"""

from __future__ import annotations

import itertools

import pytest

from conftest import random_instance
from tsp.bounds import held_karp_bound, minimum_one_tree
from tsp.instance import DATA_ROOT, load_instance
from tsp.tour import tour_length

TSPLIB_WITH_OPTIMA = ["berlin52", "eil76", "kroA100", "ch150"]


def _brute_force_optimum(instance):
    """Exact optimum by enumeration. Only usable for very small n."""
    best = None
    for order in itertools.permutations(range(1, instance.n)):
        length = tour_length(instance.dist, [0, *order])
        if best is None or length < best:
            best = length
    return best


@pytest.mark.parametrize("name", TSPLIB_WITH_OPTIMA)
def test_bound_never_exceeds_the_published_optimum(name):
    instance = load_instance(DATA_ROOT / "tsplib" / f"{name}.tsp")
    result = held_karp_bound(instance, upper_bound=instance.optimum, time_limit=4.0)
    assert result.bound <= instance.optimum


@pytest.mark.parametrize("seed", range(4))
def test_bound_never_exceeds_the_true_optimum_on_small_instances(seed):
    instance = random_instance(8, seed=seed)
    optimum = _brute_force_optimum(instance)
    result = held_karp_bound(instance, upper_bound=optimum, time_limit=2.0)
    assert result.bound <= optimum


@pytest.mark.parametrize("seed", range(3))
def test_bound_holds_on_non_metric_instances(seed):
    """The relaxation is valid for any symmetric cost matrix, metric or not."""
    instance = random_instance(8, seed=seed, metric=False)
    optimum = _brute_force_optimum(instance)
    result = held_karp_bound(instance, upper_bound=optimum, time_limit=2.0)
    assert result.bound <= optimum


def test_one_tree_degrees_sum_to_twice_the_edge_count(berlin52):
    """A 1-tree on n cities has exactly n edges: n-2 in the spanning tree plus 2 at city 0."""
    _, degree = minimum_one_tree(berlin52.dist, [0.0] * berlin52.n, berlin52.n)
    assert sum(degree) == 2 * berlin52.n
    assert degree[0] == 2


def test_zero_potentials_give_the_plain_one_tree_bound(berlin52):
    """With no potentials the bound is the raw minimum 1-tree, which must still be a bound
    and must be weaker than the optimised one."""
    plain, _ = minimum_one_tree(berlin52.dist, [0.0] * berlin52.n, berlin52.n)
    optimised = held_karp_bound(berlin52, upper_bound=berlin52.optimum, time_limit=4.0)

    assert plain <= berlin52.optimum
    assert optimised.bound > plain


def test_subgradient_ascent_tightens_the_bound(eil76):
    short = held_karp_bound(eil76, upper_bound=eil76.optimum, time_limit=0.5)
    long = held_karp_bound(eil76, upper_bound=eil76.optimum, time_limit=4.0)
    assert long.bound >= short.bound
    assert long.bound <= eil76.optimum
