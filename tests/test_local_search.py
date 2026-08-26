"""Local search invariants.

The move operators mutate a tour in place and track its length from deltas, which is exactly
the kind of code where a sign error produces a plausible-looking but wrong answer. These
tests pin down the two things that must always hold: the tour stays a permutation, and the
incrementally maintained length always equals a from-scratch recomputation.
"""

from __future__ import annotations

import random

import pytest

from conftest import random_instance
from tsp.construct import greedy_edge, nearest_neighbour
from tsp.local_search import _apply_or_opt, _reverse, local_search
from tsp.tour import is_valid_tour, tour_length


def _positions(tour):
    pos = [0] * len(tour)
    for index, city in enumerate(tour):
        pos[city] = index
    return pos


@pytest.mark.parametrize("seed", range(20))
def test_reverse_keeps_a_valid_permutation_and_consistent_positions(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 25)
    tour = list(range(n))
    rng.shuffle(tour)
    pos = _positions(tour)

    i, j = rng.randrange(n), rng.randrange(n)
    _reverse(tour, pos, i, j)

    assert is_valid_tour(tour, n)
    assert pos == _positions(tour)


@pytest.mark.parametrize("seed", range(20))
def test_reverse_produces_the_intended_edges(seed):
    """Reversing the arc after ``i`` up to ``j`` must replace exactly the two intended edges."""
    rng = random.Random(seed)
    n = rng.randint(6, 20)
    tour = list(range(n))
    rng.shuffle(tour)
    pos = _positions(tour)

    i = rng.randrange(n)
    j = (i + rng.randint(1, n - 2)) % n
    a, t2 = tour[i], tour[(i + 1) % n]
    b, t4 = tour[j], tour[(j + 1) % n]

    _reverse(tour, pos, (i + 1) % n, j)

    edges = {
        frozenset((tour[k], tour[(k + 1) % n])) for k in range(n)
    }
    assert frozenset((a, b)) in edges
    assert frozenset((t2, t4)) in edges


@pytest.mark.parametrize("seed", range(20))
def test_or_opt_move_keeps_a_valid_permutation(seed):
    rng = random.Random(seed)
    n = rng.randint(8, 25)
    tour = list(range(n))
    rng.shuffle(tour)
    pos = _positions(tour)

    seg_len = rng.randint(1, 3)
    start = rng.randrange(n)
    segment = {tour[(start + k) % n] for k in range(seg_len)}
    after = rng.choice([city for city in tour if city not in segment])

    _apply_or_opt(tour, pos, start, seg_len, after, reverse=bool(seed % 2))

    assert is_valid_tour(tour, n)
    assert pos == _positions(tour)


@pytest.mark.parametrize("metric", [True, False])
@pytest.mark.parametrize("seed", range(8))
def test_local_search_never_worsens_and_bookkeeping_is_exact(metric, seed):
    """Also covers non-metric instances, where triangle-inequality based pruning would be
    unsound. Nothing in the implementation relies on it, and this is what proves it."""
    instance = random_instance(40, seed=seed, metric=metric)
    start, length = nearest_neighbour(instance, 0)

    improved, improved_length = local_search(instance, start, length)

    assert is_valid_tour(improved, instance.n)
    assert improved_length == tour_length(instance.dist, improved)
    assert improved_length <= length


@pytest.mark.parametrize("seed", range(10))
def test_repeated_passes_never_worsen_and_converge(seed):
    """Don't-look bits are an approximation, deliberately.

    A city is only rescanned when a move touches one of its incident edges, so a further
    pass over a fresh queue occasionally still finds something (empirically a few percent of
    the time). That is the intended trade: the speedup is worth far more inside a
    metaheuristic than the last fraction of a percent from an exhaustive descent. What must
    hold is that passes are monotone and that they settle.
    """
    instance = random_instance(50, seed=seed)
    tour, length = nearest_neighbour(instance, 0)

    lengths = []
    for _ in range(6):
        tour, length = local_search(instance, tour, length)
        assert length == tour_length(instance.dist, tour)
        lengths.append(length)

    assert lengths == sorted(lengths, reverse=True)
    assert lengths[-1] == lengths[-2]


def test_local_search_does_not_modify_its_input(berlin52):
    tour, length = nearest_neighbour(berlin52, 0)
    original = list(tour)
    local_search(berlin52, tour, length)
    assert tour == original


def test_or_opt_finds_moves_two_opt_cannot(berlin52):
    """Or-opt relocates segments, which is not expressible as a single 2-opt move, so the
    combined neighbourhood should reach at least as good a local optimum on average."""
    two_opt_only = []
    combined = []
    for start in range(0, berlin52.n, 5):
        tour, length = nearest_neighbour(berlin52, start)
        two_opt_only.append(local_search(berlin52, tour, length, use_or_opt=False)[1])
        combined.append(local_search(berlin52, tour, length, use_or_opt=True)[1])

    assert sum(combined) / len(combined) < sum(two_opt_only) / len(two_opt_only)


def test_local_search_handles_tiny_instances():
    instance = random_instance(3, seed=1)
    tour, length = nearest_neighbour(instance, 0)
    result, result_length = local_search(instance, tour, length)
    assert is_valid_tour(result, 3)
    assert result_length == tour_length(instance.dist, result)


def test_local_search_improves_greedy_construction(berlin52):
    tour, length = greedy_edge(berlin52)
    _, improved = local_search(berlin52, tour, length)
    assert improved < length
