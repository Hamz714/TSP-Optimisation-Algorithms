"""Held-Karp lower bound via the 1-tree relaxation and subgradient ascent.

A heuristic tour length on its own says nothing about quality: without a reference, 48948
could be 1 percent or 40 percent above optimal. Computing a certified lower bound turns
every result into a bounded claim, because the true optimum is trapped between the bound
and the best tour found.

**The relaxation.** A *1-tree* is a spanning tree on cities ``1..n-1`` plus the two cheapest
edges incident to city 0. Every Hamiltonian tour is a 1-tree, so the minimum 1-tree cost is
a lower bound on the optimal tour. It is a weak bound on its own because 1-trees are free to
give cities a degree other than 2.

**Lagrangian strengthening.** Attach a potential ``pi_i`` to each city and solve the 1-tree
problem under modified costs ``d'(i, j) = d(i, j) + pi_i + pi_j``. Any tour's modified cost
is its true cost plus ``2 * sum(pi)``, since a tour uses exactly two edges at every city, so

    w(pi) = min_1tree_cost(d') - 2 * sum(pi)

is a valid lower bound for *every* choice of ``pi``. Penalising cities whose 1-tree degree
is not 2 pushes the relaxation towards tours, tightening the bound.

**Maximising it.** ``w`` is concave and piecewise linear in ``pi``, so it is maximised by
subgradient ascent. The subgradient is ``g_i = degree_i - 2``, and the step size follows the
standard Held-Karp target rule ``t = alpha * (UB - w) / ||g||^2`` with ``alpha`` halved after
a run of non-improving iterations. The direction is smoothed with the previous subgradient,
which is the Volgenant and Jonker refinement and converges noticeably faster than the raw
subgradient.

If the subgradient ever becomes zero, every city has degree 2, so the 1-tree *is* a tour and
the bound is exactly optimal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tsp.budget import Budget
from tsp.construct import best_nearest_neighbour
from tsp.instance import Instance
from tsp.local_search import local_search


@dataclass
class BoundResult:
    """Outcome of a bound computation."""

    instance: str
    bound: int
    upper_bound: int
    iterations: int
    seconds: float
    optimal: bool = False

    @property
    def gap_percent(self) -> float:
        """How far the incumbent tour is above the bound, as a percentage."""
        return 100.0 * (self.upper_bound - self.bound) / self.bound


def minimum_one_tree(
    dist: list[list[int]], pi: list[float], n: int
) -> tuple[float, list[int]]:
    """Minimum 1-tree under modified costs. Returns its cost and the degree of each city.

    Prim's algorithm on a dense graph, O(n^2), which is the right choice here because the
    modified cost matrix is dense and changes every iteration.
    """
    infinity = float("inf")
    in_tree = [False] * n
    cheapest = [infinity] * n
    parent = [-1] * n
    degree = [0] * n

    cheapest[1] = 0.0
    total = 0.0

    for _ in range(n - 1):
        chosen = -1
        best_cost = infinity
        for v in range(1, n):
            if not in_tree[v] and cheapest[v] < best_cost:
                best_cost = cheapest[v]
                chosen = v

        in_tree[chosen] = True
        total += best_cost
        if parent[chosen] != -1:
            degree[chosen] += 1
            degree[parent[chosen]] += 1

        row = dist[chosen]
        pi_chosen = pi[chosen]
        for v in range(1, n):
            if not in_tree[v]:
                cost = row[v] + pi_chosen + pi[v]
                if cost < cheapest[v]:
                    cheapest[v] = cost
                    parent[v] = chosen

    # City 0 rejoins through its two cheapest modified edges.
    row0 = dist[0]
    pi0 = pi[0]
    first_cost = second_cost = infinity
    first = second = -1
    for v in range(1, n):
        cost = row0[v] + pi0 + pi[v]
        if cost < first_cost:
            second_cost, second = first_cost, first
            first_cost, first = cost, v
        elif cost < second_cost:
            second_cost, second = cost, v

    total += first_cost + second_cost
    degree[0] = 2
    degree[first] += 1
    degree[second] += 1

    return total, degree


def held_karp_bound(
    instance: Instance,
    upper_bound: int | None = None,
    time_limit: float = 30.0,
    max_iterations: int = 10_000,
    smoothing: float = 0.7,
    halving_period: int = 20,
) -> BoundResult:
    """Compute a lower bound on the optimal tour length for ``instance``.

    ``upper_bound`` should be the best tour length known. If omitted, a nearest neighbour
    tour improved by local search is used. A tighter upper bound makes the step size rule
    better behaved and the resulting bound tighter.
    """
    n = instance.n
    dist = instance.dist

    if upper_bound is None:
        tour, length = best_nearest_neighbour(instance)
        tour, length = local_search(instance, tour, length)
        upper_bound = length

    budget = Budget(time_limit=time_limit, max_iterations=max_iterations).start()

    pi = [0.0] * n
    previous_gradient = [0] * n
    best = 0.0
    alpha = 2.0
    since_improvement = 0
    iterations = 0
    optimal = False

    while not budget.exhausted():
        budget.tick()
        iterations += 1

        cost, degree = minimum_one_tree(dist, pi, n)
        value = cost - 2.0 * sum(pi)

        if value > best:
            best = value
            since_improvement = 0
        else:
            since_improvement += 1

        gradient = [degree[i] - 2 for i in range(n)]
        if not any(gradient):
            # A 1-tree where every city has degree 2 is a tour, so this bound is exact.
            best = value
            optimal = True
            break

        # Volgenant and Jonker: blend with the previous direction to damp oscillation.
        direction = [gradient[i] + smoothing * previous_gradient[i] for i in range(n)]
        norm = sum(component * component for component in direction)
        if norm <= 0.0:
            break

        step = alpha * max(upper_bound - value, 1e-9) / norm
        pi = [pi[i] + step * direction[i] for i in range(n)]
        previous_gradient = gradient

        if since_improvement >= halving_period:
            alpha *= 0.5
            since_improvement = 0
            if alpha < 1e-5:
                break

    # Distances are integers, so the optimum is an integer and the bound can be rounded up.
    # The epsilon guards against a float value sitting a hair above an exact integer.
    bound = math.ceil(best - 1e-6 * max(1.0, abs(best)))

    return BoundResult(
        instance=instance.name,
        bound=bound,
        upper_bound=upper_bound,
        iterations=iterations,
        seconds=budget.elapsed,
        optimal=optimal,
    )
