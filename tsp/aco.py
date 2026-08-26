"""Rank-based Ant System with elitism, candidate lists, local search, and stagnation recovery.

Ants build tours one city at a time, choosing the next city with probability proportional to
``tau(i, j)^alpha * eta(i, j)^beta``, where ``tau`` is the learned pheromone on an edge and
``eta = 1 / d(i, j)`` is the greedy heuristic. Pheromone evaporates every iteration and is
reinforced along good tours, so edges that keep appearing in short tours become more likely
to be chosen again.

Three additions separate the configured algorithm from plain rank-based Ant System, and each
can be switched off independently through :class:`AcoConfig` so its contribution is
measurable:

``use_candidates``
    Restrict each construction step to the ``candidates`` nearest unvisited cities. This
    turns an O(n) choice into an O(k) one and, more importantly, stops ants from wasting
    probability mass on edges no good tour would ever use.

``use_local_search``
    Run 2-opt and Or-opt on the elite ants each iteration. The colony then samples the space
    of *local optima* rather than the space of raw constructed tours, which is what makes
    the difference at n in the hundreds.

``use_stagnation_recovery``
    Detect convergence by genotypic similarity, the fraction of edges the best and median
    ants share, and smooth the pheromone matrix back towards its initial value when that
    fraction crosses a threshold. This escapes a collapsed search without discarding
    everything learned so far.

Parameter defaults follow Bullnheimer, Hartl and Strauss for rank-based Ant System, with the
initial pheromone level scaled from a nearest neighbour tour as in Dorigo and Gambardella.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from tsp.budget import Budget
from tsp.construct import best_nearest_neighbour
from tsp.instance import Instance
from tsp.local_search import local_search
from tsp.result import SearchResult
from tsp.tour import assert_valid_tour

TAU_MIN = 1e-12


@dataclass
class AcoConfig:
    """Colony parameters and enhancement switches."""

    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.1
    elite_weight: int = 6
    n_ants: int | None = None
    candidates: int = 20
    neighbours: int = 10
    local_search_elites: int = 3
    similarity_threshold: float = 0.95
    smoothing_factor: float = 0.5

    use_candidates: bool = True
    use_local_search: bool = True
    use_stagnation_recovery: bool = True
    elitist: bool = True

    def ants_for(self, n: int) -> int:
        if self.n_ants is not None:
            return max(2, min(self.n_ants, n))
        return max(2, min(n, 20))


#: Rank-based Ant System with elitist reinforcement and none of the three enhancements.
ACO_BASELINE = AcoConfig(
    use_candidates=False,
    use_local_search=False,
    use_stagnation_recovery=False,
)


def _initial_pheromone(instance: Instance, config: AcoConfig, rng: random.Random) -> float:
    """Scale tau_0 to the length of a nearest neighbour tour.

    Absolute pheromone values are meaningless; what matters is their size relative to the
    deposits, which are ``weight / tour_length``. Deriving tau_0 from a reference tour length
    makes the same parameters behave the same way on instances whose distances differ by
    orders of magnitude.
    """
    _, reference = best_nearest_neighbour(instance, rng=rng)
    w = config.elite_weight
    return 0.5 * w * (w - 1) / (config.rho * max(reference, 1))


def _candidate_table(
    instance: Instance, config: AcoConfig
) -> list[list[tuple[int, float]]]:
    """For each city, its nearest candidates paired with a precomputed ``eta^beta``.

    The heuristic term never changes, so raising it to the power of beta once at startup
    removes a ``pow`` call from the innermost loop of the whole algorithm.
    """
    dist = instance.dist
    table: list[list[tuple[int, float]]] = []
    for i, row in enumerate(instance.neighbours(config.candidates)):
        entry = []
        for j in row:
            d = dist[i][j]
            eta = 1e6 if d == 0 else 1.0 / d
            entry.append((j, eta**config.beta))
        table.append(entry)
    return table


def _edge_set(tour: list[int]) -> set[tuple[int, int]]:
    edges = set()
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        edges.add((a, b) if a < b else (b, a))
    return edges


def _similarity(best: list[int], median: list[int]) -> float:
    """Fraction of edges the two tours share.

    Genotypic rather than phenotypic: two tours of identical length can be structurally
    different, and it is structural collapse that stalls the search.
    """
    shared = _edge_set(best) & _edge_set(median)
    return len(shared) / len(median)


def run_aco(
    instance: Instance,
    time_limit: float = 10.0,
    seed: int = 0,
    config: AcoConfig | None = None,
    max_iterations: int | None = None,
) -> SearchResult:
    """Run the colony until the time budget is exhausted, returning the best tour found."""
    config = config or AcoConfig()
    rng = random.Random(seed)
    n = instance.n
    dist = instance.dist
    n_ants = config.ants_for(n)
    w = config.elite_weight

    budget = Budget(time_limit=time_limit, max_iterations=max_iterations).start()

    tau0 = _initial_pheromone(instance, config, rng)
    pheromone = [[tau0] * n for _ in range(n)]
    for i in range(n):
        pheromone[i][i] = 0.0

    candidates = _candidate_table(instance, config) if config.use_candidates else None
    neighbours = instance.neighbours(config.neighbours) if config.use_local_search else None

    beta = config.beta
    alpha = config.alpha
    unit_alpha = alpha == 1.0
    evaporate = 1.0 - config.rho

    best_tour: list[int] = []
    best_length = -1

    while not budget.exhausted():
        budget.tick()

        tours: list[tuple[int, list[int]]] = []
        for _ in range(n_ants):
            if budget.exhausted():
                break
            tour, length = _construct(
                n, dist, pheromone, candidates, alpha, beta, unit_alpha, rng
            )
            tours.append((length, tour))

        if not tours:
            break

        tours.sort(key=lambda item: item[0])

        if config.use_local_search:
            improved = []
            for length, tour in tours[: config.local_search_elites]:
                tour, length = local_search(
                    instance, tour, length, neighbours=neighbours, budget=budget
                )
                improved.append((length, tour))
            tours[: config.local_search_elites] = improved
            tours.sort(key=lambda item: item[0])

        if best_length < 0 or tours[0][0] < best_length:
            best_length, best_tour = tours[0][0], list(tours[0][1])
            budget.record(best_length)

        if config.use_stagnation_recovery and len(tours) >= 3:
            median = tours[len(tours) // 2][1]
            if _similarity(tours[0][1], median) > config.similarity_threshold:
                _smooth(pheromone, n, tau0, config.smoothing_factor)

        for i in range(n):
            row = pheromone[i]
            pheromone[i] = [value if value > TAU_MIN else TAU_MIN for value in
                            [v * evaporate for v in row]]

        # Rank-based deposit: only the top w ants reinforce, weighted by how good they are.
        for rank, (length, tour) in enumerate(tours[:w]):
            _deposit(pheromone, tour, (w - rank) / length)

        if config.elitist and best_tour:
            _deposit(pheromone, best_tour, w / best_length)

    if not best_tour:
        best_tour, best_length = best_nearest_neighbour(instance, rng=rng)

    assert_valid_tour(best_tour, n, "aco")

    return SearchResult(
        tour=best_tour,
        length=best_length,
        algorithm="aco",
        instance=instance.name,
        seed=seed,
        iterations=budget.iterations,
        seconds=budget.elapsed,
        time_to_best=budget.time_to_best,
        trace=budget.trace,
    )


def _construct(
    n: int,
    dist: list[list[int]],
    pheromone: list[list[float]],
    candidates: list[list[tuple[int, float]]] | None,
    alpha: float,
    beta: float,
    unit_alpha: bool,
    rng: random.Random,
) -> tuple[list[int], int]:
    """Build one tour by probabilistic edge selection."""
    start = rng.randrange(n)
    tour = [start]
    visited = bytearray(n)
    visited[start] = 1
    current = start
    length = 0

    for _ in range(n - 1):
        choices: list[int] = []
        weights: list[float] = []
        trail = pheromone[current]

        if candidates is not None:
            for city, eta in candidates[current]:
                if not visited[city]:
                    tau = trail[city] if unit_alpha else trail[city] ** alpha
                    choices.append(city)
                    weights.append(tau * eta)

        if not choices:
            # Every candidate is already visited, so fall back to a full scan.
            row = dist[current]
            for city in range(n):
                if not visited[city] and city != current:
                    d = row[city]
                    eta = 1e6 if d == 0 else 1.0 / d
                    tau = trail[city] if unit_alpha else trail[city] ** alpha
                    choices.append(city)
                    weights.append(tau * eta**beta)

        if sum(weights) <= 0.0:
            nxt = rng.choice(choices)
        else:
            nxt = rng.choices(choices, weights=weights, k=1)[0]

        length += dist[current][nxt]
        tour.append(nxt)
        visited[nxt] = 1
        current = nxt

    length += dist[current][start]
    return tour, length


def _deposit(pheromone: list[list[float]], tour: list[int], amount: float) -> None:
    """Add ``amount`` to every edge of ``tour``, symmetrically."""
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        pheromone[a][b] += amount
        pheromone[b][a] += amount


def _smooth(pheromone: list[list[float]], n: int, tau0: float, factor: float) -> None:
    """Interpolate the whole matrix back towards tau_0.

    A full reset would throw away everything learned. Interpolating keeps the ranking of
    edges while compressing the differences between them, which restores exploration
    without restarting the search.
    """
    keep = 1.0 - factor
    shift = factor * tau0
    for i in range(n):
        row = pheromone[i]
        pheromone[i] = [shift + keep * value for value in row]
        pheromone[i][i] = 0.0
