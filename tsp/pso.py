"""Particle Swarm Optimisation for the TSP, with two encodings.

PSO is defined over a continuous space, and the TSP is a permutation problem, so the whole
design question is how to bridge the two. Both standard answers are implemented here and
selectable through :class:`PsoConfig`.

``encoding = "random_key"``
    Each particle holds a real vector of length n, and the tour is read off by sorting the
    indices by their key. Velocity updates are then the textbook continuous ones. The catch
    is that the mapping is invariant to adding a constant to every key, so the swarm drifts
    along the all-ones direction without changing a single tour, and the drift eventually
    swamps the meaningful differences between keys. Every position is therefore projected
    onto the hyperplane ``sum(x) = 0`` by subtracting its mean, which removes exactly that
    null direction and leaves the encoded tour unchanged.

``encoding = "swap"``
    Positions are permutations and a velocity is a sequence of transpositions. Subtraction
    of two permutations yields the swap sequence that transforms one into the other,
    multiplication by a scalar truncates or repeats that sequence, and addition concatenates.
    The sequences grow without bound, so they are periodically reduced to the shortest
    sequence with the same effect.

On top of the encoding sit three mechanisms, each independently switchable:

*Time-varying inertia.* The inertia weight falls linearly from ``inertia_start`` to
``inertia_end``, trading exploration for exploitation. Critically the schedule is driven by
the fraction of the *time budget* consumed, not by an iteration counter: the iteration cap
is never the binding constraint, so an iteration-driven schedule barely moves before the run
is stopped.

*Dynamic topology.* The neighbourhood radius grows from a ring (radius 1, information
spreads slowly, many local searches proceed in parallel) to a star (radius n/2, every
particle sees the global best). Also driven by budget progress.

*Order crossover.* With probability ``crossover_rate`` a particle breeds with its best
neighbour using OX1, keeping a contiguous slice of its own best tour and filling the rest in
the neighbour's relative order. The child replaces the particle only if it is strictly
better. This moves a particle to a structurally different tour in one step, which velocity
updates alone cannot do.

*Memetic local search.* With probability ``local_search_rate`` a particle's personal best is
improved by 2-opt and Or-opt. Pure PSO is a weak TSP solver at scale; this is what makes it
competitive, and it is off in the baseline configuration so its contribution stays visible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from tsp.budget import Budget
from tsp.instance import Instance
from tsp.local_search import local_search
from tsp.result import SearchResult
from tsp.tour import assert_valid_tour, tour_length

POS_MIN = -5.0
POS_MAX = 5.0
VEL_MIN = -1.0
VEL_MAX = 1.0

INITIAL_VELOCITY_LEN = 10
MAX_VELOCITY_LEN = 100


@dataclass
class PsoConfig:
    """Swarm parameters and enhancement switches."""

    encoding: str = "random_key"
    n_particles: int = 50
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive: float = 0.75
    social: float = 2.9
    crossover_rate: float = 0.05
    local_search_rate: float = 0.05
    neighbours: int = 10

    dynamic_topology: bool = True
    time_varying_inertia: bool = True

    def inertia_at(self, progress: float) -> float:
        if not self.time_varying_inertia:
            return self.inertia_start
        return self.inertia_start - (self.inertia_start - self.inertia_end) * progress


#: Random-key PSO with a fixed ring topology, fixed inertia, no crossover, no local search.
PSO_BASELINE = PsoConfig(
    crossover_rate=0.0,
    local_search_rate=0.0,
    dynamic_topology=False,
    time_varying_inertia=False,
    inertia_start=0.6,
)

#: The discrete swap-sequence formulation, for comparison against the random-key encoding.
PSO_SWAP = PsoConfig(
    encoding="swap",
    crossover_rate=0.0,
    local_search_rate=0.0,
    dynamic_topology=False,
    time_varying_inertia=False,
    inertia_start=0.6,
)


class Particle:
    """A swarm member. ``best_tour`` is the permutation; how it is stored depends on the
    encoding."""

    __slots__ = ("position", "velocity", "best_position", "best_tour", "best_length")

    def __init__(self) -> None:
        self.position: list[float] | list[int] = []
        self.velocity: list[float] | list[int] = []
        self.best_position: list[float] | list[int] = []
        self.best_tour: list[int] = []
        self.best_length: int = 0


def keys_to_tour(position: list[float]) -> list[int]:
    """Random-key decoding: sort city indices by their key."""
    return sorted(range(len(position)), key=position.__getitem__)


def tour_to_keys(tour: list[int]) -> list[float]:
    """Inverse of :func:`keys_to_tour`, used to place a crossover child back into key space."""
    n = len(tour)
    position = [0.0] * n
    span = POS_MAX - POS_MIN
    for rank, city in enumerate(tour):
        position[city] = POS_MIN + span * rank / n
    return position


def _centre(position: list[float]) -> list[float]:
    """Project onto ``sum(x) = 0``, the drift-free representative of the encoding class."""
    mean = sum(position) / len(position)
    return [value - mean for value in position]


def _order_crossover(
    parent: list[int], donor: list[int], rng: random.Random
) -> list[int]:
    """OX1: keep a contiguous slice of ``parent``, fill the rest in ``donor`` order."""
    n = len(parent)
    start = rng.randrange(n - 1)
    end = rng.randrange(start + 1, n)

    child: list[int] = [-1] * n
    child[start:end] = parent[start:end]
    taken = set(parent[start:end])

    cursor = 0
    for index in range(n):
        if child[index] != -1:
            continue
        while donor[cursor] in taken:
            cursor += 1
        child[index] = donor[cursor]
        taken.add(donor[cursor])
    return child


def _swap_sequence(target: list[int], source: list[int]) -> list[int]:
    """Adjacent transpositions turning ``source`` into ``target``, as a bubble sort trace."""
    working = list(source)
    rank = {city: index for index, city in enumerate(target)}
    swaps: list[int] = []
    for _ in range(len(working) - 1):
        moved = False
        for j in range(len(working) - 1):
            if rank[working[j]] > rank[working[j + 1]]:
                swaps.append(j)
                working[j], working[j + 1] = working[j + 1], working[j]
                moved = True
        if not moved:
            break
    return swaps


def _scale_sequence(scalar: float, sequence: list[int]) -> list[int]:
    """Truncate for ``scalar < 1``, repeat for ``scalar > 1``."""
    if not sequence or scalar <= 0:
        return []
    if scalar < 1:
        return sequence[: int(scalar * len(sequence))]
    fraction, whole = math.modf(scalar)
    return sequence * int(whole) + sequence[: int(fraction * len(sequence))]


def _reduce_sequence(sequence: list[int], n: int) -> list[int]:
    """Shortest sequence with the same net effect, found by replaying and re-deriving it."""
    identity = list(range(n))
    final = list(identity)
    for index in sequence:
        final[index], final[index + 1] = final[index + 1], final[index]
    return _swap_sequence(final, identity)


def _best_neighbour(
    index: int, particles: list[Particle], radius: int
) -> Particle:
    """Best personal best within ``radius`` positions on the ring."""
    count = len(particles)
    best = particles[index]
    for offset in range(-radius, radius + 1):
        candidate = particles[(index + offset) % count]
        if candidate.best_length < best.best_length:
            best = candidate
    return best


def run_pso(
    instance: Instance,
    time_limit: float = 10.0,
    seed: int = 0,
    config: PsoConfig | None = None,
    max_iterations: int | None = None,
) -> SearchResult:
    """Run the swarm until the time budget is exhausted, returning the best tour found."""
    config = config or PsoConfig()
    rng = random.Random(seed)
    n = instance.n
    dist = instance.dist
    swarm_size = max(3, min(config.n_particles, max(3, n)))

    budget = Budget(time_limit=time_limit, max_iterations=max_iterations).start()
    neighbours = (
        instance.neighbours(config.neighbours) if config.local_search_rate > 0 else None
    )

    particles = [
        _spawn(instance, config, rng) for _ in range(swarm_size)
    ]

    best = min(particles, key=lambda p: p.best_length)
    best_tour = list(best.best_tour)
    best_length = best.best_length
    budget.record(best_length)

    while not budget.exhausted():
        budget.tick()
        progress = budget.progress
        inertia = config.inertia_at(progress)
        radius = _radius(config, swarm_size, progress)

        for index, particle in enumerate(particles):
            if budget.exhausted():
                break

            neighbour = _best_neighbour(index, particles, radius)

            if config.encoding == "swap":
                _step_swap(particle, neighbour, inertia, config, dist, n, rng)
            else:
                _step_keys(particle, neighbour, inertia, config, dist, rng)

            if config.crossover_rate > 0 and rng.random() < config.crossover_rate:
                _crossover_step(particle, neighbour, dist, config, rng)

            if config.local_search_rate > 0 and rng.random() < config.local_search_rate:
                improved, length = local_search(
                    instance,
                    particle.best_tour,
                    particle.best_length,
                    neighbours=neighbours,
                    budget=budget,
                )
                if length < particle.best_length:
                    _adopt(particle, improved, length, config)

            if particle.best_length < best_length:
                best_length = particle.best_length
                best_tour = list(particle.best_tour)
                budget.record(best_length)

    assert_valid_tour(best_tour, n, "pso")

    return SearchResult(
        tour=best_tour,
        length=best_length,
        algorithm="pso",
        instance=instance.name,
        seed=seed,
        iterations=budget.iterations,
        seconds=budget.elapsed,
        time_to_best=budget.time_to_best,
        trace=budget.trace,
    )


def _radius(config: PsoConfig, swarm_size: int, progress: float) -> int:
    if not config.dynamic_topology:
        return 1
    return 1 + int(progress * (swarm_size / 2 - 1))


def _spawn(instance: Instance, config: PsoConfig, rng: random.Random) -> Particle:
    n = instance.n
    particle = Particle()

    if config.encoding == "swap":
        tour = list(range(n))
        rng.shuffle(tour)
        particle.position = tour
        particle.velocity = [rng.randrange(n - 1) for _ in range(INITIAL_VELOCITY_LEN)]
        particle.best_position = list(tour)
        particle.best_tour = list(tour)
    else:
        position = _centre([rng.uniform(POS_MIN, POS_MAX) for _ in range(n)])
        particle.position = position
        particle.velocity = [rng.uniform(VEL_MIN, VEL_MAX) for _ in range(n)]
        particle.best_position = list(position)
        particle.best_tour = keys_to_tour(position)

    particle.best_length = tour_length(instance.dist, particle.best_tour)
    return particle


def _adopt(particle: Particle, tour: list[int], length: int, config: PsoConfig) -> None:
    """Install ``tour`` as the particle's position and personal best."""
    particle.best_tour = list(tour)
    particle.best_length = length
    if config.encoding == "swap":
        particle.position = list(tour)
        particle.best_position = list(tour)
        particle.velocity = []
    else:
        position = _centre(tour_to_keys(tour))
        particle.position = position
        particle.best_position = list(position)
        # Old momentum points at the previous basin, so it is discarded.
        particle.velocity = [0.0] * len(tour)


def _step_keys(
    particle: Particle,
    neighbour: Particle,
    inertia: float,
    config: PsoConfig,
    dist: list[list[int]],
    rng: random.Random,
) -> None:
    position = particle.position
    velocity = particle.velocity
    personal = particle.best_position
    social = neighbour.best_position

    c1 = config.cognitive * rng.random() * 2.0
    c2 = config.social * rng.random() * 2.0

    velocity = [
        inertia * v + c1 * (p - x) + c2 * (g - x)
        for v, x, p, g in zip(velocity, position, personal, social)
    ]
    position = _centre([x + v for x, v in zip(position, velocity)])

    particle.velocity = velocity
    particle.position = position

    tour = keys_to_tour(position)
    length = tour_length(dist, tour)
    if length < particle.best_length:
        particle.best_length = length
        particle.best_tour = tour
        particle.best_position = list(position)


def _step_swap(
    particle: Particle,
    neighbour: Particle,
    inertia: float,
    config: PsoConfig,
    dist: list[list[int]],
    n: int,
    rng: random.Random,
) -> None:
    momentum = _scale_sequence(inertia, particle.velocity)
    cognitive = _scale_sequence(
        config.cognitive * rng.random() * 2.0,
        _swap_sequence(particle.best_position, particle.position),
    )
    social = _scale_sequence(
        config.social * rng.random() * 2.0,
        _swap_sequence(neighbour.best_position, particle.position),
    )

    velocity = momentum + cognitive + social
    if len(velocity) > MAX_VELOCITY_LEN:
        velocity = _reduce_sequence(velocity, n)
    particle.velocity = velocity

    position = list(particle.position)
    for index in velocity:
        position[index], position[index + 1] = position[index + 1], position[index]
    particle.position = position

    length = tour_length(dist, position)
    if length < particle.best_length:
        particle.best_length = length
        particle.best_tour = list(position)
        particle.best_position = list(position)


def _crossover_step(
    particle: Particle,
    neighbour: Particle,
    dist: list[list[int]],
    config: PsoConfig,
    rng: random.Random,
) -> None:
    child = _order_crossover(particle.best_tour, neighbour.best_tour, rng)
    length = tour_length(dist, child)
    if length < particle.best_length:
        _adopt(particle, child, length, config)
