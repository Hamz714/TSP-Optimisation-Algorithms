"""Constructive heuristics.

These serve two purposes: they are the published baselines that the metaheuristics are
measured against, and they supply the reference tour length used to scale the initial
pheromone level in :mod:`tsp.aco`.
"""

from __future__ import annotations

import random

from tsp.instance import Instance
from tsp.tour import tour_length


def nearest_neighbour(instance: Instance, start: int = 0) -> tuple[list[int], int]:
    """Nearest neighbour tour from ``start``.

    Scans the candidate list first and only falls back to a full scan when every candidate
    is already visited, which keeps the common case near O(k) per step instead of O(n).
    """
    n = instance.n
    dist = instance.dist
    neighbours = instance.neighbours(min(20, n - 1))

    tour = [start]
    unvisited = set(range(n))
    unvisited.discard(start)
    current = start
    length = 0

    while unvisited:
        nxt = -1
        for candidate in neighbours[current]:
            if candidate in unvisited:
                nxt = candidate
                break
        if nxt == -1:
            row = dist[current]
            nxt = min(unvisited, key=row.__getitem__)
        length += dist[current][nxt]
        tour.append(nxt)
        unvisited.discard(nxt)
        current = nxt

    length += dist[current][start]
    return tour, length


def best_nearest_neighbour(
    instance: Instance, starts: int | None = None, rng: random.Random | None = None
) -> tuple[list[int], int]:
    """Best nearest neighbour tour over a sample of start cities.

    Trying every start is O(n) runs, which on the larger instances costs a noticeable slice
    of the search budget before the search has begun. Sampling a bounded number of starts
    gives an almost identical reference length for a fraction of the cost.
    """
    n = instance.n
    if starts is None:
        starts = min(n, 16)
    starts = max(1, min(starts, n))

    if starts >= n:
        candidates = list(range(n))
    else:
        rng = rng or random.Random(0)
        candidates = rng.sample(range(n), starts)

    best_tour: list[int] = []
    best_length = -1
    for start in candidates:
        tour, length = nearest_neighbour(instance, start)
        if best_length < 0 or length < best_length:
            best_tour, best_length = tour, length
    return best_tour, best_length


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def greedy_edge(instance: Instance) -> tuple[list[int], int]:
    """Greedy edge matching: repeatedly add the shortest edge that keeps a Hamiltonian path
    set valid, that is, no city exceeds degree 2 and no subtour closes early.

    Typically a few percent better than nearest neighbour, at the cost of sorting O(n^2)
    edges.
    """
    n = instance.n
    dist = instance.dist

    edges = [(dist[i][j], i, j) for i in range(n) for j in range(i + 1, n)]
    edges.sort()

    degree = [0] * n
    adjacency: list[list[int]] = [[] for _ in range(n)]
    union = _UnionFind(n)
    added = 0

    for _, i, j in edges:
        if added == n - 1:
            break
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        if not union.union(i, j):
            continue
        degree[i] += 1
        degree[j] += 1
        adjacency[i].append(j)
        adjacency[j].append(i)
        added += 1

    ends = [city for city in range(n) if degree[city] < 2]
    if len(ends) == 2:
        a, b = ends
        adjacency[a].append(b)
        adjacency[b].append(a)

    tour = [0]
    previous = -1
    current = 0
    for _ in range(n - 1):
        nxt = adjacency[current][0]
        if nxt == previous and len(adjacency[current]) > 1:
            nxt = adjacency[current][1]
        tour.append(nxt)
        previous, current = current, nxt

    return tour, tour_length(dist, tour)
