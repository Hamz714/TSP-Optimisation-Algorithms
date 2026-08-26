"""Shared fixtures. Adds the repository root to the path so the tests run without installing."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsp.instance import DATA_ROOT, Instance, load_instance  # noqa: E402


@pytest.fixture(scope="session")
def berlin52() -> Instance:
    return load_instance(DATA_ROOT / "tsplib" / "berlin52.tsp")


@pytest.fixture(scope="session")
def eil76() -> Instance:
    return load_instance(DATA_ROOT / "tsplib" / "eil76.tsp")


@pytest.fixture(scope="session")
def kroA100() -> Instance:
    return load_instance(DATA_ROOT / "tsplib" / "kroA100.tsp")


@pytest.fixture(scope="session")
def tsp012() -> Instance:
    return load_instance(DATA_ROOT / "explicit" / "tsp012.txt")


def random_instance(n: int, seed: int = 0, metric: bool = True) -> Instance:
    """A small synthetic instance.

    ``metric=False`` produces a matrix that violates the triangle inequality. This is worth
    testing because the explicit instances are arbitrary symmetric matrices with no guarantee
    of metricity, and several classic local search prunings are only valid on metric
    instances.
    """
    rng = random.Random(seed)
    if metric:
        points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
        dist = [
            [
                int(
                    ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                    ** 0.5
                    + 0.5
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
    else:
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                value = rng.randint(1, 1000)
                dist[i][j] = dist[j][i] = value

    return Instance(name=f"random{n}", n=n, dist=dist)
