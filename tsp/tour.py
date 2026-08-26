"""Tour representation, validation, and on-disk format.

A tour is a list of city indices, a permutation of ``range(n)``, interpreted cyclically:
the edge from the last city back to the first is implied and always counted.
"""

from __future__ import annotations

from pathlib import Path

from tsp.instance import Instance


def tour_length(dist: list[list[int]], tour: list[int]) -> int:
    """Total length of a closed tour, including the return edge to the start."""
    total = dist[tour[-1]][tour[0]]
    for i in range(len(tour) - 1):
        total += dist[tour[i]][tour[i + 1]]
    return total


def is_valid_tour(tour: list[int], n: int) -> bool:
    """True when ``tour`` visits every city in ``range(n)`` exactly once."""
    return len(tour) == n and sorted(tour) == list(range(n))


def assert_valid_tour(tour: list[int], n: int, context: str = "") -> None:
    """Raise if ``tour`` is not a permutation of ``range(n)``.

    Used as an internal invariant check after every algorithm returns, so a bug in a move
    operator surfaces immediately rather than as a quietly wrong tour length.
    """
    if is_valid_tour(tour, n):
        return
    prefix = f"{context}: " if context else ""
    missing = sorted(set(range(n)) - set(tour))
    duplicates = sorted({c for c in tour if tour.count(c) > 1})
    raise ValueError(
        f"{prefix}invalid tour of length {len(tour)} for n = {n} "
        f"(missing {missing[:8]}, duplicated {duplicates[:8]})"
    )


def write_tour(
    path: str | Path,
    instance: Instance,
    tour: list[int],
    length: int,
    meta: dict[str, object] | None = None,
) -> Path:
    """Write a tour in a small self-describing text format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"NAME = {instance.name}",
        f"SIZE = {instance.n}",
        f"TOUR LENGTH = {length}",
    ]
    for key, value in (meta or {}).items():
        lines.append(f"{key.upper()} = {value}")
    lines.append("TOUR =")
    lines.append(",".join(str(city) for city in tour))
    path.write_text("\n".join(lines) + "\n")
    return path


def read_tour(path: str | Path) -> tuple[list[int], dict[str, str]]:
    """Read back a tour written by :func:`write_tour`."""
    meta: dict[str, str] = {}
    tour: list[int] = []
    in_tour = False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("TOUR ="):
            in_tour = True
            continue
        if in_tour:
            tour.extend(int(token) for token in line.split(",") if token)
        elif "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return tour, meta
