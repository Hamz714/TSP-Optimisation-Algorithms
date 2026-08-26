"""Instance loading for the symmetric TSP.

Two on-disk formats are supported:

``explicit``
    A plain distance matrix: ``SIZE = n,`` followed by comma separated integers and
    terminated by a ``NOTE =`` tag. The matrix may be stored full, upper triangular, or
    strict upper triangular; the layout is inferred from how many values are present.

``tsplib``
    TSPLIB95 ``.tsp`` files with an ``EDGE_WEIGHT_TYPE`` of ``EUC_2D`` or ``CEIL_2D``.
    Distances follow the TSPLIB definition exactly, which matters because the published
    optimal tour lengths are only reproducible under that rounding rule.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Instance:
    """A symmetric TSP instance held as a dense integer distance matrix."""

    name: str
    n: int
    dist: list[list[int]]
    coords: list[tuple[float, float]] | None = None
    optimum: int | None = None
    source: Path | None = None
    _neighbours: dict[int, list[list[int]]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"{self.name}: an instance needs at least 2 cities, got {self.n}")
        if len(self.dist) != self.n:
            raise ValueError(f"{self.name}: matrix has {len(self.dist)} rows, expected {self.n}")
        for i, row in enumerate(self.dist):
            if len(row) != self.n:
                raise ValueError(
                    f"{self.name}: row {i} has {len(row)} entries, expected {self.n}"
                )

    def neighbours(self, k: int) -> list[list[int]]:
        """The ``k`` nearest cities to each city, nearest first.

        Cached per ``k``. Building the lists is O(n^2 log n), which is not something to
        repeat once per algorithm run.
        """
        k = min(k, self.n - 1)
        if k not in self._neighbours:
            self._neighbours[k] = neighbour_lists(self, k)
        return self._neighbours[k]


def neighbour_lists(instance: Instance, k: int) -> list[list[int]]:
    """Build candidate lists: for each city, its ``k`` nearest neighbours, nearest first."""
    n = instance.n
    k = min(k, n - 1)
    dist = instance.dist
    lists: list[list[int]] = []
    for i in range(n):
        row = dist[i]
        order = sorted((j for j in range(n) if j != i), key=row.__getitem__)
        lists.append(order[:k])
    return lists


def _parse_explicit(text: str, name: str) -> tuple[int, list[list[int]]]:
    compact = re.sub(r"\s+", "", text)
    header = re.search(r"SIZE=(\d+),", compact)
    if header is None:
        raise ValueError(f"{name}: no 'SIZE = n,' header found")
    n = int(header.group(1))

    body = compact[header.end():]
    note = body.find("NOTE=")
    if note != -1:
        body = body[:note]
    values = [int(token) for token in body.split(",") if token]

    strict_upper = n * (n - 1) // 2
    upper = n * (n + 1) // 2
    full = n * n

    dist = [[0] * n for _ in range(n)]
    if len(values) == strict_upper:
        it = iter(values)
        for i in range(n):
            for j in range(i + 1, n):
                dist[i][j] = dist[j][i] = next(it)
    elif len(values) == upper:
        it = iter(values)
        for i in range(n):
            for j in range(i, n):
                value = next(it)
                dist[i][j] = dist[j][i] = value
        for i in range(n):
            dist[i][i] = 0
    elif len(values) == full:
        it = iter(values)
        for i in range(n):
            for j in range(n):
                dist[i][j] = next(it)
    else:
        raise ValueError(
            f"{name}: found {len(values)} distances, which matches neither a strict upper "
            f"triangular ({strict_upper}), upper triangular ({upper}), nor full ({full}) "
            f"matrix for n = {n}"
        )
    return n, dist


def _euclidean_matrix(coords: list[tuple[float, float]], rounding: str) -> list[list[int]]:
    """Distance matrix under the TSPLIB rounding rules.

    ``EUC_2D`` rounds to the nearest integer, ``CEIL_2D`` rounds up. Reproducing this
    exactly is what makes the published optima comparable.
    """
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        row = dist[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            raw = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            value = int(raw + 0.5) if rounding == "EUC_2D" else math.ceil(raw)
            row[j] = value
            dist[j][i] = value
    return dist


def _parse_tsplib(
    text: str, name: str
) -> tuple[str, int, list[list[int]], list[tuple[float, float]]]:
    spec: dict[str, str] = {}
    coords: list[tuple[float, float]] = []
    in_coords = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("EOF"):
            break
        if upper.startswith("NODE_COORD_SECTION"):
            in_coords = True
            continue
        if in_coords:
            parts = line.split()
            coords.append((float(parts[1]), float(parts[2])))
        elif ":" in line:
            key, value = line.split(":", 1)
            spec[key.strip().upper()] = value.strip()

    weight_type = spec.get("EDGE_WEIGHT_TYPE", "").upper()
    if weight_type not in ("EUC_2D", "CEIL_2D"):
        raise ValueError(
            f"{name}: EDGE_WEIGHT_TYPE {weight_type or 'missing'} is not supported "
            f"(this parser handles EUC_2D and CEIL_2D)"
        )
    if not coords:
        raise ValueError(f"{name}: no NODE_COORD_SECTION found")

    declared = int(spec.get("DIMENSION", len(coords)))
    if declared != len(coords):
        raise ValueError(
            f"{name}: DIMENSION says {declared} but {len(coords)} coordinates were read"
        )

    return spec.get("NAME", name), len(coords), _euclidean_matrix(coords, weight_type), coords


def _known_optima() -> dict[str, int]:
    path = DATA_ROOT / "tsplib" / "optima.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_instance(path: str | Path) -> Instance:
    """Load an instance, picking the parser from the file extension."""
    path = Path(path)
    text = path.read_text()
    stem = path.stem

    if path.suffix.lower() == ".tsp":
        name, n, dist, coords = _parse_tsplib(text, stem)
        return Instance(
            name=name,
            n=n,
            dist=dist,
            coords=coords,
            optimum=_known_optima().get(name),
            source=path,
        )

    n, dist = _parse_explicit(text, stem)
    return Instance(name=stem, n=n, dist=dist, source=path)


def read_tsplib_tour(path: str | Path) -> list[int]:
    """Read a TSPLIB ``.opt.tour`` file, converting 1-based city ids to 0-based."""
    tour: list[int] = []
    in_section = False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("TOUR_SECTION"):
            in_section = True
            continue
        if not in_section:
            continue
        for token in line.split():
            value = int(token)
            if value == -1:
                return tour
            tour.append(value - 1)
    return tour


def all_instances(kind: str = "all") -> list[Path]:
    """Paths to the bundled instances."""
    paths: list[Path] = []
    if kind in ("all", "explicit"):
        paths += sorted((DATA_ROOT / "explicit").glob("*.txt"))
    if kind in ("all", "tsplib"):
        paths += sorted((DATA_ROOT / "tsplib").glob("*.tsp"))
    return paths
