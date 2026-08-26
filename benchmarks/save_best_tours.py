"""Save the best tour found for each instance as a committed artefact.

Runs the strongest configuration on every instance and keeps the shorter of the two, so the
repository carries an actual solution alongside every reported number rather than a length
the reader has to take on trust. Each tour is validated and its length recomputed before it
is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsp.algorithms import resolve
from tsp.instance import all_instances, load_instance
from tsp.tour import assert_valid_tour, tour_length, write_tour

OUT = ROOT / "results" / "tours"


def main() -> int:
    bounds_path = ROOT / "benchmarks" / "results" / "bounds.json"
    bounds = json.loads(bounds_path.read_text()) if bounds_path.exists() else {}

    for path in all_instances():
        instance = load_instance(path)
        budget = 60.0 if instance.n > 200 else 10.0

        best = None
        for name in ("aco", "pso"):
            result = resolve(name)(instance, budget, 0)
            assert_valid_tour(result.tour, instance.n, name)
            assert tour_length(instance.dist, result.tour) == result.length
            if best is None or result.length < best.length:
                best = result

        entry = bounds.get(instance.name, {})
        reference = entry.get("published_optimum") or entry.get("bound")
        meta = {
            "algorithm": best.algorithm,
            "seed": best.seed,
            "seconds": f"{best.seconds:.1f}",
        }
        if reference:
            meta["reference"] = reference
            meta["gap percent"] = f"{100.0 * (best.length - reference) / reference:.2f}"
            if best.length <= reference:
                meta["status"] = "optimal"

        write_tour(OUT / f"{instance.name}.tour", instance, best.tour, best.length, meta)
        print(f"{instance.name:10s} {best.algorithm:4s} {best.length:8d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
