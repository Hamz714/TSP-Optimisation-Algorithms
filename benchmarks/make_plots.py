"""Convergence plots.

Matplotlib is the only non-standard-library dependency anywhere in the project, and it is
needed for this script alone. The solvers themselves import nothing outside the standard
library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsp.algorithms import resolve  # noqa: E402
from tsp.instance import all_instances, load_instance  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
PLOTS = RESULTS / "convergence"

CONFIGURATIONS = ["aco", "aco-baseline", "pso", "pso-baseline"]
COLOURS = {
    "aco": "#1f6feb",
    "aco-baseline": "#8ba7d1",
    "pso": "#d1495b",
    "pso-baseline": "#e3a3ac",
}


def _reference(name: str) -> tuple[int | None, str]:
    path = RESULTS / "bounds.json"
    if not path.exists():
        return None, ""
    entry = json.loads(path.read_text()).get(name, {})
    if entry.get("published_optimum"):
        return entry["published_optimum"], "published optimum"
    if entry.get("bound"):
        return entry["bound"], "Held-Karp lower bound"
    return None, ""


def _staircase(trace: list[tuple[float, int]], limit: float) -> tuple[list[float], list[int]]:
    """A best-so-far trace is a step function, so it is drawn as one."""
    times: list[float] = []
    values: list[int] = []
    for moment, length in trace:
        if times:
            times.append(moment)
            values.append(values[-1])
        times.append(moment)
        values.append(length)
    if times:
        times.append(limit)
        values.append(values[-1])
    return times, values


def plot_instance(path: Path, time_limit: float, seed: int = 0) -> Path | None:
    instance = load_instance(path)
    reference, label = _reference(instance.name)

    figure, axes = plt.subplots(figsize=(7.2, 4.2))

    for name in CONFIGURATIONS:
        result = resolve(name)(instance, time_limit, seed)
        times, values = _staircase(result.trace, time_limit)
        if not times:
            continue
        axes.plot(times, values, label=name, color=COLOURS[name], linewidth=1.6)

    if reference:
        axes.axhline(reference, color="#2f9e44", linestyle="--", linewidth=1.2, label=label)

    axes.set_xlabel("seconds")
    axes.set_ylabel("best tour length so far")
    axes.set_title(f"{instance.name} (n = {instance.n}), seed {seed}")
    axes.set_yscale("log")
    axes.grid(True, alpha=0.25, which="both")
    axes.legend(frameon=False, fontsize=9)
    figure.tight_layout()

    PLOTS.mkdir(parents=True, exist_ok=True)
    output = PLOTS / f"{instance.name}.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render convergence plots.")
    parser.add_argument("--instances", nargs="*", default=["kroA100", "ch150", "lin318", "tsp180"])
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = {path.stem: path for path in all_instances()}
    for name in args.instances:
        output = plot_instance(paths[name], args.time, args.seed)
        print(f"written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
