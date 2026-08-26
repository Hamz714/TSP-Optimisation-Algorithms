"""Command line interface: ``python -m tsp <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tsp.algorithms import ABLATION, ALGORITHMS, resolve
from tsp.bounds import held_karp_bound
from tsp.instance import all_instances, load_instance
from tsp.tour import assert_valid_tour, tour_length, write_tour


def _reference(instance, bounds: dict[str, int]) -> tuple[int | None, str]:
    if instance.optimum:
        return instance.optimum, "optimum"
    if instance.name in bounds:
        return bounds[instance.name], "lower bound"
    return None, ""


def _load_bounds() -> dict[str, int]:
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "bounds.json"
    if not path.exists():
        return {}
    return {name: entry["bound"] for name, entry in json.loads(path.read_text()).items()}


def cmd_solve(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance)
    runner = resolve(args.algo)

    result = runner(instance, args.time, args.seed)
    assert_valid_tour(result.tour, instance.n, args.algo)

    recomputed = tour_length(instance.dist, result.tour)
    if recomputed != result.length:
        print(
            f"internal error: reported length {result.length} but the tour measures {recomputed}",
            file=sys.stderr,
        )
        return 1

    reference, kind = _reference(instance, _load_bounds())

    print(f"instance    {instance.name} (n = {instance.n})")
    print(f"algorithm   {result.algorithm}  seed {result.seed}")
    print(f"tour length {result.length}")
    if reference:
        print(f"reference   {reference} ({kind})")
        print(f"gap         {result.gap_percent(reference):.2f} percent")
    print(f"iterations  {result.iterations}")
    print(f"time        {result.seconds:.2f}s, best found at {result.time_to_best:.2f}s")

    if args.out:
        path = Path(args.out) / f"{instance.name}.{result.algorithm}.tour"
        write_tour(
            path,
            instance,
            result.tour,
            result.length,
            {
                "algorithm": result.algorithm,
                "seed": result.seed,
                "seconds": f"{result.seconds:.2f}",
            },
        )
        print(f"written     {path}")

    if args.print_tour:
        print(",".join(str(city) for city in result.tour))

    return 0


def cmd_bound(args: argparse.Namespace) -> int:
    instance = load_instance(args.instance)
    result = held_karp_bound(instance, time_limit=args.time)

    print(f"instance    {instance.name} (n = {instance.n})")
    print(f"lower bound {result.bound}")
    print(f"best tour   {result.upper_bound}")
    print(f"gap         {result.gap_percent:.2f} percent")
    if result.optimal:
        print("note        the 1-tree closed into a tour, so this bound is exactly optimal")
    if instance.optimum:
        status = "ok" if result.bound <= instance.optimum else "INVALID"
        print(f"published   {instance.optimum} (bound is {status})")
    print(f"iterations  {result.iterations} in {result.seconds:.2f}s")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("instances")
    for path in all_instances():
        instance = load_instance(path)
        optimum = f"optimum {instance.optimum}" if instance.optimum else "no published optimum"
        print(f"  {str(path.relative_to(Path.cwd())):34s} n = {instance.n:4d}   {optimum}")
    print()
    print("algorithms")
    for name in ALGORITHMS:
        print(f"  {name}")
    print()
    print("ablation configurations")
    for name in ABLATION:
        if name not in ALGORITHMS:
            print(f"  {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tsp",
        description="Ant Colony and Particle Swarm Optimisation for the travelling salesman problem.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="run one algorithm on one instance")
    solve.add_argument("--instance", required=True, help="path to a .txt or .tsp instance")
    solve.add_argument("--algo", default="aco", help="algorithm name (see 'list')")
    solve.add_argument("--time", type=float, default=10.0, help="time budget in seconds")
    solve.add_argument("--seed", type=int, default=0, help="random seed")
    solve.add_argument("--out", help="directory to write the tour into")
    solve.add_argument("--print-tour", action="store_true", help="print the tour itself")
    solve.set_defaults(func=cmd_solve)

    bound = sub.add_parser("bound", help="compute a Held-Karp lower bound")
    bound.add_argument("--instance", required=True)
    bound.add_argument("--time", type=float, default=30.0)
    bound.set_defaults(func=cmd_bound)

    listing = sub.add_parser("list", help="list bundled instances and algorithm names")
    listing.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
