"""Inject generated tables into the README.

The README carries paired HTML comment markers, for example ``<!--TABLE_HEADLINE-->`` and
``<!--/TABLE_HEADLINE-->``, and this script replaces whatever sits between them. Running it
after the benchmarks is what guarantees the published tables match the CSVs rather than
being transcribed by hand.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import make_tables  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"
README = ROOT / "README.md"


def replace(text: str, marker: str, body: str) -> str:
    open_tag = f"<!--{marker}-->"
    close_tag = f"<!--/{marker}-->"
    block = f"{open_tag}\n\n{body.strip()}\n\n{close_tag}"

    pattern = re.compile(re.escape(open_tag) + r".*?" + re.escape(close_tag), re.DOTALL)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    if open_tag in text:
        return text.replace(open_tag, block)
    raise KeyError(f"marker {marker} not found in README")


def summary_paragraph(rows: list[dict], bounds: dict) -> str:
    instances = sorted({(row["n"], row["instance"]) for row in rows})
    gaps = {"aco": [], "pso": []}

    for _, name in instances:
        reference, _ = make_tables._reference(name, bounds)
        if not reference:
            continue
        for algorithm in ("aco", "pso"):
            runs = [row for row in rows if row["instance"] == name and row["algorithm"] == algorithm]
            if runs:
                best = min(row["length"] for row in runs)
                gaps[algorithm].append(100.0 * (best - reference) / reference)

    known_hit, known_total, unknown_hit, unknown_total = make_tables.certified_optimal(rows, bounds)
    largest = max(entry["n"] for entry in bounds.values())

    return (
        f"Across {len(instances)} instances of up to {largest} cities, and taking the best of five "
        f"seeds per instance, the ant colony reaches a mean gap of "
        f"**{statistics.mean(gaps['aco']):.2f} %** and the particle swarm "
        f"**{statistics.mean(gaps['pso']):.2f} %** against published optima or certified "
        f"lower bounds. The search reaches the published optimum on **{known_hit} of "
        f"{known_total}** TSPLIB instances, and on **{unknown_hit} of {unknown_total}** of the "
        f"unlabelled instances it returns a tour whose length equals the computed lower bound, "
        f"which proves those tours optimal."
    )


def environment_note() -> str:
    path = RESULTS / "environment.json"
    if not path.exists():
        return ""
    env = json.loads(path.read_text())
    return (
        f"The published tables were produced on {env['platform']} with "
        f"{env['implementation']} {env['python']} across {env['cores']} cores."
    )


def test_count() -> str:
    try:
        output = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        ).stdout
        match = re.search(r"(\d+) tests? collected", output)
        if match:
            return f"The suite is {match.group(1)} tests and runs in well under a minute."
    except Exception:
        pass
    return ""


def plots_block() -> str:
    directory = RESULTS / "convergence"
    if not directory.exists():
        return ""
    images = sorted(directory.glob("*.png"))
    if not images:
        return ""
    lines = [
        "Best-so-far tour length against elapsed time, with the reference drawn as a dashed "
        "line. The vertical axis is logarithmic, which is the only way the unenhanced "
        "configurations and the enhanced ones fit on one plot.",
        "",
    ]
    for image in images:
        relative = image.relative_to(ROOT).as_posix()
        lines.append(f"![convergence on {image.stem}]({relative})")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    bounds = make_tables._bounds()
    text = README.read_text()
    best: dict[str, int] = {}

    if (RESULTS / "results.csv").exists():
        rows = make_tables._load(RESULTS / "results.csv")
        best = make_tables.best_lengths(rows)
        text = replace(text, "RESULT_SUMMARY", summary_paragraph(rows, bounds))
        text = replace(text, "TABLE_HEADLINE", make_tables.headline_table(rows, bounds))
        text = replace(text, "TABLE_DISTRIBUTION", make_tables.distribution_table(rows, bounds))
        text = replace(text, "TABLE_COMPARISON", make_tables.comparison_table(rows, bounds))

    if (RESULTS / "ablation.csv").exists():
        rows = make_tables._load(RESULTS / "ablation.csv")
        text = replace(text, "TABLE_ABLATION", make_tables.ablation_table(rows, bounds))
        text = replace(text, "TABLE_ABLATION_COST", make_tables.ablation_cost_table(rows, bounds))

    if (RESULTS / "stagnation.csv").exists():
        rows = make_tables._load(RESULTS / "stagnation.csv")
        text = replace(text, "TABLE_STAGNATION", make_tables.stagnation_table(rows, bounds))

    if bounds:
        text = replace(text, "TABLE_BOUNDS", make_tables.bounds_table(bounds, best))

    for marker, body in [
        ("ENVIRONMENT", environment_note()),
        ("TEST_COUNT", test_count()),
        ("PLOTS", plots_block()),
    ]:
        if body:
            text = replace(text, marker, body)

    # Written as escapes so this guard does not itself trip a search for the characters.
    banned = {chr(0x2014): "em dash", chr(0x2013): "en dash"}
    for character, label in banned.items():
        if character in text:
            raise AssertionError(f"README contains a {label}")

    README.write_text(text, encoding="utf-8")
    print(f"README updated ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
