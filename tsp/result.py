"""Common return type for every search in the package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """The outcome of one run of one algorithm on one instance."""

    tour: list[int]
    length: int
    algorithm: str
    instance: str
    seed: int = 0
    iterations: int = 0
    seconds: float = 0.0
    time_to_best: float = 0.0
    trace: list[tuple[float, int]] = field(default_factory=list, repr=False)

    def gap_percent(self, reference: int | None) -> float | None:
        """Percentage above a reference length, which may be an optimum or a lower bound."""
        if not reference:
            return None
        return 100.0 * (self.length - reference) / reference
