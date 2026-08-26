"""Shared time and iteration budget, plus the convergence trace.

Every search in this package is anytime: it is stopped by a wall-clock budget rather than
by convergence, and it reports the best tour found so far whenever it is stopped. Putting
that in one place means the time-varying schedules can be driven by :meth:`Budget.progress`
rather than by an iteration counter, which is important because the iteration cap is never
the binding constraint in practice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    """A wall-clock (and optionally iteration) limit on a search.

    ``progress`` is the fraction of the budget consumed, clamped to [0, 1]. Schedules such
    as linearly decreasing inertia weight are driven by it so that they complete exactly
    once regardless of how many iterations the instance size allows.
    """

    time_limit: float = 10.0
    max_iterations: int | None = None

    started_at: float = field(default=0.0, init=False)
    iterations: int = field(default=0, init=False)
    trace: list[tuple[float, int]] = field(default_factory=list, init=False)
    time_to_best: float = field(default=0.0, init=False)

    def start(self) -> "Budget":
        self.started_at = time.perf_counter()
        self.iterations = 0
        self.trace = []
        self.time_to_best = 0.0
        return self

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def progress(self) -> float:
        """Fraction of the budget consumed, clamped to [0, 1]."""
        if self.time_limit <= 0:
            return 1.0
        fraction = self.elapsed / self.time_limit
        if self.max_iterations:
            fraction = max(fraction, self.iterations / self.max_iterations)
        return min(1.0, max(0.0, fraction))

    def exhausted(self) -> bool:
        if self.elapsed >= self.time_limit:
            return True
        return self.max_iterations is not None and self.iterations >= self.max_iterations

    def tick(self) -> None:
        self.iterations += 1

    def record(self, length: int) -> None:
        """Record a new incumbent. Called only when the global best actually improves."""
        self.time_to_best = self.elapsed
        self.trace.append((self.time_to_best, length))
