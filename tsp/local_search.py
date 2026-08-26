"""2-opt and Or-opt local search over candidate lists, with don't-look bits.

The naive 2-opt neighbourhood is O(n^2) per pass and is rescanned from scratch after every
improving move, which makes it unusable inside a metaheuristic at n in the hundreds. Three
standard accelerations are applied here:

*Candidate lists.*
    An improving 2-opt move that removes edge ``(a, succ(a))`` must introduce a strictly
    shorter edge out of ``a``. So only neighbours ``b`` with ``d(a, b) < d(a, succ(a))``
    can start one. With the neighbour lists sorted by distance the scan breaks out at the
    first candidate that fails the test, which prunes almost the whole row.

*Don't-look bits.*
    A city whose entire neighbourhood has been scanned without finding a move cannot start
    an improving move again until one of its incident edges changes. Such cities are
    deactivated and only re-queued when a move touches them, so successive passes cost
    close to the number of cities actually affected rather than n.

*Position index and shorter-side reversal.*
    ``pos[city]`` gives a city's index in the tour, so orientation tests are O(1), and a
    2-opt move reverses whichever of the two arcs is shorter, halving the expected work.

Or-opt complements 2-opt by relocating short segments (1 to 3 cities), in either
orientation, to a position elsewhere in the tour. That move cannot be expressed as a single
2-opt move, so alternating the two reaches a strictly better class of local optimum.
"""

from __future__ import annotations

from collections import deque

from tsp.budget import Budget
from tsp.instance import Instance
from tsp.tour import tour_length

DEFAULT_NEIGHBOURS = 10
MAX_SEGMENT = 3


def _reverse(tour: list[int], pos: list[int], i: int, j: int) -> None:
    """Reverse the cyclic segment of ``tour`` from index ``i`` to index ``j`` inclusive.

    Reversing an arc and reversing its complement give the same cyclic tour, so the shorter
    of the two is chosen.
    """
    n = len(tour)
    inner = (j - i) % n + 1
    if 2 * inner > n:
        i, j = (j + 1) % n, (i - 1) % n
        inner = n - inner
    for _ in range(inner // 2):
        ci, cj = tour[i], tour[j]
        tour[i], tour[j] = cj, ci
        pos[cj], pos[ci] = i, j
        i = (i + 1) % n
        j = (j - 1) % n


def _apply_or_opt(
    tour: list[int], pos: list[int], start: int, seg_len: int, after: int, reverse: bool
) -> None:
    """Move the segment of ``seg_len`` cities starting at index ``start`` so that it follows
    city ``after``, optionally reversed."""
    n = len(tour)
    segment = [tour[(start + k) % n] for k in range(seg_len)]
    rest = [tour[(start + seg_len + k) % n] for k in range(n - seg_len)]
    if reverse:
        segment.reverse()
    cut = rest.index(after) + 1
    tour[:] = rest[:cut] + segment + rest[cut:]
    for index, city in enumerate(tour):
        pos[city] = index


def local_search(
    instance: Instance,
    tour: list[int],
    length: int | None = None,
    neighbours: list[list[int]] | None = None,
    budget: Budget | None = None,
    use_two_opt: bool = True,
    use_or_opt: bool = True,
) -> tuple[list[int], int]:
    """Improve ``tour`` to a 2-opt and Or-opt local optimum.

    Returns a new tour and its length. The input tour is not modified. The length is
    maintained incrementally from move deltas; :func:`tsp.tour.tour_length` recomputes it
    from scratch and the two agree exactly (there is a test for it).
    """
    n = instance.n
    dist = instance.dist
    tour = list(tour)
    if length is None:
        length = tour_length(dist, tour)
    if n < 4:
        return tour, length
    if neighbours is None:
        neighbours = instance.neighbours(DEFAULT_NEIGHBOURS)

    pos = [0] * n
    for index, city in enumerate(tour):
        pos[city] = index

    queued = [True] * n
    active = deque(tour)

    def wake(*cities: int) -> None:
        for city in cities:
            if not queued[city]:
                queued[city] = True
                active.append(city)

    check_budget = 0
    while active:
        check_budget += 1
        if budget is not None and check_budget % 32 == 0 and budget.exhausted():
            break

        a = active.popleft()
        queued[a] = False

        gain = 0

        if use_two_opt:
            gain = _two_opt_from(a, tour, pos, dist, neighbours, wake)

        if gain == 0 and use_or_opt:
            gain = _or_opt_from(a, tour, pos, dist, neighbours, wake)

        if gain > 0:
            length -= gain
            wake(a)

    return tour, length


def _two_opt_from(
    a: int,
    tour: list[int],
    pos: list[int],
    dist: list[list[int]],
    neighbours: list[list[int]],
    wake,
) -> int:
    """Try both 2-opt moves anchored at city ``a``. Returns the gain, 0 if none was found.

    ``direction = 1`` removes the edge to ``a``'s successor, ``direction = -1`` the edge to
    its predecessor. The predecessor case is the successor move applied to the pair
    ``(pred(a), pred(b))``, which is why both share one reversal call.
    """
    n = len(tour)
    row_a = dist[a]

    for direction in (1, -1):
        index_a = pos[a]
        t2 = tour[(index_a + direction) % n]
        removed = row_a[t2]

        for b in neighbours[a]:
            added = row_a[b]
            if added >= removed:
                # Neighbours are sorted, so no later candidate can start an improving move.
                break
            index_b = pos[b]
            t4 = tour[(index_b + direction) % n]
            if t4 == a:
                continue
            delta = added + dist[t2][t4] - removed - dist[b][t4]
            if delta < 0:
                if direction == 1:
                    _reverse(tour, pos, (index_a + 1) % n, index_b)
                else:
                    _reverse(tour, pos, index_a, (index_b - 1) % n)
                wake(a, t2, b, t4)
                return -delta

    return 0


def _or_opt_from(
    a: int,
    tour: list[int],
    pos: list[int],
    dist: list[list[int]],
    neighbours: list[list[int]],
    wake,
) -> int:
    """Try relocating a short segment containing city ``a``, in either orientation.

    Every segment of length 1 to 3 that contains ``a`` is considered, not only the one that
    begins at it. Scanning only forward would make the neighbourhood asymmetric: a segment
    ending at ``a`` would be found only while processing a different city, and if that
    city's don't-look bit were already set the move would be missed entirely, leaving the
    search short of a genuine local optimum.
    """
    n = len(tour)

    for seg_len in range(1, min(MAX_SEGMENT, n - 3) + 1):
        for offset in range(seg_len):
            start = (pos[a] - offset) % n
            first = tour[start]
            last = tour[(start + seg_len - 1) % n]
            before = tour[(start - 1) % n]
            after = tour[(start + seg_len) % n]
            if before == after:
                continue

            remove_gain = dist[before][first] + dist[last][after] - dist[before][after]
            if remove_gain <= 0:
                continue

            segment = {tour[(start + k) % n] for k in range(seg_len)}

            for c in neighbours[first] + neighbours[last]:
                if c in segment or c == before:
                    continue
                c_next = tour[(pos[c] + 1) % n]
                if c_next in segment:
                    continue

                broken = dist[c][c_next]
                forward = dist[c][first] + dist[last][c_next] - broken
                backward = dist[c][last] + dist[first][c_next] - broken

                reverse = backward < forward
                insert = backward if reverse else forward
                if insert < remove_gain:
                    _apply_or_opt(tour, pos, start, seg_len, c, reverse)
                    wake(first, last, before, after, c, c_next)
                    return remove_gain - insert

    return 0
