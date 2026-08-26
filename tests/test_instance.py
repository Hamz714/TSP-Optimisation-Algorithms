"""Parsing, distance matrices, and the TSPLIB rounding rule."""

from __future__ import annotations

import pytest

from tsp.instance import (
    DATA_ROOT,
    all_instances,
    load_instance,
    neighbour_lists,
    read_tsplib_tour,
)
from tsp.tour import is_valid_tour, tour_length

OPT_TOURS = sorted((DATA_ROOT / "tsplib").glob("*.opt.tour"))


@pytest.mark.parametrize("path", all_instances(), ids=lambda p: p.stem)
def test_matrix_is_square_symmetric_and_zero_diagonal(path):
    instance = load_instance(path)
    assert len(instance.dist) == instance.n
    for i in range(instance.n):
        assert len(instance.dist[i]) == instance.n
        assert instance.dist[i][i] == 0
        for j in range(i + 1, instance.n):
            assert instance.dist[i][j] == instance.dist[j][i]
            assert instance.dist[i][j] >= 0


@pytest.mark.parametrize("path", OPT_TOURS, ids=lambda p: p.name)
def test_published_optimal_tour_reproduces_published_length(path):
    """The single most important parser test.

    TSPLIB EUC_2D distances are rounded to the nearest integer *before* summing. Get that
    wrong and every optimum in the results table is measured against a different problem.
    """
    name = path.name.replace(".opt.tour", "")
    instance = load_instance(DATA_ROOT / "tsplib" / f"{name}.tsp")
    tour = read_tsplib_tour(path)

    assert is_valid_tour(tour, instance.n)
    assert tour_length(instance.dist, tour) == instance.optimum


def test_explicit_parser_reads_strict_upper_triangular(tsp012):
    assert tsp012.n == 12
    assert tsp012.dist[0][1] == 30
    assert tsp012.dist[1][0] == 30


def test_explicit_parser_accepts_full_and_upper_triangular_layouts(tmp_path):
    expected = [[0, 3, 5], [3, 0, 7], [5, 7, 0]]

    strict = tmp_path / "strict.txt"
    strict.write_text("SIZE = 3,\n3,5,\n7,\nNOTE = \n")

    upper = tmp_path / "upper.txt"
    upper.write_text("SIZE = 3,\n0,3,5,0,7,0,\nNOTE = \n")

    full = tmp_path / "full.txt"
    full.write_text("SIZE = 3,\n0,3,5,3,0,7,5,7,0,\nNOTE = \n")

    for path in (strict, upper, full):
        assert load_instance(path).dist == expected, path.name


def test_explicit_parser_rejects_a_wrong_value_count(tmp_path):
    path = tmp_path / "broken.txt"
    path.write_text("SIZE = 4,\n1,2,3,\nNOTE = \n")
    with pytest.raises(ValueError, match="matches neither"):
        load_instance(path)


def test_tsplib_parser_rejects_unsupported_edge_weight_type(tmp_path):
    path = tmp_path / "geo.tsp"
    path.write_text("NAME: x\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: GEO\nNODE_COORD_SECTION\n1 0 0\n2 1 1\nEOF\n")
    with pytest.raises(ValueError, match="not supported"):
        load_instance(path)


def test_neighbour_lists_are_sorted_and_exclude_self(berlin52):
    lists = neighbour_lists(berlin52, 8)
    assert len(lists) == berlin52.n
    for city, row in enumerate(lists):
        assert len(row) == 8
        assert city not in row
        assert len(set(row)) == 8
        distances = [berlin52.dist[city][other] for other in row]
        assert distances == sorted(distances)


def test_neighbour_lists_are_cached_per_k(berlin52):
    assert berlin52.neighbours(5) is berlin52.neighbours(5)
    assert berlin52.neighbours(5) is not berlin52.neighbours(6)


def test_neighbour_count_is_clamped_to_available_cities(tsp012):
    assert len(tsp012.neighbours(50)[0]) == tsp012.n - 1
