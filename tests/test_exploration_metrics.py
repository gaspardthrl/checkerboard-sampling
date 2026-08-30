"""Tests for checkerboard exploration metrics."""

import math

import numpy as np
import pytest

from utils.exploration_metrics import CheckerboardEvaluator, knn_entropy, temper


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_knn_entropy_recovers_gaussian_entropy(rng):
    for std in (1.0, 0.3):
        x = rng.normal(scale=std, size=(40_000, 2))
        target = math.log(2 * math.pi * math.e * std**2)
        assert knn_entropy(x) == pytest.approx(target, abs=0.02)


def test_tempering_preserves_zero_mass_cells():
    assert temper([0.5, 0.5, 0.0], [1.0, 1.0, 1.0], 0.0) == pytest.approx(
        [0.5, 0.5, 0.0]
    )


def test_checkerboard_uniform_active_tiles_hits_target(rng):
    evaluator = CheckerboardEvaluator(4)
    points = rng.random((200_000, 2)) * 2 - 1
    points = points[evaluator.on_support(points)]
    result = evaluator.evaluate(points)
    assert result["validity"] == 1.0
    assert result["entropy"] == pytest.approx(evaluator.target_entropy, abs=0.03)
    assert result["tv_to_uniform"] < 0.02


def test_checkerboard_records_out_of_box_mass(rng):
    evaluator = CheckerboardEvaluator(4)
    points = np.vstack([rng.random((5_000, 2)) * 2 - 1, rng.normal(scale=3, size=(5_000, 2))])
    result = evaluator.evaluate(points)
    assert result["oob"] > 0.3
    assert result["validity"] < 0.5
