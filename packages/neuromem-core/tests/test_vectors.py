"""Unit tests for neuromem.vectors — numpy-backed vector math.

Covers cosine_similarity, batch_cosine_similarity, compute_centroid
per specs/001-neuromem-core/contracts/public-api.md §neuromem.vectors.
"""

import math

import numpy as np
import pytest
from neuromem.vectors import (
    batch_cosine_similarity,
    compute_centroid,
    cosine_similarity,
)

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_equal_vectors_returns_one(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_returns_negative_one(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_magnitude_first_vector_returns_zero(self) -> None:
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == 0.0
        assert not math.isnan(cosine_similarity(a, b))

    def test_zero_magnitude_second_vector_returns_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.zeros(3)
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_magnitude_returns_zero(self) -> None:
        assert cosine_similarity(np.zeros(3), np.zeros(3)) == 0.0

    def test_shape_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            cosine_similarity(np.zeros(3), np.zeros(4))

    def test_accepts_python_list_input(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_returns_python_float_not_numpy_scalar(self) -> None:
        result = cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)

    def test_empty_vector_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# batch_cosine_similarity
# ---------------------------------------------------------------------------


class TestBatchCosineSimilarity:
    def test_basic_batch_ranking(self) -> None:
        query = np.array([1.0, 0.0, 0.0])
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],  # identical → 1.0
                [0.5, 0.5, 0.0],  # partial → ~0.707
                [0.0, 1.0, 0.0],  # orthogonal → 0.0
                [-1.0, 0.0, 0.0],  # opposite → -1.0
            ]
        )
        sims = batch_cosine_similarity(query, matrix)
        assert sims.shape == (4,)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(math.sqrt(0.5), abs=1e-6)
        assert sims[2] == pytest.approx(0.0, abs=1e-6)
        assert sims[3] == pytest.approx(-1.0)

    def test_zero_magnitude_row_returns_zero(self) -> None:
        query = np.array([1.0, 0.0, 0.0])
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # zero row → should be 0.0, not NaN
                [0.0, 1.0, 0.0],
            ]
        )
        sims = batch_cosine_similarity(query, matrix)
        assert sims[1] == 0.0
        assert not np.isnan(sims).any()

    def test_zero_magnitude_query_returns_all_zeros(self) -> None:
        query = np.zeros(3)
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        sims = batch_cosine_similarity(query, matrix)
        assert np.all(sims == 0.0)
        assert not np.isnan(sims).any()

    def test_dimension_mismatch_raises_value_error(self) -> None:
        query = np.array([1.0, 0.0, 0.0])
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            batch_cosine_similarity(query, matrix)

    def test_returns_float64_dtype(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([[1.0, 0.0]], dtype=np.float32)
        sims = batch_cosine_similarity(query, matrix)
        assert sims.dtype == np.float64


# ---------------------------------------------------------------------------
# compute_centroid
# ---------------------------------------------------------------------------


class TestComputeCentroid:
    def test_two_dim_array_input(self) -> None:
        arr = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
            ]
        )
        centroid = compute_centroid(arr)
        np.testing.assert_allclose(centroid, [2.0, 3.0, 4.0])

    def test_list_of_lists_input(self) -> None:
        vectors = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
        centroid = compute_centroid(vectors)
        np.testing.assert_allclose(centroid, [3.0, 4.0])

    def test_list_of_numpy_arrays_input(self) -> None:
        vectors = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        centroid = compute_centroid(vectors)
        np.testing.assert_allclose(centroid, [2.0, 3.0])

    def test_single_vector_returns_copy(self) -> None:
        vectors = [np.array([1.0, 2.0, 3.0])]
        centroid = compute_centroid(vectors)
        np.testing.assert_allclose(centroid, [1.0, 2.0, 3.0])

    def test_empty_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            compute_centroid([])

    def test_length_mismatched_lists_raises_value_error(self) -> None:
        vectors = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        with pytest.raises(ValueError):
            compute_centroid(vectors)

    def test_returns_float64_dtype(self) -> None:
        vectors = [np.array([1.0, 2.0], dtype=np.float32)]
        centroid = compute_centroid(vectors)
        assert centroid.dtype == np.float64

    def test_does_not_mutate_input(self) -> None:
        original = np.array([[1.0, 2.0], [3.0, 4.0]])
        snapshot = original.copy()
        compute_centroid(original)
        np.testing.assert_array_equal(original, snapshot)


# ---------------------------------------------------------------------------
# Shape-validation branches (coverage holes filled in T030)
# ---------------------------------------------------------------------------


class TestShapeValidationBranches:
    """These tests exist to pin the explicit shape-check ``raise``
    branches in ``batch_cosine_similarity`` and ``compute_centroid``.
    They're small on purpose — the point is CI coverage of the error
    paths, not extra API surface."""

    def test_batch_cosine_rejects_2d_query(self) -> None:
        """Line 74–75: ``query must be 1-D`` when the caller passes a
        2-D array by mistake."""
        query_2d = np.array([[1.0, 0.0], [0.0, 1.0]])
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="1-D"):
            batch_cosine_similarity(query_2d, matrix)

    def test_batch_cosine_rejects_1d_matrix(self) -> None:
        """Line 76–77: ``matrix must be 2-D`` when the caller flattens
        it by mistake."""
        query = np.array([1.0, 0.0])
        matrix_1d = np.array([1.0, 0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="2-D"):
            batch_cosine_similarity(query, matrix_1d)

    def test_compute_centroid_zero_size_numpy_array_raises(self) -> None:
        """Line 113–114: ``vectors is empty`` for a zero-size numpy
        array (the early list-length check at line 104 only catches
        empty Python lists)."""
        empty_arr = np.empty((0, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="empty"):
            compute_centroid(empty_arr)

    def test_compute_centroid_accepts_1d_numpy_array(self) -> None:
        """Line 115–116: a single 1-D numpy array is returned verbatim
        (as a copy) rather than going through the ``mean(axis=0)``
        path. This matches the docstring: "Accepts either a 2-D (N,D)
        array or a list of 1-D vectors."
        """
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        centroid = compute_centroid(vec)
        np.testing.assert_array_equal(centroid, vec)
        # And it must be a copy, not a view.
        centroid[0] = 999.0
        assert vec[0] == 1.0

    def test_compute_centroid_rejects_3d_input(self) -> None:
        """Line 117–118: 3-D input is a programming error."""
        cube = np.zeros((2, 3, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            compute_centroid(cube)
