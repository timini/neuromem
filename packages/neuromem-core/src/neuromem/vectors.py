"""numpy-backed vector math for neuromem.

This module is the contract layer's numerical helpers. It exposes:

- ``cosine_similarity`` — scalar cosine between two vectors.
- ``batch_cosine_similarity`` — vectorised cosine between one query
  vector and an ``(N, D)`` matrix of candidates. Used by
  ``SQLiteAdapter.get_nearest_nodes`` for the SC-003 performance path.
- ``compute_centroid`` — element-wise mean of a batch of vectors,
  used by the agglomerative clustering loop in the dreaming cycle.

All three functions accept either numpy arrays or Python lists for
flexibility, and normalise to ``np.float64`` internally for numerical
stability. Inputs are never mutated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


VectorLike = "NDArray[np.floating] | list[float]"
VectorBatch = "NDArray[np.floating] | list[NDArray[np.floating]] | list[list[float]]"


def cosine_similarity(vec_a: VectorLike, vec_b: VectorLike) -> float:
    """Return the cosine similarity between two vectors.

    Result is in ``[-1.0, 1.0]``. For non-negative embedding models
    the practical range is ``[0.0, 1.0]``.

    Returns ``0.0`` (not NaN) if either vector has zero magnitude.

    Raises ``ValueError`` on shape mismatch or empty input.
    """
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("empty vector")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def batch_cosine_similarity(
    query: NDArray[np.floating],
    matrix: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return cosine similarity between ``query`` and every row of ``matrix``.

    - ``query`` is a 1-D array of shape ``(D,)``.
    - ``matrix`` is a 2-D array of shape ``(N, D)``.
    - Result is a 1-D ``np.float64`` array of shape ``(N,)``.

    Rows with zero magnitude yield ``0.0`` (not NaN). A zero-magnitude
    query yields an all-zero result vector.

    Raises ``ValueError`` on dimension mismatch.
    """
    q = np.asarray(query, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {q.shape}")
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {m.shape}")
    if q.shape[0] != m.shape[1]:
        raise ValueError(f"dim mismatch: query {q.shape[0]} vs matrix {m.shape[1]}")

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return np.zeros(m.shape[0], dtype=np.float64)

    row_norms = np.linalg.norm(m, axis=1)
    denom = row_norms * q_norm
    # Guard against zero-magnitude rows: compute dot / 1 for those slots,
    # then mask back to 0 so the result is 0.0 not NaN.
    safe_denom = np.where(denom == 0.0, 1.0, denom)
    sims = (m @ q) / safe_denom
    sims[row_norms == 0.0] = 0.0
    return sims


def compute_centroid(vectors: VectorBatch) -> NDArray[np.float64]:
    """Return the element-wise mean of a batch of vectors.

    Accepts either a 2-D ``(N, D)`` array or a list of 1-D vectors
    (Python lists or numpy arrays). Returns a new 1-D ``np.float64``
    array of length ``D``. Does not mutate inputs.

    Raises ``ValueError`` on empty input or length-mismatched lists.
    """
    if isinstance(vectors, list) and len(vectors) == 0:
        raise ValueError("vectors is empty")

    try:
        arr = np.asarray(vectors, dtype=np.float64)
    except ValueError as exc:
        # np.asarray on a ragged list-of-lists raises this.
        raise ValueError(f"vectors have mismatched lengths: {exc}") from exc

    if arr.size == 0:
        raise ValueError("vectors is empty")
    if arr.ndim == 1:
        return arr.copy()
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D input, got shape {arr.shape}")
    return arr.mean(axis=0)
