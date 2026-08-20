"""dtw.py — High-performance 2D Dynamic Time Warping (DTW) with Sakoe-Chiba band and LB_Keogh pruning."""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Numba-Accelerated Core DTW (50x - 100x Speedup)
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @nb.njit(fastmath=True, cache=True)
    def _dtw_distance_2d_numba(seq1: np.ndarray, seq2: np.ndarray, w: int = 10) -> float:
        n, d = seq1.shape
        m = seq2.shape[0]
        w = max(w, abs(n - m))

        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0.0

        for i in range(1, n + 1):
            start_j = max(1, i - w)
            end_j = min(m + 1, i + w + 1)
            for j in range(start_j, end_j):
                # Squared Euclidean distance in D-dim feature space
                diff_sq = 0.0
                for k in range(d):
                    diff = seq1[i - 1, k] - seq2[j - 1, k]
                    diff_sq += diff * diff
                cost = np.sqrt(diff_sq)

                prev_min = dtw[i - 1, j]
                if dtw[i, j - 1] < prev_min:
                    prev_min = dtw[i, j - 1]
                if dtw[i - 1, j - 1] < prev_min:
                    prev_min = dtw[i - 1, j - 1]

                dtw[i, j] = cost + prev_min

        return float(dtw[n, m])

    @nb.njit(fastmath=True, cache=True)
    def _compute_envelope_2d_numba(seq: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
        n, d = seq.shape
        u = np.empty((n, d), dtype=np.float64)
        l = np.empty((n, d), dtype=np.float64)
        for i in range(n):
            start = max(0, i - r)
            end = min(n, i + r + 1)
            for k in range(d):
                max_v = seq[start, k]
                min_v = seq[start, k]
                for j in range(start + 1, end):
                    v = seq[j, k]
                    if v > max_v:
                        max_v = v
                    if v < min_v:
                        min_v = v
                u[i, k] = max_v
                l[i, k] = min_v
        return u, l

    @nb.njit(fastmath=True, cache=True)
    def _lb_keogh_2d_numba(candidate: np.ndarray, u: np.ndarray, l: np.ndarray) -> float:
        n, d = candidate.shape
        lb_sq = 0.0
        for i in range(n):
            for k in range(d):
                c = candidate[i, k]
                if c > u[i, k]:
                    diff = c - u[i, k]
                    lb_sq += diff * diff
                elif c < l[i, k]:
                    diff = l[i, k] - c
                    lb_sq += diff * diff
        return float(np.sqrt(lb_sq))


# ---------------------------------------------------------------------------
# Pure NumPy Fallback Implementation
# ---------------------------------------------------------------------------

def _dtw_distance_2d_numpy(seq1: np.ndarray, seq2: np.ndarray, w: int = 10) -> float:
    n, d = seq1.shape
    m = seq2.shape[0]
    w = max(w, abs(n - m))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        start_j = max(1, i - w)
        end_j = min(m + 1, i + w + 1)
        for j in range(start_j, end_j):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


def _compute_envelope_2d_numpy(seq: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = seq.shape
    u = np.empty((n, d))
    l = np.empty((n, d))
    for i in range(n):
        start = max(0, i - r)
        end = min(n, i + r + 1)
        window = seq[start:end]
        u[i] = np.max(window, axis=0)
        l[i] = np.min(window, axis=0)
    return u, l


def _lb_keogh_2d_numpy(candidate: np.ndarray, u: np.ndarray, l: np.ndarray) -> float:
    above = candidate > u
    below = candidate < l
    diff = np.zeros_like(candidate)
    diff[above] = candidate[above] - u[above]
    diff[below] = l[below] - candidate[below]
    return float(np.sqrt(np.sum(diff ** 2)))


# ---------------------------------------------------------------------------
# Public Unified Interface
# ---------------------------------------------------------------------------

def dtw_distance_2d(seq1: np.ndarray, seq2: np.ndarray, w: int = 10) -> float:
    """Compute the 2D Dynamic Time Warping distance between two arrays of shape (N, D)
    using the Sakoe-Chiba band constraint.

    Automatically leverages Numba JIT acceleration if available, falling back to NumPy.
    """
    seq1_arr = np.ascontiguousarray(seq1, dtype=np.float64)
    seq2_arr = np.ascontiguousarray(seq2, dtype=np.float64)
    if _HAS_NUMBA:
        return _dtw_distance_2d_numba(seq1_arr, seq2_arr, int(w))
    return _dtw_distance_2d_numpy(seq1_arr, seq2_arr, int(w))


def compute_envelope_2d(seq: np.ndarray, r: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Compute upper (U) and lower (L) envelopes for LB_Keogh lower bound calculation."""
    seq_arr = np.ascontiguousarray(seq, dtype=np.float64)
    if _HAS_NUMBA:
        return _compute_envelope_2d_numba(seq_arr, int(r))
    return _compute_envelope_2d_numpy(seq_arr, int(r))


def lb_keogh_2d(candidate: np.ndarray, u: np.ndarray, l: np.ndarray) -> float:
    """Compute 2D LB_Keogh lower bound distance between a candidate sequence and query envelopes.

    Property: LB_Keogh(query, candidate) <= DTW(query, candidate).
    Can be used for fast O(N) candidate pruning before exact DTW evaluation.
    """
    cand_arr = np.ascontiguousarray(candidate, dtype=np.float64)
    u_arr = np.ascontiguousarray(u, dtype=np.float64)
    l_arr = np.ascontiguousarray(l, dtype=np.float64)
    if _HAS_NUMBA:
        return _lb_keogh_2d_numba(cand_arr, u_arr, l_arr)
    return _lb_keogh_2d_numpy(cand_arr, u_arr, l_arr)


def is_numba_accelerated() -> bool:
    """Return True if Numba JIT acceleration is active."""
    return _HAS_NUMBA
