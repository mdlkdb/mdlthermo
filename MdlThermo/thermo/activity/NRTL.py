from typing import Iterable

import numpy as np
from numpy.typing import NDArray


# def NRTL(x1, x2, T, activity_parameters: list[float]):
#     if len(activity_parameters) > 5:
#         raise ValueError("Too many activity parameters provided (expected at most 5)")

#     a12, a21, b12, b21, c = (activity_parameters + [0] * 5)[:5]

#     tow12 = a12 + b12 / (T + 273.15)
#     tow21 = a21 + b21 / (T + 273.15)

#     x = [x1, x2]
#     tow = np.array([[0, tow12], [tow21, 0]])
#     G = np.exp(-c * tow)

#     gamma = np.zeros(2)
#     for i in range(2):
#         a = 0
#         for j in range(2):
#             a += x[j] * tow[j][i] * G[j][i]
#         b = 0
#         for k in range(2):
#             b += x[j] * G[j][i]
#         c = a / b
#         d = 0
#         for j in range(2):
#             e = 0
#             for k in range(2):
#                 e += x[k] * G[k][j]
#             f = x[j] * G[i][j] / e

#             g = 0
#             for m in range(2):
#                 g += x[m] * tow[m][j] * G[m][j]
#             h = 0
#             for k in range(2):
#                 h += x[k] * G[k][j]

#             d += f * (tow[i][j] - g / h)

#         gamma[i] = np.exp(d + c)

#     return gamma


# def Modifed_Raults_Law(T, others):
#     c1 = others["c1"]
#     c2 = others["c2"]
#     x1 = others["x1"]
#     x2 = others["x2"]
#     activity_model = others["activity_model"]
#     Psat = others["vapor_pressure_model"]

#     r12, r21 = activity_model(
#         x1, x2, T, 7.79716106, -0.107372719, -3131.20054, 1540.90442, 0.3
#     )
#     _left = x1 * r12 * Psat(c1["CASRN"], T)
#     _right = x2 * r21 * Psat(c2["CASRN"], T)

#     return _left + _right - 101.325


def NRTL(
    x: Iterable[float],
    T: float,
    a: Iterable[float],
    b: Iterable[float] | None = None,
    alpha: float = 0.3,
) -> NDArray:
    """
    General multicomponent NRTL activity coefficient calculation.

    Parameters
    ----------
    x : iterable of float, shape (N,)
        Liquid mole fractions.
    T : float
        Temperature [K].
    a : iterable of iterable of float, shape (N, N)
        Base binary interaction parameter matrix.
    b : iterable of iterable of float, shape (N, N), optional
        Temperature-dependent binary interaction parameter matrix.
        tau_ij = a_ij + b_ij / T
    alpha : iterable of iterable of float, shape (N, N), or float
        Non-randomness parameter. If scalar, applied to all off-diagonal pairs.

    Returns
    -------
    np.ndarray
        Activity coefficients gamma, shape (N,)
    """

    _x = np.asarray(x, dtype=float)
    _a = np.asarray(a, dtype=float)

    if _x.ndim != 1:
        raise ValueError("`x` must be a 1D array.")

    if _a.ndim != 2 or _a.shape[0] != _a.shape[1]:
        raise ValueError("`a` must be a square (N, N) matrix.")

    if len(_x) != _a.shape[0]:
        raise ValueError(
            f"Length of `x` must match matrix size. Got len(x)={len(x)}, N={_a.shape[0]}."
        )

    if np.any(_x < 0):
        raise ValueError("All mole fractions must be non-negative.")

    if T <= 0:
        raise ValueError("`T` must be positive.")

    n = len(_x)
    if b is None:
        _b = np.zeros((n, n), dtype=float)
    else:
        _b = np.asarray(b, dtype=float)
        if _b.shape != (n, n):
            raise ValueError(f"`b` must have shape ({n}, {n}).")

    alpha_matrix = np.full((n, n), alpha, dtype=float)
    np.fill_diagonal(alpha_matrix, 0.0)

    # Common convention: diagonal = 0
    a = a.copy()
    b = b.copy()
    alpha_matrix = alpha_matrix.copy()
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(b, 0.0)
    np.fill_diagonal(alpha_matrix, 0.0)

    x_sum = x.sum()
    if x_sum <= 0:
        raise ValueError("Sum of mole fractions must be positive.")
    if not np.isclose(x_sum, 1.0, atol=1e-10):
        x = x / x_sum

    # tau_ij = a_ij + b_ij / T
    tau = a + b / T

    # G_ij = exp(-alpha_ij * tau_ij)
    G = np.exp(-alpha_matrix * tau)

    # term1_i = sum_j(x_j * tau_ji * G_ji) / sum_k(x_k * G_ki)
    TG = tau * G
    denom_i = x @ G
    numer_i = x @ TG
    term1 = numer_i / denom_i

    # A_j = sum_m(x_m * tau_mj * G_mj) / sum_k(x_k * G_kj)
    A = (x @ TG) / (x @ G)

    # term2_i = sum_j [ x_j * G_ij / sum_k(x_k * G_kj) * (tau_ij - A_j) ]
    term2 = np.sum(
        x[np.newaxis, :] * G / (x @ G)[np.newaxis, :] * (tau - A[np.newaxis, :]),
        axis=1,
    )

    ln_gamma = term1 + term2
    gamma = np.exp(ln_gamma)

    return gamma
