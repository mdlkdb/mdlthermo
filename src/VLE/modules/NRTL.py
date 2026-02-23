"""
Deprecated Code
"""

import numpy as np
import matplotlib.pyplot as plt

import vapor_pressure.equations as vp


def NRTL(x1, x2, T, a12, a21, b12, b21, c):
    tow12 = a12 + b12 / (T + 273.15)
    tow21 = a21 + b21 / (T + 273.15)

    x = [x1, x2]
    tow = np.array([[0, tow12], [tow21, 0]])
    G = np.exp(-c * tow)

    gamma = np.zeros(2)
    for i in range(2):
        a = 0
        for j in range(2):
            a += x[j] * tow[j][i] * G[j][i]
        b = 0
        for k in range(2):
            b += x[j] * G[j][i]
        c = a / b
        d = 0
        for j in range(2):
            e = 0
            for k in range(2):
                e += x[k] * G[k][j]
            f = x[j] * G[i][j] / e

            g = 0
            for m in range(2):
                g += x[m] * tow[m][j] * G[m][j]
            h = 0
            for k in range(2):
                h += x[k] * G[k][j]

            d += f * (tow[i][j] - g / h)

        gamma[i] = np.exp(d + c)

    return gamma


def Modifed_Raults_Law(T, others):
    c1 = others["c1"]
    c2 = others["c2"]
    x1 = others["x1"]
    x2 = others["x2"]
    activity_model = others["activity_model"]
    Psat = others["vapor_pressure_model"]

    r12, r21 = activity_model(
        x1, x2, T, 7.79716106, -0.107372719, -3131.20054, 1540.90442, 0.3
    )
    _left = x1 * r12 * Psat(c1["CASRN"], T)
    _right = x2 * r21 * Psat(c2["CASRN"], T)

    return _left + _right - 101.325


def Bisection(
    xl: float,
    xu: float,
    func,
    max_iter: int = 10000,
    tol: float = 0.0001,
    **kwargs,
) -> float:
    for _iter in range(max_iter):
        xr = (xl + xu) / 2

        if len(kwargs) == 0:
            ff = func(xl) * func(xr)
        else:
            ff = func(xl, kwargs) * func(xr, kwargs)
        if abs(ff) < tol:
            break
        elif _iter == max_iter - 1:
            xr = 0.0
        elif ff < 0:
            xu = xr
        elif ff > 0:
            xl = xr

    return xr
