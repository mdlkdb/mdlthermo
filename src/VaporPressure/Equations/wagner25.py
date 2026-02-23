import json
from typing import Tuple, Dict
from os.path import join as opj
from pathlib import Path

import numpy as np
from rdkit import Chem

FILE_DIR = Path(__file__).parent

with open(opj(FILE_DIR, "wagner25_parameters.json"), "r") as f:
    wagner25_parameters = json.load(f)


def wagner25(
    T: float,
    Tc: float,
    lnPr: float,
    A: float,
    B: float = 0.0,
    C: float = 0.0,
    D: float = 0.0,
) -> float:
    """
    Input
    ------
      Temperature (K)\n
      Critical Temperature (K)\n
      lnPr (kPa)
      A\n
      B\n
      C\n
      D\n

    Return
    -------
      Psat (kPa)
    """

    if B == "":
        B = 0.0
    if C == "":
        C = 0.0
    if D == "":
        D = 0.0

    tow = 1 - T / Tc
    lnP = Tc / T * (A * tow + B * tow**1.5 + C * tow**2.5 + D * tow**5) + lnPr

    return np.exp(lnP)


def retrieve_parameter(InChIKey: str) -> dict:

    return wagner25_parameters[InChIKey]


def get_temp_range(InChIKey1: str, InChIKey2: str) -> Tuple[float, float]:
    param1 = retrieve_parameter(InChIKey1)
    param2 = retrieve_parameter(InChIKey2)

    Tmin = max(param1["Tmin"], param2["Tmin"])
    Tmax = min(param1["Tmax"], param2["Tmax"])

    return Tmin, Tmax


def cal_vapor_pressure(SMILES: str, T: float, check_temperature=False) -> float:
    """
    Input
    ------
      CASRN: CAS Registry Number\n
      T: Temperature (K)
    Return
    -------
      Psat(T) (kPa)
    """

    mol = Chem.MolFromSmiles(SMILES)
    InChIKey = Chem.inchi.MolToInchiKey(mol)
    try:
        parameters = retrieve_parameter(InChIKey)
    except:
        raise Exception()

    if check_temperature:
        if parameters["Tmin"] > T or parameters["Tmax"] < T:
            raise Exception()

    return wagner25(
        T,
        parameters["Tc"],
        parameters["lnPr"],
        parameters["A"],
        parameters["B"],
        parameters["C"],
        parameters["D"],
    )
