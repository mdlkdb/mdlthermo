import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from src import descriptor
from src.GCGCN import predict_HFORM
from src.GCGCN.modules import _get_input_matrices, fragment_molecule
from src.GCGCN._utils import GCGCN_smarts


smiles = "CCCCF"

n0, e0 = descriptor.get_molecular_graph(smiles, GCGCN_smarts, max_nodes=30)
n1, e1 = _get_input_matrices(smiles)


FILE_PATH = Path(__file__).parent / "src" / "GCGCN" / "parameters" / "HFORM0.json"
with open(FILE_PATH, "rb") as file:
    param_list = json.load(file)


def GCGCN(nfm: np.ndarray, efm: np.ndarray, nfm1, efm1, param_list: list) -> float:
    # GC-GCN layer
    x = np.dot(efm, nfm)
    y = np.dot(efm1, nfm1)

    x = np.dot(x, param_list[0]) + param_list[1]
    y = np.dot(y, param_list[0]) + param_list[1]

    x = np.where(x > 0, x, 0)
    y = np.where(y > 0, y, 0)

    # GC-GCN layer
    x = np.dot(efm, x)
    x = np.dot(x, param_list[2]) + param_list[3]
    x = np.where(x > 0, x, 0)

    y = np.dot(efm1, y)
    y = np.dot(y, param_list[2]) + param_list[3]
    y = np.where(y > 0, y, 0)

    # Node-wise summation
    x = np.sum(x, axis=0)
    y = np.sum(y, axis=0)

    # Dense layer
    x = np.dot(x, param_list[5]) + param_list[6]
    x = np.where(x > 0, x, 0)

    # Dense layer
    x = np.dot(x, param_list[7]) + param_list[8]
    x = np.where(x > 0, x, 0)

    # Dense layer
    x = np.dot(x, param_list[9]) + param_list[10]

    return x


a = GCGCN(n0, e0, n1, e1, param_list)
