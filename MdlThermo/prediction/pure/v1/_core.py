import json
from pathlib import Path
from typing import Literal, Any, cast
from functools import cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.linalg import fractional_matrix_power


__all__ = [
    "embed_smiles",
    "predict_IG_Hform",
    "predict_Hfus",
    "predict_Pc",
    "predict_Vc",
    "predict_Tc",
    "predict_Tb",
    "predict_Tm",
    "predict_Tf",
]


PARAMETER_DIR = Path(__file__).parents[1]
SMARTS_LIST = [
    "[CX4H3]",
    "[CX3H2v4]",
    "[CX2H1v4]",
    "[!R;CX4H2]",
    "[!R;CX4H]",
    "[!R;CX4H0]",
    "[!R;CX3H1v4]",
    "[!R;CX3H0v4]",
    "[!R;CX2H0;$([CX2H0](=*)(=*))]",
    "[!R;CX2H0;$([CX2H0](#*))]",
    "[R;CX4H2]",
    "[R;CX4H]",
    "[R;CX4H0]",
    "[R;CX3H1v4]",
    "[R;CX3H0v4]",
    "[R;CX2H0;$([CX2H0](=*)(=*))]",
    "[R;CX2H0;$([CX2H0](#*))]",
    "[R;cX4h2]",
    "[R;cX4h]",
    "[R;cX3h1v4]",
    "[R;cX3h0v4]",
    "[FX1H0]",
    "[ClX1H0]",
    "[BrX1H0]",
    "[IX1H0]",
    "[OX2H1]",
    "[!R;OX2H0]",
    "[R;OX2H0]",
    "[R;oX2h0]",
    "[OX1H0v2]",
    "[NX3H2v3]",
    "[NX3H1v3;!R]",
    "[NX3H1v3;R]",
    "[nX3h1v3;R]",
    "[NX3H0v3;!R]",
    "[NX3H0v3;R]",
    "[nX3h0v3;R]",
    "[NX2H0v3;!R]",
    "[NX2H0v3;R]",
    "[nX2h0v3;R]",
    "[NX1H0v3]",
    "[SX2H1v2]",
    "[SX2H0v2;!R]",
    "[SX2H0v2;R]",
    "[sX2h0v2;R]",
    "[SX1H0v2]",
]


@cache
def _load_parameters(code: str):
    parameters_list = []
    for i in range(10):
        with open(PARAMETER_DIR / "parameters" / f"{code}{i}.json", "rb") as file:
            parameters = json.load(file)
            parameters_list.append(parameters)

    return parameters_list


def _fragment_molecule(SMILES: str) -> dict:
    """
    Fragment the molecule into groups.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    dict
        Dictionary of the groups.
    """
    mol = Chem.MolFromSmiles(SMILES)
    group_index = {}
    heavy_atoms = np.arange(0, Descriptors.HeavyAtomCount(mol), 1)

    if len(heavy_atoms) < 3:
        raise ValueError("Unsupported Molecule")

    # Find -NO2 group
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3v4;!R](=O)[OX1]")):
        Candi = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3v4;!R](=O)[OX1]"))
        for a in range(len(Candi)):
            if set(Candi[a]) & set(heavy_atoms) == set(Candi[a]):
                group_index[len(group_index)] = [Candi[a], 46]
                heavy_atoms = np.setdiff1d(heavy_atoms, Candi[a])

    # Find -COOH group
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CH0v4;!R](=O)[OX2H1]")):
        Candi = mol.GetSubstructMatches(Chem.MolFromSmarts("[CH0v4;!R](=O)[OX2H1]"))
        for a in range(len(Candi)):
            if set(Candi[a]) & set(heavy_atoms) == set(Candi[a]):
                group_index[len(group_index)] = [Candi[a], 47]
                heavy_atoms = np.setdiff1d(heavy_atoms, Candi[a])

    # Find -COO- group
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CH0v4;!R](=O)[OX2H0]")):
        Candi = mol.GetSubstructMatches(Chem.MolFromSmarts("[CH0v4;!R](=O)[OX2H0]"))
        for a in range(len(Candi)):
            if set(Candi[a]) & set(heavy_atoms) == set(Candi[a]):
                group_index[len(group_index)] = [Candi[a], 48]
                heavy_atoms = np.setdiff1d(heavy_atoms, Candi[a])

    # Find Other groups
    for index, Subgroups in enumerate(SMARTS_LIST):
        if mol.HasSubstructMatch(Chem.MolFromSmarts(Subgroups)):
            Candi = mol.GetSubstructMatches(Chem.MolFromSmarts(Subgroups))
            for a in range(len(Candi)):
                if set(Candi[a]) & set(heavy_atoms) == set(Candi[a]):
                    group_index[len(group_index)] = [Candi[a], index]
                    heavy_atoms = np.setdiff1d(heavy_atoms, Candi[a])

    if len(heavy_atoms) != 0:
        raise ValueError("Unsupported Molecule!")

    return group_index


NodeLength = Literal[25]
FeatureLength = Literal[64]


class V1GCGCNInput:
    nfm: np.ndarray[tuple[NodeLength, FeatureLength], np.dtype[np.float64]]
    efm: np.ndarray[tuple[NodeLength, NodeLength], np.dtype[np.float64]]


def embed_smiles(SMILES: str) -> V1GCGCNInput:
    """
    Convert the molecule to input matrices.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.
    max_atom : int, optional
        Maximum number of atoms in the molecule. The default is 25.

    Raises
    ------
    ValueError
        If the molecule is not valid.

    Returns
    -------
    nfm : numpy.ndarray
        Node feature matrix.
    efm : numpy.ndarray
        Edge feature matrix.
    """
    # Check if the molecule is valid
    try:
        mol = Chem.MolFromSmiles(SMILES)
    except ValueError:
        raise ValueError("Wrong SMILES format")

    if mol is None:
        raise ValueError("Molecule is not valid.")

    group_index_dict = _fragment_molecule(SMILES)

    atom_to_group = {}
    for i in group_index_dict:
        for j in group_index_dict[i][0]:
            atom_to_group[j] = i

    # Get node feature matrix
    nfm = cast(
        np.ndarray[tuple[NodeLength, FeatureLength], np.dtype[np.float64]],
        np.zeros((25, 49), dtype=np.float64),
    )
    for group_index in group_index_dict:
        nfm[group_index, group_index_dict[group_index][1]] = 1.0

    # Get edge feature matrix
    efm = np.zeros((len(group_index_dict), len(group_index_dict)))
    for group_index in group_index_dict:
        atom_idxs = group_index_dict[group_index][0]

        for atom_idx in atom_idxs:
            target_atom = mol.GetAtomWithIdx(atom_idx)

            for neigbor_atom in target_atom.GetNeighbors():
                neigbor_atom_idx = neigbor_atom.GetIdx()

                if neigbor_atom_idx not in atom_idxs:
                    efm[group_index, atom_to_group[neigbor_atom_idx]] += 1

    # Fill diagonal to 1 to connect the group to itself
    np.fill_diagonal(efm, 1)

    # Normalize the edge feature matrix
    degmat = np.diag(np.sum(efm, axis=1))
    degmat_m0p5 = fractional_matrix_power(degmat, -0.5)
    efm = np.matmul(np.matmul(degmat_m0p5, efm), degmat_m0p5)

    # Padding the edge feature matrix to the maximum atom 25
    padding = 25 - len(group_index_dict)
    efm = np.pad(efm, ((0, padding), (0, padding)), "constant", constant_values=0.0)

    x = V1GCGCNInput()
    x.nfm = nfm
    x.efm = efm

    return x


def GCGCN(
    nfm: np.ndarray[tuple[NodeLength, FeatureLength], np.dtype[np.float64]],
    efm: np.ndarray[tuple[NodeLength, NodeLength], np.dtype[np.float64]],
    param_list: list[Any],
) -> float:
    """
    Predict the property of the molecule.

    Parameters
    ----------
    nfm : numpy.ndarray
        Node feature matrix.
    efm : numpy.ndarray
        Edge feature matrix.
    param_list : list
        Parameter list.

    Returns
    -------
    float
        Predicted property.
    """
    # GC-GCN layer
    x = np.dot(efm, nfm)
    x = np.dot(x, param_list[0]) + param_list[1]
    x = np.where(x > 0, x, 0)

    # GC-GCN layer
    x = np.dot(efm, x)
    x = np.dot(x, param_list[2]) + param_list[3]
    x = np.where(x > 0, x, 0)

    # Node-wise summation
    x = np.sum(x, axis=0)

    # Dense layer
    x = np.dot(x, param_list[5]) + param_list[6]
    x = np.where(x > 0, x, 0)

    # Dense layer
    x = np.dot(x, param_list[7]) + param_list[8]
    x = np.where(x > 0, x, 0)

    # Dense layer
    x = np.dot(x, param_list[9]) + param_list[10]

    return x


def predict_IG_Hform(_x: V1GCGCNInput) -> tuple[float, float]:
    parameters_list = _load_parameters("HFORM")
    results: list[float] = []
    for parameters in parameters_list:
        results.append(5400 * GCGCN(_x.nfm, _x.efm, parameters) - 4000)
    val = np.average(results).item()
    unc = np.std(results).item()

    return val, unc


def predict_Hfus(_x: V1GCGCNInput) -> tuple[float, float]:
    parameters_list = _load_parameters("HFUS")
    results: list[float] = []
    for parameters in parameters_list:
        results.append(100 * GCGCN(_x.nfm, _x.efm, parameters))
    val = np.average(results).item()
    unc = np.std(results).item()

    return val, unc


def predict_Pc(_x: V1GCGCNInput) -> tuple[float, float]:
    parameters_list = _load_parameters("PC")
    results: list[float] = []
    for parameters in parameters_list:
        results.append(13000 * GCGCN(_x.nfm, _x.efm, parameters))
    val = np.average(results).item()
    unc = np.std(results).item()

    return val, unc


def predict_Tc(_x: V1GCGCNInput) -> tuple[float, float]:
    parameters_list = _load_parameters("TC")
    results: list[float] = []
    for parameters in parameters_list:
        results.append(2000 * GCGCN(_x.nfm, _x.efm, parameters))
    val = np.average(results).item()
    unc = np.std(results).item()

    return val, unc


def predict_Vc(_x: V1GCGCNInput) -> tuple[float, float]:
    parameters_list = _load_parameters("VC")
    results: list[float] = []
    for parameters in parameters_list:
        results.append(1.2 * GCGCN(_x.nfm, _x.efm, parameters))
    val = np.average(results).item()
    unc = np.std(results).item()

    return val, unc


def predict_Tb(_x: V1GCGCNInput) -> tuple[float, float]:
    parameters_list = _load_parameters("TBN")
    results: list[float] = []
    for parameters in parameters_list:
        results.append(1620 * GCGCN(_x.nfm, _x.efm, parameters))
    val = np.average(results).item()
    unc = np.std(results).item()

    return val, unc


def predict_Tf(_x: V1GCGCNInput) -> tuple[float, float]:
    parameters_list = _load_parameters("TF")
    results: list[float] = []
    for parameters in parameters_list:
        results.append(550 * GCGCN(_x.nfm, _x.efm, parameters))
    val = np.average(results).item()
    unc = np.std(results).item()

    return val, unc


def predict_Tm(_x: V1GCGCNInput) -> tuple[float, float]:
    parameters_list = _load_parameters("TMN")
    results: list[float] = []
    for parameters in parameters_list:
        results.append(720 * GCGCN(_x.nfm, _x.efm, parameters))
    val = np.average(results).item()
    unc = np.std(results).item()

    return val, unc
