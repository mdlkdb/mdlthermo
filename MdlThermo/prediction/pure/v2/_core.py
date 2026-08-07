import json
from pathlib import Path
from functools import cache

import numpy as np
from rdkit import Chem
from scipy.linalg import fractional_matrix_power


__all__ = ["embed_smiles", "predict_omega"]


PARAMETER_DIR = Path(__file__).parents[1]
OMEGA_SMARTS_LIST = [
    "[CX4v4]",
    "[CX3v4;$([CX3v4](=*))]",
    "[CX2v4;$([CX2v4](=*)(=*))]",
    "[CX2v4;$([CX2v4](#*))]",
    "[C-v3]",
    "[c]",
    "[NX3v3]",
    "[NX2v3;$([NX2v3](=*))]",
    "[NX1v3;$([NX1v3](#*))]",
    "[N+v4]",
    "[N-v2]",
    "[n+0]",
    "[OX2v2]",
    "[OX1v2;$([OX1v2](=*))]",
    "[O+v3]",
    "[O-v1]",
    "[o]",
    "[Sv2]",
    "[Sv4]",
    "[Sv6]",
    "[s]",
    "[Pv3]",
    "[Pv5]",
    "[FX1v1]",
    "[ClX1v1]",
    "[BrX1v1]",
    "[IX1v1]",
    "[Siv3]",
    "[Siv4]",
]


class V2GCGCNInput:
    nfm: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    efm: np.ndarray[tuple[int, int], np.dtype[np.float64]]


@cache
def _load_paremeters():
    with open(PARAMETER_DIR / "parameters" / "omega_param.json", "rb") as file:
        omega_param = json.load(file)
    omega_param = [np.array(param) for param in omega_param]

    return omega_param


def _get_omega_h(mol: Chem.Mol):
    """Get the feature matrix of the molecule.

    The index of the feature matrix corresponds to each group of the group
    contribution method. That is, the feature matrix is an integer encoding in
    which the number of groups constituting a molecule is counted.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule read by rdkit.

    Returns
    -------
    h : np.ndarray of shape (25, 36)
        The (node) feature matrix of the molecule.

    Raises
    ------
    ValueError : If the molecule is too large (heavy atom > 25), or if
        fragmentation failed.
    """
    h = np.zeros((25, 29 + 6))  # num. groups (29) + num. H (5) + ring (1)

    # If the number of heavy atoms of the molecule is more than 25,
    if mol.GetNumAtoms() > 25:
        raise ValueError("The molecule is too large.")

    # Find the SMARTS patterns
    for smarts_id, smarts in enumerate(OMEGA_SMARTS_LIST):
        atom_id = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        atom_id = np.array(atom_id).flatten()

        if atom_id.size > 0:  # If atom_id is not empty,
            h[np.array(atom_id), smarts_id] = 1

    # 분자의 기초 그룹 합이 전체 heavy atom 수와 안맞으면 fragmentation 실패.
    if not np.sum(h) == mol.GetNumHeavyAtoms():
        raise ValueError("Fragmentation failed.")

    # 원자의 수소 개수와 고리 여부를 찾는다.
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()

        # 수소 개수를 구한다.
        h[atom_id, 29 + atom.GetTotalNumHs()] = 1

        # Aliphatic 고리 구조 여부를 구한다.
        if atom.IsInRing() and not atom.GetIsAromatic():
            h[atom_id, 29 + 6 - 1] = 1

    return h


def _get_omega_a(mol: Chem.Mol):
    """Get the normalized adjacency matrix of the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule read by rdkit.

    Returns
    -------
    a_norm : np.ndarray of shape (25, 25)
        Normalized adjacency matrix of the molecule.
    """
    # Calculate the adjacency matrix A.
    a = Chem.GetAdjacencyMatrix(mol).astype("float64")

    # Fill the diagonal of A with 1 for self-loops.
    np.fill_diagonal(a, 1)

    # Calculate the D^(-0.5) matrix for normalization.
    d = np.diag(np.sum(a, axis=1))
    d_sqrt_inv = fractional_matrix_power(d, -0.5)

    # Compute the normalized adjacency matrix A^(~) = D^(-0.5) A D^(-0.5).
    a_norm = np.matmul(np.matmul(d_sqrt_inv, a), d_sqrt_inv)

    # Pad the matrix with zeros to match the size of max_atom.
    pad = 25 - len(a)

    a_norm = np.pad(a_norm, ((0, pad), (0, pad)), "constant", constant_values=0.0)
    return a_norm


def embed_smiles(SMILES: str) -> V2GCGCNInput:
    """Get the matrices of the molecule.

    Parameters
    ----------
    smiles : str
        The SMILES representation of the molecule.

    Returns
    -------
    Tuple of the matrices (h, a)
        - h : np.ndarray of shape (25, 33)
            The (node) feature matrix of the molecule.
        - a : np.ndarray of shape (25, 25)
            Normalized adjacency matrix of the molecule.

    Raises
    ------
    ValueError : If the SMILES is not interpreted by rdkit.

    See Also
    --------
    get_h, get_a
    """
    mol = Chem.MolFromSmiles(SMILES)

    if mol is None:
        raise ValueError("The molecule could not interpreted.")
    else:
        h = _get_omega_h(mol)
        a = _get_omega_a(mol)

    x = V2GCGCNInput()
    x.nfm = h
    x.efm = a

    return x


def predict_omega(_x: V2GCGCNInput) -> float:
    """
    Predicts the omega value for a given SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.

    Returns
    -------
    float
        Predicted acentric factor.
    """
    h = _x.nfm
    a = _x.efm
    omega_param = _load_paremeters()

    # GCN Layer 1
    x = np.dot(np.dot(a, h), omega_param[0].T) + omega_param[1]

    # Layer normalization on features (axis=-1 if on last dim)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 1e-5)
    x = x * omega_param[2] + omega_param[3]

    # LeakyReLU activation
    x = np.where(x > 0, x, 0.001 * x)

    # GCN Layer 2
    x = np.dot(np.dot(a, x), omega_param[4].T) + omega_param[5]

    # Layer normalization on features
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 1e-5)
    x = x * omega_param[6] + omega_param[7]

    # LeakyReLU activation
    x = np.where(x > 0, x, 0.001 * x)

    # GCN Layer 3
    x = np.dot(np.dot(a, x), omega_param[8].T) + omega_param[9]

    # Layer normalization on features
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 1e-5)
    x = x * omega_param[10] + omega_param[11]

    # LeakyReLU activation
    x = np.where(x > 0, x, 0.001 * x)

    # Pooling (average over nodes)
    x = np.mean(x, axis=0)

    # Dense Layer 1
    x = np.dot(x, omega_param[12].T) + omega_param[13]

    # Layer normalization
    mean = np.mean(x, keepdims=True)
    variance = np.var(x, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 1e-5)
    x = x * omega_param[14] + omega_param[15]

    # LeakyReLU activation
    x = np.where(x > 0, x, 0.001 * x)

    # Dense Layer 2
    x = np.dot(x, omega_param[16].T) + omega_param[17]

    # Layer normalization
    mean = np.mean(x, keepdims=True)
    variance = np.var(x, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 1e-5)
    x = x * omega_param[18] + omega_param[19]

    # LeakyReLU activation
    x = np.where(x > 0, x, 0.001 * x)

    # Output Layer
    x = np.dot(x, omega_param[20].T) + omega_param[21]

    # Final scaling
    return 0.55139863 + 0.23780635 * x[0]
