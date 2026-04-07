import json
from pathlib import Path
from functools import cache
from typing import Literal

import numpy as np
from rdkit import Chem
from scipy.linalg import fractional_matrix_power


__all__ = ["embed_smiles", "predict_Vp"]


FILE_PATH = Path(__file__).parent
SMARTS_LIST = [
    "[CX4v4]",
    "[CX3v4;$([CX3v4](=*))]",
    "[CX2v4;$([CX2v4](=*)(=*))]",
    "[CX2v4;$([CX2v4](#*))]",
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
    "[p]",
    "[FX1v1]",
    "[ClX1v1]",
    "[BrX1v1]",
    "[IX1v1]",
]


class V3GCGCNInput:
    nfm: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    efm: np.ndarray[tuple[int, int], np.dtype[np.float64]]


@cache
def _load_parameters(model_name: Literal["Antoine", "Wagner", "King-Al-Najjar"]):
    if model_name == "Antoine":
        with open(FILE_PATH / "antoine_param.json", "rb") as f:
            param_list = json.load(f)
        alpha = 0.01
    elif model_name == "Wagner":
        with open(FILE_PATH / "wagner_param.json", "rb") as f:
            param_list = json.load(f)
        alpha = 0.1
    elif model_name == "King-Al-Najjar":
        with open(FILE_PATH / "kingalnajjar_param.json", "rb") as f:
            param_list = json.load(f)
        alpha = 0.01
    else:
        raise ValueError("Unsupported model name")

    return param_list, alpha


def get_pvap_h(mol: Chem.Mol):
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
    h : np.ndarray of shape (max_atom, num_feat)
        The (node) feature matrix of the molecule.

    Raises
    ------
    ValueError : If the molecule is too large (heavy atom > 25), or if
        fragmentation failed.
    """
    h = np.zeros((25, 33))

    # If the number of heavy atoms of the molecule is more than 25,
    if mol.GetNumAtoms() > 25:
        raise ValueError("The molecule is too large.")

    # Find the SMARTS patterns
    for smarts_id, smarts in enumerate(SMARTS_LIST):
        atom_id = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        atom_id = np.array(atom_id).flatten()

        if atom_id.size > 0:  # If atom_id is not empty,
            h[np.array(atom_id), smarts_id] = 1

    # If the sum of the basic groups of the molecule does not match the number
    # of heavy atoms, fragmentation failed.
    if not np.sum(h) == mol.GetNumHeavyAtoms():
        raise ValueError("Fragmentation failed.")

    # Find the number of hydrogens and the aliphatic ring structure.
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()

        # Find the number of hydrogens.
        h[atom_id, 33 - 7 + atom.GetTotalNumHs()] = 1

        # Find the aliphatic ring structure.
        if atom.IsInRing() and not atom.GetIsAromatic():
            h[atom_id, 33 - 1] = 1

    return h


def get_pvap_a(mol: Chem.Mol):
    """
    Get the normalized adjacency matrix of the molecule.

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


def embed_smiles(SMILES: str) -> V3GCGCNInput:
    """
    Get the matrices of the molecule.

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

    h = get_pvap_h(mol)
    a = get_pvap_a(mol)
    x = V3GCGCNInput()
    x.nfm = h
    x.efm = a

    return x


def pred_pvap(a, h, t, pvap_model: Literal["Antoine", "Wagner", "King-Al-Najjar"]):
    """Predicts the natural logarithm of vapor pressure.

    Predicts the natural logarithm of vapor pressure (ln P) for a compound
    using a selected vapor pressure model and graph convolutional layers.

    Parameters
    ----------
    a : numpy.ndarray of shape (25, 25)
        Adjacency matrix representing molecular structure.
    h : numpy.ndarray of shape (25, 33)
        Node features matrix.
    t : float
        Temperature at which the vapor pressure is to be predicted (K).
    pvap_model : {'Antoine', 'Wagner', 'King-Al-Najjar'}
        Vapor pressure model to use.

    Returns
    -------
    ln_p : float
        Predicted natural logarithm of vapor pressure.

    Raises
    ------
    ValueError
        If 'pvap_model' is not recognized.

    Notes
    -----
    Uses leaky ReLU activation with a specified alpha for each model.
    Applies layer normalization after each convolutional layer. The final
    output depends on the selected vapor pressure model (Antoine, Wagner,
    or King-Al-Najjar) and is calculated using model-specific equations.
    """
    param_list, alpha = _load_parameters(pvap_model)

    # Graph convolutional layer 1
    x = np.matmul(a, np.matmul(h, param_list[0])) + param_list[1]

    # LeakyReLU
    x = np.where(x > 0, x, alpha * x)

    # Layer normalization
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 0.001)

    # Graph convolutional layer 2
    x = np.matmul(a, np.matmul(x, param_list[4])) + param_list[5]

    # LeakyReLU
    x = np.where(x > 0, x, alpha * x)

    # Layer normalization
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 0.001)
    x = x * param_list[6] + param_list[7]

    # Node-wise summation
    x = np.sum(x, axis=0)

    # Multi-layer perceptron 1
    x = np.dot(x, param_list[8]) + param_list[9]

    # LeakyReLU
    x = np.where(x > 0, x, alpha * x)

    # Layer normalization
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 0.001)
    x = x * param_list[10] + param_list[11]

    # Multi-layer perceptron 2
    x = np.dot(x, param_list[12]) + param_list[13]

    # LeakyReLU
    x = np.where(x > 0, x, alpha * x)

    # Layer normalization
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + 0.001)
    x = x * param_list[14] + param_list[15]

    # Multi-layer perceptron 3
    x = np.dot(x, param_list[16]) + param_list[17]

    # LeakyReLU
    x = np.where(x > 0, x, alpha * x)

    # Vapor pressure model
    if pvap_model == "Antoine":
        ln_p = x[0] - x[1] / (t + x[2])
    elif pvap_model == "Wagner":
        tau = np.max([0.0, 1 - 0.001 * t / (x[4] + 1.5)])
        ln_p = (
            x[0] * tau + x[1] * tau**1.5 + x[2] * tau**2.5 + x[3] * tau**5
        ) / (1 - tau) + x[5]
    elif pvap_model == "King-Al-Najjar":
        tau = np.max([0.0, 1 - 0.001 * t / (x[4] + 1.5)])
        ln_p = (
            (-x[0] - x[1]) * tau**0.5
            - x[2] * tau
            - x[1] * tau**1.5
            - x[2] * (tau**2 + tau**4 + tau**6)
            + (-2 * x[0] + x[1] + x[2] + x[3]) * tau**0.5 / (1 - tau)
            + 3 * x[0] * np.arctanh(tau**0.5)
        ) + x[5]

    return ln_p


def pred_pvap_ensemble(a, h, t) -> float:
    """Predicts vapor pressure by applying an ensemble.

    Predicts vapor pressure by applying an ensemble of multiple vapor
    pressure models.

    Parameters
    ----------
    a : numpy.ndarray of shape (25, 25)
        Adjacency matrix representing molecular structure.
    h : numpy.ndarray of shape (25, 33)
        Node features matrix.
    t : float
        Temperature at which the vapor pressure is to be predicted (K).

    Returns
    -------
    p : float
        Predicted vapor pressure in kPa.

    Notes
    -----
    Combines predictions from the 'Antoine', 'Wagner', and 'King-Al-Najjar'
    models using a weighted average, then converts the result from ln(p) to
    kPa.
    """
    # Get vapor pressure prediction for each model
    ln_p1 = pred_pvap(a, h, t, "Antoine")
    ln_p2 = pred_pvap(a, h, t, "Wagner")
    ln_p3 = pred_pvap(a, h, t, "King-Al-Najjar")

    # Get ensemble
    ln_p = 0.249305638 * ln_p1 + 0.234145692 * ln_p2 + 0.51654867 * ln_p3
    p = np.exp(ln_p * np.log(10)) / 1000  # kPa

    return p


def predict_Vp(_x: V3GCGCNInput, T: float) -> float:
    h = _x.nfm
    a = _x.efm

    return pred_pvap_ensemble(a, h, T)
