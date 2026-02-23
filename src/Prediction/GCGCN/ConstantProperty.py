import json
from os.path import join as opj
from pathlib import Path
from typing import Tuple

import numpy as np
from rdkit.Chem import Descriptors, MolFromSmarts, MolFromSmiles
from rdkit import Chem
from scipy.linalg import fractional_matrix_power


FILE_DIR = Path(__file__).parent

TBN_param_list = []
TMN_param_list = []
TF_param_list = []
TC_param_list = []
PC_param_list = []
VC_param_list = []
HFORM_param_list = []
HFUS_param_list = []
for i in range(10):
    with open(opj(FILE_DIR, "parameters", f"TBN{i}.json"), "rb") as file:
        param_list = json.load(file)
        TBN_param_list.append(param_list)
    with open(opj(FILE_DIR, "parameters", f"TMN{i}.json"), "rb") as file:
        param_list = json.load(file)
        TMN_param_list.append(param_list)
    with open(opj(FILE_DIR, "parameters", f"TF{i}.json"), "rb") as file:
        param_list = json.load(file)
        TF_param_list.append(param_list)
    with open(opj(FILE_DIR, "parameters", f"TC{i}.json"), "rb") as file:
        param_list = json.load(file)
        TC_param_list.append(param_list)
    with open(opj(FILE_DIR, "parameters", f"PC{i}.json"), "rb") as file:
        param_list = json.load(file)
        PC_param_list.append(param_list)
    with open(opj(FILE_DIR, "parameters", f"VC{i}.json"), "rb") as file:
        param_list = json.load(file)
        VC_param_list.append(param_list)
    with open(opj(FILE_DIR, "parameters", f"HFORM{i}.json"), "rb") as file:
        param_list = json.load(file)
        HFORM_param_list.append(param_list)
    with open(opj(FILE_DIR, "parameters", f"HFUS{i}.json"), "rb") as file:
        param_list = json.load(file)
        HFUS_param_list.append(param_list)


SMARTS_list = [
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

smarts_omega_list = [
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


def fragment_molecule(SMILES: str) -> dict:
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
    mol = MolFromSmiles(SMILES)
    group_index = {}
    heavy_atoms = np.arange(0, Descriptors.HeavyAtomCount(mol), 1)

    if len(heavy_atoms) < 3:
        raise ValueError("Unsupported Molecule")

    # Find -NO2 group
    if mol.HasSubstructMatch(MolFromSmarts("[NX3v4;!R](=O)[OX1]")):
        Candi = mol.GetSubstructMatches(MolFromSmarts("[NX3v4;!R](=O)[OX1]"))

        for a in range(len(Candi)):
            if set(Candi[a]) & set(heavy_atoms) == set(Candi[a]):
                group_index[len(group_index)] = [Candi[a], 46]
                heavy_atoms = np.setdiff1d(heavy_atoms, Candi[a])

    # Find -COOH group
    if mol.HasSubstructMatch(MolFromSmarts("[CH0v4;!R](=O)[OX2H1]")):
        Candi = mol.GetSubstructMatches(MolFromSmarts("[CH0v4;!R](=O)[OX2H1]"))

        for a in range(len(Candi)):
            if set(Candi[a]) & set(heavy_atoms) == set(Candi[a]):
                group_index[len(group_index)] = [Candi[a], 47]
                heavy_atoms = np.setdiff1d(heavy_atoms, Candi[a])

    # Find -COO- group
    if mol.HasSubstructMatch(MolFromSmarts("[CH0v4;!R](=O)[OX2H0]")):
        Candi = mol.GetSubstructMatches(MolFromSmarts("[CH0v4;!R](=O)[OX2H0]"))

        for a in range(len(Candi)):
            if set(Candi[a]) & set(heavy_atoms) == set(Candi[a]):
                group_index[len(group_index)] = [Candi[a], 48]
                heavy_atoms = np.setdiff1d(heavy_atoms, Candi[a])

    # Find Other groups
    for index, Subgroups in enumerate(SMARTS_list):
        if mol.HasSubstructMatch(MolFromSmarts(Subgroups)):
            Candi = mol.GetSubstructMatches(MolFromSmarts(Subgroups))

            for a in range(len(Candi)):
                if set(Candi[a]) & set(heavy_atoms) == set(Candi[a]):
                    group_index[len(group_index)] = [Candi[a], index]
                    heavy_atoms = np.setdiff1d(heavy_atoms, Candi[a])

    if len(heavy_atoms) != 0:
        raise ValueError("Unsupported Molecule!")

    return group_index


def _get_input_matrices(SMILES: str, max_atom=25) -> Tuple[np.ndarray, np.ndarray]:
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
        mol = MolFromSmiles(SMILES)
    except ValueError:
        raise ValueError("Wrong SMILES format")

    if mol is None:
        raise ValueError("Molecule is not valid.")

    group_index_dict = fragment_molecule(SMILES)

    atom_to_group = {}
    for i in group_index_dict:
        for j in group_index_dict[i][0]:
            atom_to_group[j] = i

    # Get node feature matrix
    nfm = np.zeros((max_atom, len(SMARTS_list) + 3))
    for group_index in group_index_dict:
        nfm[group_index, group_index_dict[group_index][1]] = 1

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
    padding = max_atom - len(group_index_dict)
    efm = np.pad(efm, ((0, padding), (0, padding)), "constant", constant_values=0.0)

    return nfm, efm


def GCGCN(nfm: np.ndarray, efm: np.ndarray, param_list: list) -> float:
    """Predict the property of the molecule.

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
    x = np.where(x > 0, x, 0)  # ReLU

    # GC-GCN layer
    x = np.dot(efm, x)
    x = np.dot(x, param_list[2]) + param_list[3]
    x = np.where(x > 0, x, 0)  # ReLU

    # Node-wise summation
    x = x.reshape(-1)
    x = np.dot(x, param_list[4])

    # Dense layer
    x = np.dot(x, param_list[5]) + param_list[6]
    x = np.where(x > 0, x, 0)

    # Dense layer
    x = np.dot(x, param_list[7]) + param_list[8]
    x = np.where(x > 0, x, 0)

    # Dense layer
    x = np.dot(x, param_list[9]) + param_list[10]

    return x


def predict_HFORM(SMILES: str) -> Tuple[float, float]:
    """
    Predict the heat of formation of the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    Tuple[float, float]
        Predicted heat of formation and its uncertainty. (kJ/mol)
    """
    nfm, efm = _get_input_matrices(SMILES)

    result = []
    for param_list in HFORM_param_list:
        result.append(5400 * GCGCN(nfm, efm, param_list) - 4000)

    val = np.average(result)
    unc = np.std(result)

    return val, unc


def predict_HFUS(SMILES: str) -> Tuple[float, float]:
    """
    Predict the heat of fusion of the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    Tuple[float, float]
        Predicted heat of fusion and its uncertainty. (kJ/mol)
    """
    nfm, efm = _get_input_matrices(SMILES)

    result = []
    for param_list in HFUS_param_list:
        result.append(100 * GCGCN(nfm, efm, param_list))

    val = np.average(result)
    unc = np.std(result)

    return val, unc


def predict_PC(SMILES: str) -> Tuple[float, float]:
    """
    Predict the critical pressure of the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    Tuple[float, float]
        Predicted critical pressure and its uncertainty. (kPa)
    """
    nfm, efm = _get_input_matrices(SMILES)

    result = []
    for param_list in PC_param_list:
        result.append(13000 * GCGCN(nfm, efm, param_list))

    val = np.average(result)
    unc = np.std(result)

    return val, unc


def predict_TC(SMILES: str) -> Tuple[float, float]:
    """
    Predict the critical temperature of the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    Tuple[float, float]
        Predicted critical temperature and its uncertainty. (K)
    """
    nfm, efm = _get_input_matrices(SMILES)

    result = []
    for param_list in TC_param_list:
        result.append(2000 * GCGCN(nfm, efm, param_list))

    val = np.average(result)
    unc = np.std(result)

    return val, unc


def predict_TBN(SMILES: str) -> Tuple[float, float]:
    """
    Predict the boiling point of the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    Tuple[float, float]
        Predicted boiling point and its uncertainty. (K)
    """
    nfm, efm = _get_input_matrices(SMILES)

    result = []
    for param_list in TBN_param_list:
        result.append(1620 * GCGCN(nfm, efm, param_list))

    val = np.average(result)
    unc = np.std(result)

    return val, unc


def predict_TF(SMILES: str) -> Tuple[float, float]:
    """
    Predict the flash point of the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    Tuple[float, float]
        Predicted flash point and its uncertainty. (K)
    """
    nfm, efm = _get_input_matrices(SMILES)

    result = []
    for param_list in TF_param_list:
        result.append(550 * GCGCN(nfm, efm, param_list))

    val = np.average(result)
    unc = np.std(result)

    return val, unc


def predict_TMN(SMILES: str) -> Tuple[float, float]:
    """
    Predict the melting point of the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    Tuple[float, float]
        Predicted melting point and its uncertainty. (K)
    """
    nfm, efm = _get_input_matrices(SMILES)

    result = []
    for param_list in TMN_param_list:
        result.append(720 * GCGCN(nfm, efm, param_list))

    val = np.average(result)
    unc = np.std(result)

    return val, unc


def predict_VC(SMILES: str) -> Tuple[float, float]:
    """
    Predict the critical volume of the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string of the molecule.

    Returns
    -------
    Tuple[float, float]
        Predicted critical volume and its uncertainty. (L/mol)
    """
    nfm, efm = _get_input_matrices(SMILES)

    result = []
    for param_list in VC_param_list:
        result.append(1.2 * GCGCN(nfm, efm, param_list))

    val = np.average(result)
    unc = np.std(result)

    return val, unc


def get_omega_h(mol):
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
    for smarts_id, smarts in enumerate(smarts_omega_list):
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


def get_omega_a(mol):
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


def get_omega_input(smiles):
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
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("The molecule could not interpreted.")
    else:
        h = get_omega_h(mol)
        a = get_omega_a(mol)

    return h, a


def predict_omega(smiles: str) -> float:
    """Predicts the omega value for a given SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.

    Returns
    -------
    float
        Predicted acentric factor.
    """
    h, a = get_omega_input(smiles)  # Get node and edge matrices

    # Get parameters
    with open(opj(FILE_DIR, "parameters", "omega_param.json"), "rb") as file:
        omega_param = json.load(file)
    omega_param = [np.array(param) for param in omega_param]

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
