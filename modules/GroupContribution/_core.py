import re
from typing import Tuple, List, Dict

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.linalg import fractional_matrix_power


__all__ = ["get_molecular_descriptor", "get_molecular_graph", "sort_groups"]


AtomIdx = int
GroupIdx = int
MatchedAtoms = Tuple[AtomIdx, ...]


def _map_groups(
    mol: Chem.Mol,
    groups: List[str],
    strict: bool = True,
) -> Dict[int, Tuple[MatchedAtoms, GroupIdx]]:
    """
    Map functional groups to atom indices in a molecule.

    This function identifies substructure matches of the given SMARTS patterns
    within a molecule and assigns each matched group to a unique group index.
    Each atom may belong to at most one group; once atoms are assigned, they
    are excluded from further matches.

    In strict mode, the function raises an exception if any heavy atoms in the
    molecule are not covered by the provided functional groups.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object to be analyzed.
    groups : list of str
        List of SMARTS patterns representing functional groups.
    strict : bool, optional
        If True, require that all heavy atoms in the molecule are matched by
        one of the provided functional groups. If unmatched heavy atoms remain,
        an exception is raised. Default is True.

    Returns
    -------
    dict[int, tuple[MatchedAtoms, GroupIdx]]
        Mapping from internal group index to a tuple containing:
        - MatchedAtoms : tuple of int
            Atom indices corresponding to the matched functional group.
        - GroupIdx : int
            Index of the functional group in the input ``groups`` list.

    Raises
    ------
    ValueError
        If ``mol`` is None.
    ValueError
        If any SMARTS pattern in ``groups`` is invalid.
    Exception
        If ``strict=True`` and the molecule is not fully covered by the provided
        functional groups.

    Notes
    -----
    - Atom assignment is greedy and depends on the order of ``groups``.
      Earlier SMARTS patterns take precedence over later ones.
    - Only heavy atoms (non-hydrogen atoms) are considered when checking
      coverage in strict mode.
    - This function assumes that substructure matches returned by RDKit
      are tuples of atom indices.

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> groups = ["C", "O"]
    >>> _map_groups(mol, groups)
    {0: ((0,), 0), 1: ((1,), 0), 2: ((2,), 1)}
    """
    if mol is None:
        raise ValueError("Mol is None")

    group_index: Dict[int, Tuple[MatchedAtoms, GroupIdx]] = {}
    heavy_atoms: set[int] = {atom.GetIdx() for atom in mol.GetHeavyAtoms()}

    for group_idx, smarts in enumerate(groups):
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS: {smarts}")

        if mol.HasSubstructMatch(patt):
            for match in mol.GetSubstructMatches(patt):
                match_set = set(match)
                if match_set.issubset(heavy_atoms):
                    group_index[len(group_index)] = (match, group_idx)
                    heavy_atoms -= match_set

    if strict and len(heavy_atoms) != 0:
        raise Exception(
            "Unmatched atoms remain; molecule is not fully covered by the provided functional groups"
        )

    return group_index


def get_molecular_descriptor(
    SMILES: str,
    groups: List[str],
    strict=True,
    sanitize_mol=True,
) -> List[int]:
    """
    Generate a molecular descriptor vector based on functional group occurrences.

    This function parses a SMILES string into an RDKit molecule, identifies
    functional groups using predefined SMARTS patterns, and returns a descriptor
    vector where each element represents the count of a corresponding functional
    group in the molecule.

    Parameters
    ----------
    SMILES : str
        SMILES string representing the molecule.
    groups : list of str
        List of SMARTS patterns defining functional groups.
    strict : bool, optional
        If True, require that all heavy atoms in the molecule are covered by
        the provided functional groups. If unmatched heavy atoms remain,
        an exception is raised. Default is True.
    sanitize_mol : bool, optional
        Whether to sanitize the molecule during SMILES parsing.
        This flag is passed directly to ``Chem.MolFromSmiles``.
        Default is True.

    Returns
    -------
    list of int
        Descriptor vector is length of group types, where each element
        corresponds to the number of times the associated functional group
        appears in the molecule.

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed into a valid RDKit molecule.
    ValueError
        If ``strict=True`` and the molecule is not fully covered by the
        provided functional groups.
    ValueError
        If any SMARTS pattern in ``groups`` is invalid.

    Notes
    -----
    - Functional group assignment is performed by the internal
      ``_map_groups`` function.
    - The descriptor vector preserves the order of functional groups
      as provided in the input ``groups`` list.
    - The matching process is greedy and order-dependent; earlier SMARTS
      patterns take precedence over later ones.
    """
    mol = Chem.MolFromSmiles(SMILES, sanitize=sanitize_mol)

    if mol is None:
        raise ValueError("Mol is None")

    group_descriptor = [0] * len(groups)
    group_index_dict = _map_groups(mol, groups, strict=strict)

    for i in group_index_dict:
        group_descriptor[group_index_dict[i][1]] += 1

    return group_descriptor


def get_molecular_graph(
    SMILES: str,
    groups: List[str],
    max_nodes=30,
    normalize=True,
    self_loop=True,
    strict=True,
    sanitize_mol=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a molecule into a graph representation based on functional groups.

    This function parses a SMILES string into an RDKit molecule, identifies
    functional groups using SMARTS patterns, and constructs a graph where
    each node represents a functional group. Edges represent connectivity
    between groups based on inter-atomic bonds.

    The graph is returned as a node feature matrix and an edge (adjacency)
    matrix. Both matrices are padded to ``max_nodes``.

    Parameters
    ----------
    SMILES : str
        SMILES string representing the molecule.
    groups : list of str
        List of SMARTS patterns defining functional groups.
    max_nodes : int, optional
        Maximum number of functional group nodes allowed in the graph.
        If the number of detected groups exceeds this value, an exception
        is raised. Default is 30.
    normalize : bool, optional
        Whether to apply symmetric normalization to the edge feature matrix
        using :math:`D^{-1/2} A D^{-1/2}`. Default is True.
    self_loop : bool, optional
        Whether to add self-loops to the adjacency matrix
        (i.e., set diagonal elements to 1). Default is True.
    strict : bool, optional
        If True, require that all heavy atoms in the molecule are covered by
        the provided functional groups. Default is True.
    sanitize_mol : bool, optional
        Whether to sanitize the molecule during SMILES parsing.
        This flag is passed directly to ``Chem.MolFromSmiles``.
        Default is True.

    Returns
    -------
    node_features : numpy.ndarray
        Node feature matrix of shape ``(max_nodes, len(groups))``.
        Each row corresponds to a functional group node encoded as a one-hot
        vector over the provided functional groups.
    edge_features : numpy.ndarray
        Edge feature matrix of shape ``(max_nodes, max_nodes)``.
        The matrix is optionally normalized and padded with zeros.

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed into a valid RDKit molecule.
    ValueError
        If the number of functional group nodes exceeds ``max_nodes``.
    ValueError
        If ``strict=True`` and the molecule is not fully covered by the
        provided functional groups.
    ValueError
        If any SMARTS pattern in ``groups`` is invalid.

    Notes
    -----
    - Each functional group is treated as a single graph node.
    - Edges are weighted by the number of inter-group atomic bonds.
    - Functional group assignment is greedy and order-dependent; earlier
      SMARTS patterns take precedence over later ones.
    - Node and edge matrices are zero-padded to ``max_nodes``.
    """

    mol = Chem.MolFromSmiles(SMILES, sanitize=sanitize_mol)
    group_index_dict = _map_groups(mol, groups, strict=strict)

    if len(group_index_dict) > max_nodes:
        raise ValueError(
            f"number of functional group nodes ({len(group_index_dict)}) "
            f"exceeds max_nodes ({max_nodes})"
        )

    atom_to_group = {}
    for i in group_index_dict:
        for j in group_index_dict[i][0]:
            atom_to_group[j] = i

    # Get node feature matrix
    nfm = np.zeros((max_nodes, len(groups)))
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
    if self_loop:
        np.fill_diagonal(efm, 1)

    # Normalize the edge feature matrix
    if normalize:
        degmat = np.diag(np.sum(efm, axis=1))
        degmat_m0p5 = fractional_matrix_power(degmat, -0.5)
        efm = np.matmul(np.matmul(degmat_m0p5, efm), degmat_m0p5)

    # Padding the edge feature matrix to the maximum atom 25
    padding = max_nodes - len(group_index_dict)
    efm = np.pad(efm, ((0, padding), (0, padding)), "constant", constant_values=0.0)

    return nfm, efm


def _compute_sort_key(smarts: str) -> Tuple:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"Invalid SMARTS: {smarts}")

    # 1. Ring / aromaticity
    have_ring = False
    if re.search(r"(?<!!)R;|;R(?![^\[]*])", smarts):
        have_ring = True
    if re.search(r"(?<![A-Z])[cno]", smarts):
        have_ring = True

    # 2. Heavy atoms
    num_heavy_atoms = patt.GetNumHeavyAtoms()

    # 3. Molecular weight (heavy atoms)
    mw = Descriptors.HeavyAtomMolWt(patt)

    # 4. Available connections
    num_available_connections = 0
    for atom in patt.GetAtoms():
        if atom.IsInRing() or atom.GetIsAromatic():
            have_ring = True

        atom_smarts = re.sub(r"\$\([^)]*\)", "", atom.GetSmarts())

        m = re.search(r"X(\d+)", atom_smarts)
        max_connections = int(m.group(1)) if m else 1

        m = re.search(r"H(\d+)", atom_smarts)
        hydrogens = int(m.group(1)) if m else 0

        num_available_connections += max_connections - hydrogens - atom.GetDegree()

    # 5. Double bonds
    num_double_bonds = sum(
        1 for b in patt.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
    )

    # 6. Triple bonds
    num_triple_bonds = sum(
        1 for b in patt.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE
    )

    return (
        -int(have_ring),
        -num_heavy_atoms,
        -mw,
        num_available_connections,
        -num_double_bonds,
        -num_triple_bonds,
    )


def sort_groups(groups: List[str]) -> List[str]:
    """
    Sort SMARTS patterns according to chemically motivated priority rules.

    The sorting criteria are applied in the following order (higher priority first):
    1. Ring or aromaticity presence (ring-containing first)
    2. Number of heavy atoms (larger first)
    3. Heavy-atom molecular weight (larger first)
    4. Number of available connections (smaller first)
    5. Number of double bonds (larger first)
    6. Number of triple bonds (larger first)

    Parameters
    ----------
    groups : list of str
        List of SMARTS patterns to be sorted.

    Returns
    -------
    list of str
        Sorted list of SMARTS patterns.
    """

    return sorted(groups, key=_compute_sort_key)
