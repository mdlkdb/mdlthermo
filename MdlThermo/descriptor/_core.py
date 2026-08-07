from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from scipy.linalg import fractional_matrix_power

type SMARTS = str


class GroupMappingResult(TypedDict):
    group_idx: int
    atoms: tuple[int, ...]


# from rdkit.Chem import Descriptors

# def _count_available_bonds(mol: Chem.Mol):
#     total_num_Hs = 0
#     for atom in mol.GetAtoms():
#         try:
#             total_num_Hs += atom.GetTotalNumHs()
#         except:
#             pass

#         atom.SetNumExplicitHs(0)
#         atom.SetNoImplicit(True)

#     SMILES = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)
#     new_mol = Chem.MolFromSmiles(SMILES)
#     new_num_Hs = sum(atom.GetTotalNumHs() for atom in new_mol.GetAtoms())

#     return max(new_num_Hs - total_num_Hs, 0)


# def sort_groups(groups: list[SMARTS]) -> list[SMARTS]:
#     """Sort SMARTS group patterns by matching priority.

#     Args:
#         groups: SMARTS patterns to sort. The input list is not modified.

#     Returns:
#         A new list containing the SMARTS patterns in priority order.

#     Raises:
#         ValueError: If a pattern cannot be parsed as SMARTS.
#     """

#     def sort_key(group: SMARTS) -> tuple:
#         patt = Chem.MolFromSmarts(group)

#         if patt is None:
#             raise ValueError(f"Invalid SMARTS: {group}")

#         rw_mol = Chem.RWMol(patt)
#         num_available_bonds = _count_available_bonds(rw_mol)
#         num_heavy_atoms = rw_mol.GetNumHeavyAtoms()
#         molecular_weight = Descriptors.HeavyAtomMolWt(rw_mol)
#         has_ring = False
#         for atom in patt.GetAtoms():
#             if atom.IsInRing() or atom.GetIsAromatic():
#                 has_ring = True
#                 break
#         num_triple_bonds = sum(bond.GetBondType() == Chem.BondType.TRIPLE for bond in patt.GetBonds())
#         num_double_bonds = sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in patt.GetBonds())

#         return (
#             -num_available_bonds,
#             -num_heavy_atoms,
#             -molecular_weight,
#             -has_ring,
#             -num_triple_bonds,
#             -num_double_bonds,
#         )

#     return sorted(groups, key=sort_key)


def map_groups(
    mol: Chem.Mol,
    groups: list[SMARTS],
    *,
    strict: bool = True,
) -> list[GroupMappingResult]:
    """Greedily map non-overlapping SMARTS groups onto a molecule.

    Groups are processed in order. A match is accepted only when all atoms
    in the match are still-unassigned heavy atoms. Accepted atoms are removed
    from consideration, so earlier patterns and earlier RDKit match
    candidates take precedence.

    Args:
        mol: RDKit molecule to map.
        groups: SMARTS patterns defining the available groups.
        strict: Whether every heavy atom must be assigned to a group.
        sort: Whether to process a priority-sorted copy of ``groups``. When
            enabled, returned group indices refer to the sorted order.

    Returns:
        Mapping records in accepted-match order. Each record contains
        ``group_idx``, the zero-based pattern index, and ``atoms``, a tuple
        of zero-based RDKit atom indices.

    Raises:
        ValueError: If strict mapping leaves heavy atoms unmatched, or if
            sorting encounters an invalid SMARTS pattern.

    Note:
        Valid SMARTS patterns are required. Without sorting, an invalid
        pattern can raise an underlying RDKit exception instead of
        ``ValueError``.
    """
    matched_groups: list[GroupMappingResult] = []
    heavy_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    for group_idx, SMARTS in enumerate(groups):
        patt = Chem.MolFromSmarts(SMARTS)
        candidates: tuple[tuple[int, ...], ...] = mol.GetSubstructMatches(patt)
        for candidate in candidates:
            if set(candidate).issubset(heavy_atoms):
                heavy_atoms.difference_update(candidate)
                matched_groups.append({"group_idx": group_idx, "atoms": candidate})

    if strict and len(heavy_atoms) != 0:
        unmatched_SMILES = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=sorted(heavy_atoms),
            canonical=True,
            isomericSmiles=False,
        )
        unmatched_groups = unmatched_SMILES.replace(".", " / ")

        raise ValueError(
            f"Molecule does not consist only of the suggested groups.\nUnmatched atoms: {heavy_atoms}\nUnmatched fragments: {unmatched_groups}\nSMILES: {Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)}"
        )

    return matched_groups


def get_molecular_descriptor(
    mol: Chem.Mol,
    groups: list[SMARTS],
    *,
    strict: bool = True,
) -> list[int]:
    """Count greedily mapped occurrences of each molecular group.

    Overlapping matches are not counted independently, and earlier SMARTS
    patterns take precedence.

    Args:
        mol: RDKit molecule to describe.
        groups: SMARTS patterns whose accepted occurrences are counted.
        strict: Whether every heavy atom must be assigned to a group.

    Returns:
        A list of length ``len(groups)``. Element ``i`` is the number of
        accepted matches for ``groups[i]``.

    Raises:
        ValueError: If strict mapping leaves one or more heavy atoms
            unmatched.

    Note:
        Valid SMARTS patterns are required. An invalid pattern can raise an
        underlying RDKit exception.
    """

    group_descriptor = [0] * (len(groups))
    group_mapping_results = map_groups(mol, groups, strict=strict)
    for group_mapping_result in group_mapping_results:
        group_descriptor[group_mapping_result["group_idx"]] += 1

    return group_descriptor


def get_molecular_graph(
    mol: Chem.Mol,
    groups: list[SMARTS],
    *,
    normalize: bool = True,
    self_loop: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build group feature and connectivity matrices for a molecule.

    Cross-group neighbor relationships are counted in the connectivity
    matrix. Self-loops can be set before symmetric degree normalization is
    applied as ``D**-0.5 @ efm @ D**-0.5``.

    Args:
        mol: RDKit molecule to encode.
        groups: SMARTS patterns defining group features.
        normalize: Whether to symmetrically normalize the connectivity
            matrix.
        self_loop: Whether to set every diagonal connectivity entry to
            ``1.0`` before normalization.
        sort: Whether to process a priority-sorted copy of ``groups``. When
            enabled, feature columns and connectivity indices refer to the
            sorted order.

    Returns:
        A pair ``(nfm, efm)`` of ``numpy.float64`` arrays. If ``N`` is the
        number of heavy atoms and ``G`` is the number of patterns, ``nfm``
        has shape ``(N, G)`` and ``efm`` has shape ``(N, N)``.
    """
    group_mapping_results = map_groups(mol, groups, strict=True)

    len_groups = len(groups)
    len_matched_groups = len(group_mapping_results)
    nfm = np.zeros((len_matched_groups, len_groups))
    efm = np.zeros((len_matched_groups, len_matched_groups))

    atom_idx_to_node_idx: dict[int, int] = {}
    for node_idx, group_mapping_result in enumerate(group_mapping_results):
        nfm[node_idx, group_mapping_result["group_idx"]] = 1.0
        for atom_idx in group_mapping_result["atoms"]:
            atom_idx_to_node_idx[atom_idx] = node_idx

    for node_idx, group_mapping_result in enumerate(group_mapping_results):
        atom_idxes = group_mapping_result["atoms"]
        for atom_idx in group_mapping_result["atoms"]:
            target_atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor_atom in target_atom.GetNeighbors():
                neighbor_atom = cast(Chem.Atom, neighbor_atom)
                neighbor_atom_idx = neighbor_atom.GetIdx()
                if neighbor_atom_idx not in atom_idxes:
                    efm[
                        node_idx,
                        atom_idx_to_node_idx[neighbor_atom_idx],
                    ] += 1.0

    if self_loop:
        np.fill_diagonal(efm, 1)

    if normalize:
        degmat = np.diag(np.sum(efm, axis=1))
        degmat_m0p5 = fractional_matrix_power(degmat, -0.5)
        efm = np.matmul(np.matmul(degmat_m0p5, efm), degmat_m0p5)

    return nfm, efm
