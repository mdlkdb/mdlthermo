from typing import List, Dict
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from . import parameters


def auto_fragment(SMILES: str) -> Dict[int, int]:
    """
    Fragment a molecule given by a SMILES string into UNIFAC groups automatically.

    This function takes a SMILES (Simplified Molecular Input Line Entry System) string,
    converts it to an RDKit molecule object, and identifies which UNIFAC groups the
    molecule consists of. It returns a dictionary where the keys are subgroup IDs and
    the values are the counts of each subgroup in the molecule.

    Parameters:
    ----------
    SMILES : str
        A valid SMILES string representing the chemical structure of the molecule.

    Returns:
    -------
    Dict[group index, number of groups]
        A dictionary containing the UNIFAC subgroup IDs as keys and the number of times
        each subgroup appears in the molecule as values.

    Raises:
    ------
    ValueError
        If the molecule cannot be fully represented by the available UNIFAC groups.

    Notes:
    ------
    - The function uses RDKit to parse the molecule and match it against predefined UNIFAC
      subgroup SMARTS patterns from the `parameters.subgroup_SMARTS` list.
    - Groups marked with the "EX" SMARTS pattern are excluded as they are not supported.
    - After a subgroup is identified, the matched atoms are removed from further consideration.

    Dependencies:
    -------------
    - RDKit: For molecular structure manipulation and substructure matching.
    - numpy: For array operations.
    - parameters: A module containing the predefined UNIFAC subgroup SMARTS patterns.

    Example:
    --------
    ```python
    from your_module import auto_fragment

    SMILES = "CCO"
    subgroups = auto_fragment(SMILES)
    print(subgroups)
    ```
    """
    mol = Chem.MolFromSmiles(SMILES)

    num_atoms = Descriptors.HeavyAtomCount(mol)
    list_atoms = np.arange(num_atoms)
    subgroups = []
    for item in parameters.subgroup_SMARTS:
        subgroup_ID = item[0]
        SMARTS = item[1]

        # These groups are not supported in current version.
        if SMARTS == "EX":
            continue

        if mol.HasSubstructMatch(Chem.MolFromSmarts(SMARTS)) == True:
            matched_fragments = mol.GetSubstructMatches(Chem.MolFromSmarts(SMARTS))
            for matched_atoms in matched_fragments:
                if set(matched_atoms) & set(list_atoms) == set(matched_atoms):
                    subgroups.append(subgroup_ID)
                    list_atoms = np.setdiff1d(list_atoms, matched_atoms)

        if len(list_atoms) == 0:
            break

    if len(list_atoms) != 0:
        raise ValueError("Molecule is not only consist of UNIFAC groups")

    return dict(Counter(subgroups))


def UNIFAC(
    components: List[Dict[int, int]],
    x: List[float],
    T: float,
    check_temperature=True,
) -> List[float]:
    """
    Calculate the UNIFAC activity coefficients for a mixture of components.

    The UNIFAC (UNIversal Functional Activity Coefficient) model is used to estimate
    activity coefficients in multicomponent systems. This method calculates both the
    combinatorial and residual contributions to the activity coefficient.

    Parameters:
    ----------
    components : List[Dict[int, int]]
        A list of dictionaries, where each dictionary represents a component. The keys
        are subgroup IDs, and the values are the number of times each subgroup appears
        in the component.
        Example: [{1: 2, 2: 4}, {1: 1, 3: 1, 2: 4}]

    x : List[float]
        A list of mole fractions for each component in the mixture. The length of this list
        should match the number of components.

    T : float
        The temperature of the system (in Kelvin) at which the activity coefficients are
        calculated.

    check_temperature : bool, optional
        If `True`, the function checks whether the given temperature falls within the valid
        range for the interaction parameters. If the temperature is outside the range, an
        error is raised. Default is `True`.

    Returns:
    -------
    List[float]
        A list of activity coefficients for each component in the mixture.

    Raises:
    ------
    ValueError
        - If there are missing interaction parameters between main groups.
        - If the given temperature is outside the valid range (when `check_temperature` is `True`).

    Notes:
    ------
    - The activity coefficients are calculated using both combinatorial and residual terms.
    - Interaction parameters are retrieved from the `parameters` module, which includes:
        - `parameters.sub_to_main`: Mapping from subgroups to main groups.
        - `parameters.subgroup_par`: Parameters for each subgroup (`Ri`, `Qi` values).
        - `parameters.maingroup_par`: Interaction parameters between main groups (`a1`, `a2`, `a3`).

    Example:
    --------
    ```python
    components = [{1: 2, 2: 4}, {1: 1, 3: 1, 2: 4}]
    mole_fractions = [0.5, 0.5]
    temperature = 298.15

    activity_coefficients = UNIFAC(components, mole_fractions, temperature)
    print(activity_coefficients)  # 출력: [1.05, 0.97]
    ```
    """

    num_components = len(components)
    component_indexes = range(num_components)

    cs: List[List[int]] = []
    for component in components:
        component_groups = []
        for key, count in component.items():
            component_groups.extend([key] * count)
        cs.append(component_groups)

    total_subgroups = list(set(group for c in cs for group in c))
    total_maingroups = set(
        parameters.sub_to_main[subgroup] for subgroup in total_subgroups
    )

    # check temperature
    if check_temperature:
        Tmax = []
        Tmin = []
        iter_maingroups = []
        for m in total_maingroups:
            for n in total_maingroups:
                if m == n:
                    continue
                else:
                    if (m, n) in parameters.maingroup_par.keys():
                        iter_maingroups.append((m, n))
                        Tmax.append(parameters.maingroup_par[(m, n)]["Tmax"])
                        Tmin.append(parameters.maingroup_par[(m, n)]["Tmin"])
                    else:
                        raise ValueError(
                            f"There are no interaction parameters between maingroup {m} and maingroup {n}."
                        )
        if len(Tmax) != 0:
            Tmax = max(Tmax)
            Tmin = min(Tmin)
            if T > Tmax or T < Tmin:
                raise ValueError(
                    "Current method is unsupported for the given temperature condition."
                )

    vks = []
    for c in cs:
        vk = {}
        for k in total_subgroups:
            vk[k] = c.count(k)
        vks.append(vk)

    # Combinatorial Term
    r = []  # index = component
    q = []  # index = component
    for index in component_indexes:
        SumR, SumQ = 0.0, 0.0
        for k in cs[index]:
            SumR += parameters.subgroup_par[k]["Ri"]
            SumQ += parameters.subgroup_par[k]["Qi"]
        r.append(SumR)
        q.append(SumQ)

    J = []  # index = component
    L = []  # index = component
    for i in component_indexes:
        rx = 0.0
        qx = 0.0
        for j in component_indexes:
            rx += r[j] ** (3 / 4) * x[j]
            qx += q[j] * x[j]
        J.append(r[i] ** (3 / 4) / rx)
        L.append(q[i] / qx)

    ln_rc = []  # index = component
    for i in component_indexes:
        ln_rc.append(
            1 - J[i] + np.log(J[i]) - 5 * q[i] * (1 - J[i] / L[i] + np.log(J[i] / L[i]))
        )

    # Residual Term
    e = {}  # key = (subgroup, component)
    for k in total_subgroups:
        for i in component_indexes:
            e[(k, i)] = vks[i][k] * parameters.subgroup_par[k]["Qi"] / q[i]

    theta = {}  # key = subgroup
    for k in total_subgroups:
        xqe = 0
        xq = 0
        for i in component_indexes:
            xqe += x[i] * q[i] * e[(k, i)]
            xq += x[i] * q[i]
        theta[k] = xqe / xq

    u = {}  # key = (subgroup, subgroup)
    for m in total_subgroups:
        for k in total_subgroups:
            i = parameters.sub_to_main[m]
            j = parameters.sub_to_main[k]
            if i == j:
                u[(m, k)] = 0.0
            else:
                u[(m, k)] = (
                    parameters.maingroup_par[(i, j)]["a1"]
                    + parameters.maingroup_par[(i, j)]["a2"] * T
                    + parameters.maingroup_par[(i, j)]["a3"] / 1000 * T * T
                )

    tow = {}  # key = (subgroup, subgroup)
    for m in total_subgroups:
        for k in total_subgroups:
            tow[(m, k)] = np.exp(-u[(m, k)] / T)

    betta = {}  # key = (component, subgroup)
    for i in component_indexes:
        for k in total_subgroups:
            sumbetta = 0
            for m in total_subgroups:
                sumbetta += e[(m, i)] * tow[(m, k)]
            betta[(i, k)] = sumbetta

    s = {}  # key = subgroup
    for k in total_subgroups:
        sums = 0
        for m in total_subgroups:
            sums += theta[m] * tow[(m, k)]
        s[k] = sums

    ln_rr = []  # index = component
    for i in component_indexes:
        sum_residual = 0
        for k in total_subgroups:
            sum_residual += theta[k] * betta[(i, k)] / s[k] - e[(k, i)] * np.log(
                betta[(i, k)] / s[k]
            )
        ln_rr.append(q[i] * (1 - sum_residual))

    ln_r = []  # key = component
    for i in component_indexes:
        ln_r.append(ln_rc[i] + ln_rr[i])

    activity_coefficient = np.exp(ln_r)

    return activity_coefficient
