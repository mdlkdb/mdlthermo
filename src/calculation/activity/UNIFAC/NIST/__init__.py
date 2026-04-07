from typing import List

from .modules import auto_fragment, UNIFAC


def cal_binary_UNIFAC(
    SMILES1: str,
    SMILES2: str,
    x1: float,
    x2: float,
    T: float,
    check_temperature=False,
) -> List[float, float]:
    """
    Calculate the activity coefficients for a binary mixture using the UNIFAC model.

    This function takes two SMILES strings representing the components of a binary mixture,
    their mole fractions, and the temperature. It then calculates the activity coefficients
    for each component using the UNIFAC model.

    Parameters:
    ----------
    SMILES1 : str
        A valid SMILES string representing the first component in the binary mixture.

    SMILES2 : str
        A valid SMILES string representing the second component in the binary mixture.

    x1 : float
        The mole fraction of the first component.

    x2 : float
        The mole fraction of the second component.

    T : float
        The temperature of the mixture (in Kelvin).

    check_temperature : bool, optional
        If `True`, the function checks whether the temperature is within the valid range
        for the interaction parameters. If the temperature is outside the range, an error is raised.
        Default is `False`.

    Returns:
    -------
    Tuple[float, float]
        A tuple containing the activity coefficients for the two components in the mixture.

    Raises:
    ------
    ValueError
        - If either component cannot be fully represented by the UNIFAC groups.
        - If the temperature is outside the valid range (when `check_temperature` is `True`).

    Notes:
    ------
    - The function uses `auto_fragment()` to decompose the SMILES strings into UNIFAC groups.
    - The actual calculation is performed by the `UNIFAC()` function, which accounts for both
      combinatorial and residual contributions to the activity coefficient.
    - The mole fractions `x1` and `x2` should sum to approximately 1.0 for accurate results.

    Example:
    --------
    ```python
    from your_module import cal_binary_UNIFAC

    SMILES1 = "CCO"
    SMILES2 = "CCCC"
    x1 = 0.5
    x2 = 0.5
    temperature = 298.15

    activity_coefficients = cal_binary_UNIFAC(SMILES1, SMILES2, x1, x2, temperature)
    print(activity_coefficients)  # 출력 예시: (1.05, 0.98)
    ```
    """
    component1 = auto_fragment(SMILES1)
    component2 = auto_fragment(SMILES2)

    return UNIFAC(
        [component1, component2],
        [x1, x2],
        T,
        check_temperature=check_temperature,
    )
