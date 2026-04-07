# Installation

```
# TODO

1. default groups 완성 하기
2.
3.
4.
```

git bash

```
git clone https://github.com/neros12/MDL_modules.git
```

or

code >> download_zip

### Directory Structure

In order to import this projects, your direcotry structure must be like

```
ROOT FOLDER
├── MDL_modules/
│
└── your_script.py
```

After installing MDL_modules, you could import module into Python code like:

```
from MDL_modules import *
```

# Module Documentation

## GCGCN - pure constant estimation

```
from MDL_modules import GCGCN
```

```
GCGCN.predict_TBN(SMILES: str)
```

```
GCGCN.predict_TMN(SMILES: str)
```

```
GCGCN.predict_TF(SMILES: str)
```

```
GCGCN.predict_TC(SMILES: str)
```

```
GCGCN.predict_PC(SMILES: str)
```

```
GCGCN.predict_VC(SMILES: str)
```

```
GCGCN.predict_HFORM(SMILES: str)
```

```
GCGCN.predict_HFUS(SMILES: str)
```

### Reference

[1] Hwang, Sun Yoo, and Jeong Won Kang. "Group Contribution-Based Graph Convolution Network: Pure Property Estimation Model." International Journal of Thermophysics 43.9 (2022): 136.

## COMSO-SAC

```
from MDL_modules import COSMOSAC
```

```
COSMOSAC.cal_COSMOSAC(SMILES1, SMILES2, x1, x2, T, predict=False):
```

### Reference

[1] Kang, Sung Shin, Jonghwi Lee, and Jeong Won Kang. "An extended COSMO-SAC method for the prediction of carboxylic acid solubility." Fluid Phase Equilibria 521 (2020): 112673.  
[2] Bell, Ian H., et al. "A benchmark open-source implementation of COSMO-SAC." Journal of chemical theory and computation 16.4 (2020): 2635-2646.  
[3] Ryu, Beom Chan, et al. "Group Contribution Based Graph Convolution Network: Predicting Vapor–Liquid Equilibrium with COSMO-SAC-ML." International Journal of Thermophysics 44.4 (2023): 49.  
[4] Eric Mullins, Y.A. Liu, Adel Ghaderi, Stephen Fast, "Sigma Profile Database for Predicting Solid Solubility in Pure and Mixed Solvent Mixtures for Organic Pharmacological Compounds with COSMO-Based Thermodynamic Methods," Ind. Eng. Chem. Research, 47, 1707-1725 (2008)

## UNIFAC

```
from MDL_modules.UNIFAC.NIST import cal_NIST_UNIFAC
```

```
COSMOSAC.cal_NIST_UNIFAC(SMILES1, SMILES2, x1, x2, T, cmp1_subgroups, cmp2_subgroups):
```

### Reference

[1] Kang, Jeong Won, Vladimir Diky, and Michael Frenkel. "New modified UNIFAC parameters using critically evaluated phase equilibrium data." Fluid Phase Equilibria 388 (2015): 128-141.

## Vapor Pressure

```
from MDL_modules.vapor_pressure import equations, GCGCN
```

### Reference

[1] Frenkel, Michael D., et al. "ThermoData Engine (TDE) Version 9.0 (Pure Compounds, Binary Mixtures, Ternary Mixtures, and Chemical Reactions); NIST Standard Reference Database 103b." (2014).

# Contact

E-mail: neros12@naver.com
