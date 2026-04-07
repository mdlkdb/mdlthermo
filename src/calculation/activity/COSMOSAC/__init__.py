from .modules import COSMOSAC


def cal_COSMOSAC(SMILES1, SMILES2, x1, x2, T, predict=False):
    if predict:
        COSMOSAC_module = COSMOSAC(version=2010, predict=True)
    else:
        COSMOSAC_module = COSMOSAC(version=2019, predict=False)

    COSMOSAC_module.add_comp(SMILES=SMILES1)
    COSMOSAC_module.add_comp(SMILES=SMILES2)
    COSMOSAC_module.x = [x1, x2]
    COSMOSAC_module.T = T

    return COSMOSAC_module.gam()
