from src.calculation.pure.v3 import embed_smiles, predict_Vp

smiles = "CCCCC"
a = embed_smiles(smiles)
print(predict_Vp(a, 300))
