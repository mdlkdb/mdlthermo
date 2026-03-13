from rdkit import Chem

from src import descriptor


smiles = "CCCC[Si]F"
mol = Chem.MolFromSmiles(smiles)

organic_group = descriptor.defaults.Organic

print(descriptor.get_molecular_descriptor(smiles, organic_group))
print(descriptor.get_molecular_graph(smiles, organic_group))
