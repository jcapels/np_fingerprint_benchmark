
from deepmol.loaders import CSVLoader
from deepmol.standardizer import ChEMBLStandardizer

def standardize_molecules(dataset_path, smiles_field, id_field, output_path):

    dataset = CSVLoader(dataset_path, smiles_field=smiles_field, id_field=id_field).create_dataset()

    ChEMBLStandardizer(n_jobs=30).standardize(dataset, inplace=True)

    dataset.to_csv(output_path, index=False)