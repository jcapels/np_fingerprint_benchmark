from pathway_prediction.data_integration.standardize_molecules import standardize_molecules


standardize_molecules("coconut_reduced.csv", smiles_field="canonical_smiles", id_field="identifier", output_path="coconut_standardized.csv")

standardize_molecules("lotusdb.csv", smiles_field="SMILES", id_field="lotus_id", output_path="lotusdb_standardized.csv")