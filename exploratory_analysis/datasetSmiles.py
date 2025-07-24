import pandas as pd

# === 1. Carregar os datasets ===
pairs_df = pd.read_csv("pairs_df.csv")
compounds_df = pd.read_csv("compounds.csv")

# === 2. Normalizar nomes das colunas ===
pairs_df.columns = pairs_df.columns.str.strip().str.lower()
compounds_df.columns = compounds_df.columns.str.strip().str.lower()

# === 3. Identificar colunas de ID e SMILES ===
mol_id_col = "id" if "id" in compounds_df.columns else compounds_df.columns[0]
smiles_col = "smiles" if "smiles" in compounds_df.columns else compounds_df.columns[1]

# === 4. Juntar os SMILES às moléculas ===
merged = pairs_df.merge(compounds_df[[mol_id_col, smiles_col]], left_on="mol1", right_on=mol_id_col)
merged = merged.merge(compounds_df[[mol_id_col, smiles_col]], left_on="mol2", right_on=mol_id_col, suffixes=("_mol1", "_mol2"))

# === 5. Criar dataset final apenas com as colunas necessárias ===
final_df = merged[["mol1", "smiles_mol1", "mol2", "smiles_mol2"]]

# === 6. Guardar  ===
final_df.to_csv("merged_pairs_with_smiles.csv", index=False)



