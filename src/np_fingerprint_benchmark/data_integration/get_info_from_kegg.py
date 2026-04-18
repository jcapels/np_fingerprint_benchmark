import pandas as pd
from np_fingerprint_benchmark.data_integration.kegg_api import KeggApi
from Bio.KEGG import REST as kegg_api

def get_secondary_metabolites_and_their_pathways():
    secondary_metabolites_pathways = pd.read_html("https://www.genome.jp/dbget-bin/www_bget?pathway+map01060")
    metabolites_in_pathway_df = pd.DataFrame()
    for pathway in secondary_metabolites_pathways[6:]:
        pathway_id = pathway.iloc[:, 0].values[0]
        pathway_name = pathway.iloc[:, 1].values[0]
        metabolites_in_pathway = KeggApi.get_links("compound", f"path:{pathway_id}")
        metabolites_in_pathway = metabolites_in_pathway

        rows = []

        # Iterate over the metabolites in the pathway
        for compound_id in metabolites_in_pathway[1].tolist():
            compound_id = compound_id.replace("cpd:", "")
            # Create a dictionary for each row
            row = {
                "Pathway": pathway_id,
                "Pathway Name": pathway_name,
                "Metabolite": compound_id,
                "SMILES": KeggApi.get_compound_smiles(compound_id)
            }
            rows.append(row)  # Add the row dictionary to the list

        # Create a new DataFrame from the collected rows
        new_metabolites_df = pd.DataFrame(rows)
        metabolites_in_pathway_df = pd.concat((metabolites_in_pathway_df, new_metabolites_df))
        
        metabolites_in_pathway_df.to_csv("kegg_pathways.csv", index=False)

def get_ec_numbers_from_ko_pathway(ko_pathway_id):
    ko_pathway_id = "ko" + ko_pathway_id[3:]
    # Fetch the KO pathway file
    ko_pathway_data = kegg_api.kegg_get(ko_pathway_id).read().split('\n')

    ec_numbers = set()
    in_orthology_section = False

    for line in ko_pathway_data:
        if line.startswith('ORTHOLOGY'):
            in_orthology_section = True
            continue
        if in_orthology_section and line.startswith('///'):
            break
        if in_orthology_section and '[EC:' in line:
            # Extract all EC numbers from the line
            ec_part = line.split('[EC:')[1:]
            for part in ec_part:
                ec = part.split(']')[0]
                ec_numbers.update(ec.split())

    return sorted(ec_numbers)
