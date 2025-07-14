import pandas as pd
from pathway_prediction.data_integration.kegg_api import KeggApi

def get_secondary_metabolites_and_their_pathways(self):
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