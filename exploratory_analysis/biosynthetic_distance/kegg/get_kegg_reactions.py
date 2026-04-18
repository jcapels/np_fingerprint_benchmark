import re
import urllib

import requests
from tqdm import tqdm
from np_fingerprint_benchmark.data_integration.get_info_from_kegg import get_secondary_metabolites_and_their_pathways

import pandas as pd
from Bio.KEGG import REST as kegg_api

def get_reactions_in_pathway(pathway_id: str):
    ko_pathway_id = "ko" + pathway_id[3:]
    reactions = set()
    try:
        ko_pathway_data = kegg_api.kegg_get(ko_pathway_id).read().split('\n')
        in_orthology_section = False

        ks = set()

        for line in ko_pathway_data:
            if line.startswith('ORTHOLOGY'):
                in_orthology_section = True
                continue
            if in_orthology_section and line.startswith('///'):
                break
            if in_orthology_section and 'K' in line:
                # Extract all EC numbers from the line
                pattern = r'\bK\d{5}\b'
                matches = re.findall(pattern, line)
                for k in matches:
                    ks.add(k)
        
        for k in ks:
            link_handle = kegg_api.kegg_link("reaction", k)
            pattern = r"R\d+"
            reaction_ids = re.findall(pattern, link_handle.read())
            for reaction in reaction_ids:
                reactions.add(reaction)
        
    except urllib.error.HTTPError:
        pass
    
    return reactions

def fetch_kegg_reaction(reaction_id: str, pathway_id) -> dict:
    """
    Fetches reaction data from KEGG API.
    Returns a dictionary with reaction equation, metabolites, and other metadata.
    """
    url = f"http://rest.kegg.jp/get/{reaction_id}"
    response = requests.get(url)
    reactions_data = []
    if response.status_code == 200:
        kegg_reaction = parse_kegg_reaction(response.text)
        equation = kegg_reaction['equation']
        reactants, products = equation.split('<=>')
        reactants = [m.strip() for m in reactants.split(' + ')]
        products = [m.strip() for m in products.split(' + ') if m.strip()]

        for l in reactants:
            for r in products:
                if " " in l:
                    l = l.split(" ")[1]
                if " " in r:
                    r = r.split(" ")[1]
                reactions_data.append({
                    "reaction_id": reaction_id,
                    "pathway_id": pathway_id,
                    "left": l,
                    "right": r
                })
        return reactions_data 
    return None

def parse_kegg_reaction(kegg_reaction_text: str) -> dict:
    """
    Parses KEGG reaction text into a structured dictionary.
    Example:
        "EQUATION: A + B <=> C + D"
        "NAME: reaction_name"
        "ENZYME: 1.2.3.4"
    """
    data = {}
    for line in kegg_reaction_text.split('\n'):
        if line.startswith('EQUATION'):
            data['equation'] = line.split('EQUATION ')[1].strip()
        elif line.startswith('NAME'):
            line = " ".join(line.split())
            data['name'] = line.split('NAME ')[1].strip()
        elif line.startswith('ENZYME'):
            data['enzyme'] = line.split('ENZYME ')[1].strip()
    return data

def get_reactions_for_secondary_metabolites_pathways():
    # Get the list of pathways (e.g., map01060)
    secondary_metabolites_pathways = pd.read_html("https://www.genome.jp/dbget-bin/www_bget?pathway+map01060")
    results = pd.DataFrame()
    for pathway in secondary_metabolites_pathways[6:]:
        # Get reactions for the pathway
        pathway_id = pathway.iloc[:, 0].values[0]
        reactions_data = get_reactions_in_pathway(pathway_id)
        print(len(reactions_data))
        for reaction in reactions_data:
            reactions_data_list = fetch_kegg_reaction(reaction, pathway_id)
            if reactions_data_list:
                results = pd.concat((results, pd.DataFrame(reactions_data_list)))
        results.to_csv("kegg_reactions.csv", index=False)
        # Create a DataFrame

    # Export to CSV
    results.to_csv("kegg_reactions.csv", index=False)
    return results

# Run the function
get_reactions_for_secondary_metabolites_pathways()

get_secondary_metabolites_and_their_pathways()

