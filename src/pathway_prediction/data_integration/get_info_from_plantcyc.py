from anytree import Node

def data_by_record(filename):
    """
    Auxiliar function to read the dat file and divide the info by record
    Parameters
    ----------
    filename: str
    json_path for the dat file

    Returns
    -------
    all_records: list
    list of dicts where each dict has the info of each record
    """
    all_records = []
    datafile = open(filename, 'r', encoding='ISO-8859-1')
    line = datafile.readline().strip()

    while line:
        if line[0] != "#" and "UNIQUE-ID" in line:
            record = []
            while line != '//':
                record.append(line)
                line = datafile.readline().strip()

            record_dic = {}
            for info in record:
                if info.count(' - ') == 1:
                    key, value = info.split(' - ')
                    if key not in record_dic:
                        record_dic[key] = [value]
                    else:
                        record_dic[key].append(value)

            all_records.append(record_dic)
        else:
            line = datafile.readline().strip()

    return all_records

def get_compounds_database(compounds):
    compounds_database = {}

    for compound in compounds:
        types = compound['UNIQUE-ID'][0]
        
        if "SMILES" in compound:
            smiles = compound['SMILES'][0]

            if 'COMMON-NAME' in compound:
                name = compound['COMMON-NAME'][0]
                compound_dict = {"SMILES": smiles,
                        "name": name}
            else: 
                compound_dict = {"SMILES": smiles,
                        "name": None}
            compounds_database[types] = compound_dict

    return compounds_database

def get_all_children(nodes):
    all_children = []

    for node in nodes:
        all_children.append(node.pathway)
        if not node.is_leaf:
            all_children.extend(get_all_children(node.children))

    return all_children

import re

def parse_compounds_by_primaries(reaction_list):
    """
    Parse compound IDs from the LEFT-PRIMARIES and RIGHT-PRIMARIES sections of reaction strings.

    Parameters:
        reaction_list (list of str): A list of reaction strings.

    Returns:
        dict: A dictionary where the keys are reaction IDs, and the values are dictionaries with 'left' and 'right' compounds.
    """
    compound_data = []

    for reaction in reaction_list:
        # Extract the reaction ID (e.g., RXN-14108, 2.4.1.234-RXN)
        reaction_id_match = re.search(r'^\(([^ ]+)', reaction)
        if reaction_id_match:
            reaction_id = reaction_id_match.group(1)
        else:
            continue

        # Extract compounds from LEFT-PRIMARIES
        left_match = re.search(r'\(:LEFT-PRIMARIES ([^)]*)\)', reaction)
        left_compounds = left_match.group(1).split() if left_match else []

        # Extract compounds from RIGHT-PRIMARIES
        right_match = re.search(r'\(:RIGHT-PRIMARIES ([^)]*)\)', reaction)
        right_compounds = right_match.group(1).split() if right_match else []

        # Add the extracted data to the dictionary
        compound_data.extend(left_compounds)
        compound_data.extend(right_compounds)

    return compound_data

def get_dataset(pathways_path, compounds_path, classes_path):

    compounds = data_by_record(compounds_path)

    compounds_database = get_compounds_database(compounds)

    records = data_by_record(classes_path)
    pathways = ["SECONDARY-METABOLITE-BIOSYNTHESIS"]
    nodes = {"SECONDARY-METABOLITE-BIOSYNTHESIS": Node("SECONDARY-METABOLITE-BIOSYNTHESIS")}
    for record in records:
        types_ = record['TYPES']
        for type_ in types_:
            if type_ in pathways:
                nodes[record['UNIQUE-ID'][0]] = Node(record['UNIQUE-ID'][0], parent=nodes[type_], pathway = record['UNIQUE-ID'][0])
                pathways.extend(record['UNIQUE-ID'])

    secondary_metabolites_pathways_classes = get_all_children(nodes["SECONDARY-METABOLITE-BIOSYNTHESIS"].children)

    import pandas as pd
    pathways = data_by_record(pathways_path)

    metabolites_in_pathway_df = pd.DataFrame()
    for pathway in pathways:
        secondary_metabolism = False
        pathway_id = pathway["UNIQUE-ID"][0]
        i=0
        pathway_name = None
        if 'COMMON-NAME' in pathway:
            pathway_name = pathway['COMMON-NAME'][0]
        while i < len(pathway["TYPES"]) and not secondary_metabolism:
            types = pathway["TYPES"][i]
            if types in secondary_metabolites_pathways_classes:
                secondary_metabolism = True
            
            i+=1

        if secondary_metabolism:
            rows = []
            compounds_in_pathway = parse_compounds_by_primaries(pathway["REACTION-LAYOUT"])
            for compound_in_pathway in compounds_in_pathway:
                if compound_in_pathway in compounds_database:
                    compound_info = compounds_database[compound_in_pathway]
                    smiles = compound_info["SMILES"]
                    name = compound_info["name"]

                    row = {
                    "Pathway ID": pathway_id, 
                    "Type": types,
                    "Pathway Name": pathway_name,
                    "Metabolite": compound_in_pathway,
                    "Name": name,
                    "SMILES": smiles
                    }
                    rows.append(row)
        
            new_metabolites_df = pd.DataFrame(rows)
            metabolites_in_pathway_df = pd.concat((metabolites_in_pathway_df, new_metabolites_df))

    metabolites_in_pathway_df.drop_duplicates(subset=["Pathway ID", "Metabolite"], inplace=True)
    metabolites_in_pathway_df.to_csv("plantcyc_pathways.csv", index=False)

