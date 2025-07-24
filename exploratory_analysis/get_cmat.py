
from tqdm import tqdm
import numpy as np


def cosine_similarity(x, y):
    """
    Compute cosine similarity between two vectors.
    Parameters:
        x, y : array-like
            Input vectors (must be the same length).
    Returns:
        float : Similarity score between 0 and 1.
    """
    x = np.array(x)
    y = np.array(y)
    
    dot_product = np.dot(x, y)
    norm_product = np.linalg.norm(x) * np.linalg.norm(y)
    
    if norm_product == 0:
        return np.nan  
    
    return dot_product / norm_product


def compute_all_similarities(dataset):
    """
    Compute all pairwise similarities in the dataset.
    Parameters:
        dataset : list of lists
            Each inner list represents a sample.
    Returns:
        list of tuples : Each tuple contains two indices and their similarity score.
    """
    fp = dataset.X.copy()
    
# Preallocate vector of correct size
    n_samples = fp.shape[0]
    n_comps = int((n_samples ** 2 - n_samples) / 2)
    distances = np.zeros((1, n_comps,), dtype=np.float32)

    # Start pairwise similarity counter
    count = 0

    # Loop over all unique pairs (skip redundant comparisons)
    for i in tqdm(range(n_samples - 1), desc="Computing similarities"):
        for j in range(i + 1, n_samples):
            
            distances[0, count] = cosine_similarity(fp[i, :], fp[j, :])
            count += 1

    return distances

import numpy as np


 

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle 


def generate_mst(similarities_matrix, labels):

    # Step 2: Convert the Correlation Matrix to a Distance Matrix
    distance_matrix = 1 - similarities_matrix

    # Create a graph from the distance matrix
    G = nx.Graph()
    num_nodes = distance_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    # Compute the minimum spanning tree without inversion
    mst_without_inversion = nx.minimum_spanning_tree(G)

    # Set the labels for the nodes in the graph
    for node in mst_without_inversion.nodes():
        if node in labels:
            mst_without_inversion.nodes[node]['label'] = labels[node]
        else:
            mst_without_inversion.nodes[node]['label'] = str(node)

    return mst_without_inversion

def fingerprint_pipeline(dataset, fingerprints, labels):
    
    import numpy as np

    similarity_matrices = [] 
    i=0
    for featurizer in fingerprints:
        featurizer.featurize(dataset, inplace=True)
        dataset.to_csv(f"{featurizer.__class__.__name__}_fp.csv")

        similarities = compute_all_similarities(dataset) 

        similarity_matrices.append(similarities) 

    similarities = np.concatenate(similarity_matrices, axis=0) 
        
    
    correlation_matrix = np.corrcoef(similarities) 
    with open("correlation_matrix.pkl","wb") as f:
        pickle.dump(correlation_matrix, f)
         
    
    with open("similarities.pkl", "wb") as f:
        pickle.dump(similarities, f)
    
    mst = generate_mst(similarities, labels)   
    with open("mst.pkl","wb") as f:
        pickle.dump(mst, f)
    

    return correlation_matrix, mst

def violin_plot(similarity_matrix):
    
    labels = [
    "",
    "NPClassifierFP",
    "Biosynfoni",
    "NP_AUX",
    "MHFP", 
    "MorganFP",
    "NPBERT",
    "ModernBERT"
    ]

    fig, ax = plt.subplots()
    ax.set_xticklabels(labels=labels, )
    ax.set_xlabel("Fingerprint Method")           
    ax.set_ylabel("Cosine Similarity")
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)

    for i in range(0,similarity_matrix.shape[0]):
        ax.violinplot(dataset=similarity_matrix[i],positions=[i])
    plt.xticks(rotation=45)
    plt.ylim(-1, 1.4)
    plt.title("Distribution of Similarity Scores by Fingerprint")
    plt.tight_layout()

    plt.savefig("violinplot_similarity.png", dpi=300)

def generate_mst_from_similarity(similarity_matrix, labels):
    """
    Generate a minimum spanning tree (MST) from a similarity matrix.
    
    Parameters:
        similarity_matrix : np.ndarray
            A square matrix representing pairwise similarities.
        labels : dict
            A dictionary mapping node indices to labels.
    
    Returns:
        nx.Graph : The minimum spanning tree as a NetworkX graph.
    """
    mst = generate_mst(similarity_matrix, labels)
    with open("mst.pkl","wb") as f:
        pickle.dump(mst, f)


if __name__ == "__main__":
    from deepmol.loaders import CSVLoader
    import pandas as pd
    import numpy as np
    from deepmol.compound_featurization import NPClassifierFP, BiosynfoniKeys, NeuralNPFP, MHFP, MorganFingerprint
    from deepmol.tokenizers import NPBERTTokenizer
    import os
    from deepmol.compound_featurization import LLM
    from transformers import BertConfig, BertModel, ModernBertModel, ModernBertConfig
    

 #Load data from CSV file
#     loader = CSVLoader(dataset_path='30k_sample.csv',
#                    smiles_field='smiles',
#                    id_field='ids',
#                    mode='auto')
# # create the dataset
#     csv_dataset = loader.create_dataset(sep=',', header=0)
#     fingerprint_pipeline(
#         csv_dataset,                     # O dataset carregado com SMILES
#         fingerprints=[NPClassifierFP(), BiosynfoniKeys(), NeuralNPFP(), MHFP(), MorganFingerprint(), LLM(model_path="../NPBERT", model=BertModel, config_class=BertConfig,
#                           tokenizer=NPBERTTokenizer(vocab_file=os.path.join("../NPBERT", "vocab.txt")), device="cuda:1"),
#                           LLM(model_path="../ModernBERT", model=ModernBertModel, config_class=ModernBertConfig, device="cuda:1")
#                           ],
        # labels={
        #         0: "NPClassifierFP",
        #         1: "BiosynfoniKeys",
        #         2: "NeuralNPFP",
        #         3: "MHFP", 
        #         4: "MorganFingerprint",
        #         5: "NPBERT",
        #         6: "ModernBERT"
        # } 
#                ) 
    labels={
                0: "NPClassifierFP",
                1: "BiosynfoniKeys",
                2: "NeuralNPFP",
                3: "MHFP", 
                4: "MorganFP",
                5: "NPBERT",
                6: "ModernBERT"
        } 
    
    with open("similarities.pkl", "rb") as f:
        similarity_matrix = pickle.load(f)
    violin_plot(similarity_matrix)

    generate_mst_from_similarity(similarity_matrix, labels)
