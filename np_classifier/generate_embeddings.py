from deepmol.loaders import CSVLoader


dataset = CSVLoader(dataset_path="merged_dataset.csv", smiles_field="SMILES", id_field="key").create_dataset()

import os
from deepmol.compound_featurization import LLM
from transformers import BertConfig, BertModel

from deepmol.standardizer import ChEMBLStandardizer

from deepmol.tokenizers import NPBERTTokenizer

transformer = LLM(model_path="../NPBERT", model=BertModel, config_class=BertConfig,
                          tokenizer=NPBERTTokenizer(vocab_file=os.path.join("../NPBERT", "vocab.txt")), device="cuda:0")


ChEMBLStandardizer().standardize(dataset, inplace=True)
# Featurize the datasets
transformer.featurize(dataset, inplace=True)

# Create the dictionary
data_dict = {id_: x for id_, x in zip(dataset.ids, dataset.X)}

# to pickle
import pickle

# Specify the filename
filename = 'np_bert.pkl'

# Write the data to a file using pickle
with open(filename, 'wb') as file:
    pickle.dump(data_dict, file)

    
