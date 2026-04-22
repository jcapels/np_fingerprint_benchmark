
import pandas as pd
from sklearn.metrics import f1_score
from torchmetrics.classification import  MultilabelF1Score
import torch.nn as nn


from torch.optim import Adam

from deepmol.loaders.loaders import CSVLoaderForMaskedLM

from np_fingerprint_benchmark.np_bert_dataset import CSVLoaderNPBERT

def train_model(train_dataset, validation_dataset):
    import pytorch_lightning as pl
    from deepmol.models import BERT
    from torchmetrics.text import Perplexity
    model = BERT(vocab_size=train_dataset.tokenizer.vocab_size, max_length=518,
                accelerator="gpu", devices=[0], max_epochs=10, batch_size=56,
                metric = Perplexity(), patience=10, hidden_size=512, num_hidden_layers=5, num_attention_heads=4,
                intermediate_size=256, max_position_embeddings=520).fit(train_dataset, validation_dataset=validation_dataset)

    model.save("NPBERT")
    
    print(model.evaluate(validation_dataset))

def train_full_model(train_dataset):
    import pytorch_lightning as pl
    from deepmol.models import BERT
    from torchmetrics.text import Perplexity
    model = BERT(vocab_size=train_dataset.tokenizer.vocab_size, max_length=518,
                accelerator="gpu", devices=[0], max_epochs=10, batch_size=56,
                metric = Perplexity(), patience=10, hidden_size=512, num_hidden_layers=5, num_attention_heads=4,
                intermediate_size=256, max_position_embeddings=520).fit(train_dataset)

    model.save("NPBERT")
    

def evaluate_model(validation_dataset):
    from deepmol.models import BERT
    BERT.load("NPBERT", mode="masked_learning").evaluate(validation_dataset)



dataset = CSVLoaderNPBERT("../data/integrated_dataset_km_training_wo_fintuning_datasets.csv", id_field="ids",
                    smiles_field="smiles", max_length=518,  
                    masking_probability=0.15, vocabulary_path="vocab.txt").create_dataset()

train_full_model(dataset)
