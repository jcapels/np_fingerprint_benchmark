
from deepmol.loaders.loaders import CSVLoaderForMaskedLM
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torchmetrics.classification import  MultilabelF1Score
import torch.nn as nn


from torch.optim import Adam

def train_model(train_dataset, validation_dataset):
    import pytorch_lightning as pl
    from deepmol.models import BERT, ModernBERT, RoBERTa, DeBERTa, TransformerModelForMaskedLM
    from torchmetrics.text import Perplexity
    model = ModernBERT(vocab_size=train_dataset.tokenizer.vocab_size, 
                accelerator="gpu", strategy="ddp_find_unused_parameters_true", devices=[0,1,2,3], max_epochs=10, batch_size=56,
                metric = Perplexity(), patience=10).fit(train_dataset, validation_dataset=validation_dataset)

    model.save("data/train_test_split_scaffolds/NPModernBERT_small_0_2")
    
    print(model.evaluate(validation_dataset))

def train_model_large_full_dataset(train_dataset):
    import pytorch_lightning as pl
    from deepmol.models import BERT, ModernBERT, RoBERTa, DeBERTa, TransformerModelForMaskedLM
    from torchmetrics.text import Perplexity
    model = ModernBERT(vocab_size=train_dataset.tokenizer.vocab_size,
                       hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                accelerator="gpu", strategy="ddp_find_unused_parameters_true", devices=[0, 1,2, 3], max_epochs=10, batch_size=8,
                metric = Perplexity(), patience=2).fit(train_dataset)

    model.save("./ModernBERT")

def train_model_save(train_dataset):
    import pytorch_lightning as pl
    from deepmol.models import BERT, ModernBERT, RoBERTa, DeBERTa, TransformerModelForMaskedLM
    from torchmetrics.text import Perplexity
    # model = ModernBERT(vocab_size=train_dataset.tokenizer.vocab_size,
    #                    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
    #             accelerator="gpu", strategy="ddp_find_unused_parameters_true", devices=[0], max_epochs=10, batch_size=8,
    #             metric = Perplexity(), patience=2)

    model = ModernBERT.load_from_checkpoint("checkpoints/epoch-epoch=09.ckpt", map_location='cpu')
    model.trainer = pl.Trainer(accelerator="cpu", max_epochs=10)
    model.evaluate(train_dataset)
    model.save("./ModernBERT")

# loader = CSVLoaderForMaskedLM(dataset_path="data/train_test_split_scaffolds/train_dataset.csv",
#                            smiles_field='smiles',
#                            id_field='ids',
#                            mode='auto', vocabulary_path="vocab.txt",
#                            masking_probability=0.2)
# train_dataset = loader.create_dataset(sep=',')

loader = CSVLoaderForMaskedLM(dataset_path="./data/integrated_dataset_km_training_wo_fintuning_datasets.csv",
                           smiles_field='smiles',
                           id_field='ids',
                           mode='auto', vocabulary_path="vocab.txt",
                           masking_probability=0.2, shard_size=100)
train_dataset = loader.create_dataset(sep=',')

# train_model_large_full_dataset(train_dataset)
train_model_save(train_dataset)


# loader = CSVLoaderForMaskedLM(dataset_path="data/train_test_split_scaffolds/validation_dataset_wo_redundancy.csv",
#                            smiles_field='smiles',
#                            id_field='ids',
#                            mode='auto', vocabulary_path="vocab.txt",
#                            masking_probability=0.2)
# validation_dataset = loader.create_dataset(sep=',')

# loader = CSVLoaderForMaskedLM(dataset_path="data/train_test_split_scaffolds/test_dataset_wo_redundancy.csv",
#                            smiles_field='smiles',
#                            id_field='ids',
#                            mode='auto', vocabulary_path="vocab.txt",
#                            masking_probability=0.2)
# test_dataset = loader.create_dataset(sep=',')

# train_model(train_dataset, validation_dataset)
# train_model_large(train_dataset, validation_dataset)

# evaluate_model(test_dataset)
# from deepmol.models import ModernBERT
# model = ModernBERT.load("data/train_test_split_scaffolds/NPModernBERT_large_0_2", mode = "masked_learning")
# torch.save(model.model.state_dict(), "data/train_test_split_scaffolds/NPModernBERT_large_0_2/model.pt")





