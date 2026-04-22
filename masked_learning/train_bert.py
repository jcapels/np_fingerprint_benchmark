
from deepmol.loaders.loaders import CSVLoaderForMaskedLM
import pandas as pd

def train_model(train_dataset, validation_dataset):
    import pytorch_lightning as pl
    from deepmol.models import BERT, ModernBERT, RoBERTa, DeBERTa, TransformerModelForMaskedLM
    from torchmetrics.text import Perplexity
    model = BERT(vocab_size=train_dataset.tokenizer.vocab_size, 
                accelerator="gpu", strategy="ddp_find_unused_parameters_true", devices=[0,1,2,3], max_epochs=10, batch_size=56,
                metric = Perplexity()).fit(train_dataset, validation_dataset=validation_dataset)

    model.save("data/train_test_split_scaffolds/BERT_small_0_2")
    
    print(model.evaluate(validation_dataset))

def train_model_large(train_dataset, validation_dataset):
    import pytorch_lightning as pl
    from deepmol.models import BERT, ModernBERT, RoBERTa, DeBERTa, TransformerModelForMaskedLM
    from torchmetrics.text import Perplexity
    model = BERT(vocab_size=train_dataset.tokenizer.vocab_size,
                       hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                accelerator="gpu", strategy="ddp_find_unused_parameters_true", devices=[0, 1, 2, 3], max_epochs=10, batch_size=16,
                metric = Perplexity(), patience=2).fit(train_dataset, validation_dataset=validation_dataset)

    model.save("data/train_test_split_scaffolds/BERT_large_0_2_v2")
    
    print(model.evaluate(validation_dataset))

def evaluate_model(test_dataset):
    metrics_df = pd.DataFrame()
    from deepmol.models import ModernBERT
    model = ModernBERT.load("data/train_test_split_scaffolds/NPModernBERT_small_0_2", mode="masked_learning")
    # model.evaluate(train_dataset))
    # print(model.evaluate(validation_dataset))
    print(model.evaluate(test_dataset))


loader = CSVLoaderForMaskedLM(dataset_path="data/train_test_split_scaffolds/train_dataset.csv",
                           smiles_field='smiles',
                           id_field='ids',
                           mode='auto', vocabulary_path="vocab.txt",
                           masking_probability=0.2)
train_dataset = loader.create_dataset(sep=',')

loader = CSVLoaderForMaskedLM(dataset_path="data/train_test_split_scaffolds/validation_dataset_wo_redundancy.csv",
                           smiles_field='smiles',
                           id_field='ids',
                           mode='auto', vocabulary_path="vocab.txt",
                           masking_probability=0.2)
validation_dataset = loader.create_dataset(sep=',')

# loader = CSVLoaderForMaskedLM(dataset_path="data/train_test_split_scaffolds/test_dataset_wo_redundancy.csv",
#                            smiles_field='smiles',
#                            id_field='ids',
#                            mode='auto', vocabulary_path="vocab.txt",
#                            masking_probability=0.2)
# test_dataset = loader.create_dataset(sep=',')

train_model_large(train_dataset, validation_dataset)
# evaluate_model(test_dataset)




