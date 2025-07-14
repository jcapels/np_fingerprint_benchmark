from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.splitters import RandomSplitter
from sklearn.metrics import f1_score, mean_squared_error
import optuna
import pandas as pd

import re

def replace_chars(text, replacement="_"):
    return re.sub(r"[\s()]", replacement, text)

dataset = pd.read_csv("train_dataset.csv", nrows=2)
labels = dataset.columns[2:]
# LOAD THE DATA
loader = CSVLoader('train_dataset.csv',
                   smiles_field='SMILES',labels_fields=labels)
train_dataset = loader.create_dataset(sep=",")
train_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]

loader = CSVLoader('validation_dataset.csv',
                   smiles_field='SMILES',labels_fields=labels)
validation_dataset = loader.create_dataset(sep=",")
validation_dataset._label_names = [str(i) for i, label_ in enumerate(validation_dataset._label_names)]



# OPTIMIZE THE PIPELINE
po = PipelineOptimization(direction='maximize', study_name='np_classifier', sampler=optuna.samplers.TPESampler(seed=42),
                          storage='sqlite:///np_classifier.db')
metric = Metric(f1_score, average="macro")
po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset, objective_steps='deepchem',
            metric=metric, n_trials=50, data=train_dataset, save_top_n=5, trial_timeout=60*60*24)