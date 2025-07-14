from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.splitters import RandomSplitter
from sklearn.metrics import f1_score, mean_squared_error
import optuna
import pandas as pd

from copy import deepcopy, copy
from typing import Literal

try:
    from deepchem.models import TextCNNModel
    from deepmol.pipeline_optimization._keras_model_objectives import _get_keras_model
    from deepmol.pipeline_optimization._deepchem_models_objectives import *
    from deepmol.compound_featurization import CoulombFeat
except ImportError:
    pass

from deepmol.base import DatasetTransformer, PassThroughTransformer
from deepmol.compound_featurization import NPClassifierFP, NeuralNPFP
from deepmol.datasets import Dataset
from deepmol.datasets._utils import _get_n_classes
from deepmol.encoders.label_one_hot_encoder import LabelOneHotEncoder
from deepmol.pipeline_optimization._feature_selector_objectives import _get_feature_selector
from deepmol.pipeline_optimization._featurizer_objectives import _get_featurizer
from deepmol.pipeline_optimization._scaler_objectives import _get_scaler
from deepmol.pipeline_optimization._sklearn_model_objectives import _get_sk_model
from deepmol.pipeline_optimization._standardizer_objectives import _get_standardizer
from deepmol.scalers.sklearn_scalers import MinMaxScaler

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

def generic_steps(trial, data, featurizer):

    mode = data.mode
    n_classes = _get_n_classes(data)
    # TODO: in multitask when n_classes > 2 for different tasks, this will not work (LabelOneHotEncoder needs to work
    #  on different y columns)
    label_encoder = LabelOneHotEncoder() if mode == 'classification' and n_classes[0] > 2 else PassThroughTransformer()
    feature_selector = PassThroughTransformer()
    scaler = PassThroughTransformer()

    
    data = featurizer.fit_transform(data)
    input_shape = (data.X.shape[0],)
    model_type_choice = trial.suggest_categorical('model_type', ['keras', 'sklearn'])
    if model_type_choice == 'keras':
        model = _get_keras_model(trial, input_shape, data)
    elif model_type_choice == "sklearn":
        if mode == 'classification':
            sk_mode = 'classification_binary' if set(data.y) == {0, 1} else 'classification_multiclass'
        else:
            sk_mode = mode
        model = _get_sk_model(trial, task_type=sk_mode)

    final_steps = [('label_encoder', label_encoder), ('standardizer', _get_standardizer(trial, featurizer)),
                   ('featurizer', featurizer), ('scaler', scaler), ('feature_selector', feature_selector),
                   ('model', model)]
    return final_steps

def np_classifier_fp_steps(trial, data):
    featurizer = NPClassifierFP()

    return generic_steps(trial, data, featurizer)

def neural_npfp_steps(trial, data):
    featurizer_type = trial.suggest_categorical('model', ['aux', 'base', 'ae'])
    featurizer = NeuralNPFP(featurizer_type)

    return generic_steps(trial, data, featurizer)

    

# OPTIMIZE THE PIPELINE
po = PipelineOptimization(direction='maximize', study_name='np_classifier_neural_fp', sampler=optuna.samplers.TPESampler(seed=42),
                          storage='sqlite:///np_classifier_neural_fp.db')
metric = Metric(f1_score, average="macro")
po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset, objective_steps=neural_npfp_steps,
            metric=metric, n_trials=50, data=train_dataset, save_top_n=5, trial_timeout=60*60*24)