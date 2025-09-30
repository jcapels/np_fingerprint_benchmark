from deepmol.pipeline import Pipeline
from deepmol.compound_featurization import NPClassifierFP

import tensorflow as tf
from deepmol.models import KerasModel

import os
import random
import numpy as np
import pandas as pd
from deepmol.loaders import CSVLoader
from deepmol.pipeline import Pipeline
from deepmol.datasets import SmilesDataset
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
import torch

def set_gpu(gpu_ids_list):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpus_used = [gpus[i] for i in gpu_ids_list]
            tf.config.set_visible_devices(gpus_used, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

set_gpu([0])

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

def top_k_categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

def model_build(): # num = number of categories
    input_f = layers.Input(shape=(6144,))
    # input_b = layers.Input(shape=(4096,))
    # input_fp = layers.Concatenate()([input_f,input_b])
    
    X = layers.Dense(2048, activation = 'relu')(input_f)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(3072, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.Dropout(0.2)(X)
    output = layers.Dense(730, activation = 'sigmoid')(X)
    model = keras.Model(inputs = [input_f], outputs = output)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss=['binary_crossentropy'])
    return model


def run_for_original_split():
    import pandas as pd

    dataset = pd.read_csv("train_dataset.csv")

    from deepmol.datasets import SmilesDataset


    dataset = pd.read_csv("train_dataset.csv")
    labels = dataset.columns[2:]

    validation_dataset = pd.read_csv("validation_dataset.csv")

    train_dataset = pd.concat([dataset, validation_dataset])

    if os.path.exists("np_classifier_original.csv"):
        results = pd.read_csv("np_classifier_original.csv")
    else:

        results = pd.DataFrame(columns=["rep", "f1", "recall", "precision"])

    for j in range(5):

        # Set the random seed
        tf.random.set_seed(42)

        # Set a manual seed
        torch.manual_seed(j)
        torch.cuda.manual_seed(j)

        # Set seed for Python's random module
        random.seed(j)

        # Set seed for NumPy
        np.random.seed(j)

        train_dataset_ = train_dataset.sample(frac=1, random_state=j)
        label_names = [str(i) for i, label_ in enumerate(labels)]

        train_dataset_ = SmilesDataset(smiles=train_dataset_.SMILES, y=train_dataset_.iloc[:, 2:],
                                        label_names=label_names, mode="multilabel")


        loader = CSVLoader('test_dataset.csv',
                        smiles_field='SMILES',labels_fields=labels, mode="multilabel")
        test_dataset = loader.create_dataset(sep=",")
        test_dataset._label_names = [str(i) for i, label_ in enumerate(test_dataset._label_names)]

        pipeline = Pipeline(steps=[("np fingerprint", NPClassifierFP()), 
                ("model", KerasModel(model_builder=model_build(), batch_size=128, epochs=100,
                                        mode="multilabel"))], path="np_classifier_trained").fit(train_dataset_)
        
        predictions = pipeline.predict(test_dataset)

        f1_macro = f1_score(test_dataset.y, predictions, average="macro")
        recall_macro = recall_score(test_dataset.y, predictions, average="macro")
        precision_macro = precision_score(test_dataset.y, predictions, average="macro")

        f1_macro_pathway = f1_score(test_dataset.y[:, 7], predictions[:, 7], average="macro")
        recall_macro_pathway = recall_score(test_dataset.y[:, 7], predictions[:, 7], average="macro")
        precision_macro_pathway = precision_score(test_dataset.y[:, 7], predictions[:, 7], average="macro")

        f1_macro_superclass = f1_score(test_dataset.y[:, 7:77], predictions[:, 7:77], average="macro")
        recall_macro_superclass = recall_score(test_dataset.y[:, 7:77], predictions[:, 7:77], average="macro")
        precision_macro_superclass = precision_score(test_dataset.y[:, 7:77], predictions[:, 7:77], average="macro")

        f1_macro_class = f1_score(test_dataset.y[:, 77:], predictions[:, 77:], average="macro")
        recall_macro_class = recall_score(test_dataset.y[:, 77:], predictions[:, 77:], average="macro")
        precision_macro_class = precision_score(test_dataset.y[:, 77:], predictions[:, 77:], average="macro")


        results = pd.concat([results, pd.DataFrame({
                "rep": [j],
                "f1": [f1_macro],
                "recall": [recall_macro],
                "precision": [precision_macro],
                "f1_pathway": [f1_macro_pathway],
                "recall_pathway": [recall_macro_pathway],
                "precision_pathway": [precision_macro_pathway],
                "f1_superclass": [f1_macro_superclass],
                "recall_superclass": [recall_macro_superclass],
                "precision_superclass": [precision_macro_superclass],
                "f1_class": [f1_macro_class],
                "recall_class": [recall_macro_class],
                "precision_class": [precision_macro_class]

            })])
        results.to_csv(f"np_classifier_original.csv", index=False)

        tf.keras.backend.clear_session()

def run_for_ginestra_split():
    import pandas as pd

    dataset = pd.read_csv("ginestra_data/train_ginestra.csv")

    from deepmol.datasets import SmilesDataset


    dataset = pd.read_csv("ginestra_data/train_ginestra.csv")
    labels = dataset.columns[2:]

    validation_dataset = pd.read_csv("ginestra_data/val_ginestra.csv")

    train_dataset = pd.concat([dataset, validation_dataset])

    if os.path.exists("np_classifier_original_ginestra.csv"):
        results = pd.read_csv("np_classifier_original_ginestra.csv")
    else:

        results = pd.DataFrame(columns=["rep", "f1", "recall", "precision"])

    for j in range(5):

        # Set the random seed
        tf.random.set_seed(42)

        # Set a manual seed
        torch.manual_seed(j)
        torch.cuda.manual_seed(j)

        # Set seed for Python's random module
        random.seed(j)

        # Set seed for NumPy
        np.random.seed(j)

        train_dataset_ = train_dataset.sample(frac=1, random_state=j)
        label_names = [str(i) for i, label_ in enumerate(labels)]

        train_dataset_ = SmilesDataset(smiles=train_dataset_.SMILES, y=train_dataset_.iloc[:, 2:],
                                        label_names=label_names, mode="multilabel")


        loader = CSVLoader('ginestra_data/test_ginestra.csv',
                        smiles_field='SMILES',labels_fields=labels, mode="multilabel")
        test_dataset = loader.create_dataset(sep=",")
        test_dataset._label_names = [str(i) for i, label_ in enumerate(test_dataset._label_names)]

        pipeline = Pipeline(steps=[("np fingerprint", NPClassifierFP()), 
                ("model", KerasModel(model_builder=model_build(), batch_size=128, epochs=100,
                                        mode="multilabel"))], path="np_classifier_trained").fit(train_dataset_)
        
        predictions = pipeline.predict(test_dataset)

        f1_macro = f1_score(test_dataset.y, predictions, average="macro")
        recall_macro = recall_score(test_dataset.y, predictions, average="macro")
        precision_macro = precision_score(test_dataset.y, predictions, average="macro")

        f1_macro_pathway = f1_score(test_dataset.y[:, 7], predictions[:, 7], average="macro")
        recall_macro_pathway = recall_score(test_dataset.y[:, 7], predictions[:, 7], average="macro")
        precision_macro_pathway = precision_score(test_dataset.y[:, 7], predictions[:, 7], average="macro")

        f1_macro_superclass = f1_score(test_dataset.y[:, 7:77], predictions[:, 7:77], average="macro")
        recall_macro_superclass = recall_score(test_dataset.y[:, 7:77], predictions[:, 7:77], average="macro")
        precision_macro_superclass = precision_score(test_dataset.y[:, 7:77], predictions[:, 7:77], average="macro")

        f1_macro_class = f1_score(test_dataset.y[:, 77:], predictions[:, 77:], average="macro")
        recall_macro_class = recall_score(test_dataset.y[:, 77:], predictions[:, 77:], average="macro")
        precision_macro_class = precision_score(test_dataset.y[:, 77:], predictions[:, 77:], average="macro")


        results = pd.concat([results, pd.DataFrame({
                "rep": [j],
                "f1": [f1_macro],
                "recall": [recall_macro],
                "precision": [precision_macro],
                "f1_pathway": [f1_macro_pathway],
                "recall_pathway": [recall_macro_pathway],
                "precision_pathway": [precision_macro_pathway],
                "f1_superclass": [f1_macro_superclass],
                "recall_superclass": [recall_macro_superclass],
                "precision_superclass": [precision_macro_superclass],
                "f1_class": [f1_macro_class],
                "recall_class": [recall_macro_class],
                "precision_class": [precision_macro_class]

            })])
        results.to_csv(f"np_classifier_original_ginestra.csv", index=False)

        tf.keras.backend.clear_session()

# run_for_original_split()
run_for_ginestra_split()

