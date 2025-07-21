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

def evaluate_pipeline(pipeline_path, name):

    dataset = pd.read_csv("train_dataset.csv")
    labels = dataset.columns[2:]

    validation_dataset = pd.read_csv("validation_dataset.csv")

    train_dataset = pd.concat([dataset, validation_dataset])

    if os.path.exists(f"{name}.csv"):
        results = pd.read_csv(f"{name}.csv")
    else:

        results = pd.DataFrame(columns=["rep", "f1", "recall", "precision"])

    for j in range(4, 5):

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
                                       label_names=label_names)


        loader = CSVLoader('test_dataset.csv',
                        smiles_field='SMILES',labels_fields=labels)
        test_dataset = loader.create_dataset(sep=",")
        test_dataset._label_names = [str(i) for i, label_ in enumerate(test_dataset._label_names)]


        pipeline = Pipeline.load(pipeline_path)
        pipeline.fit(train_dataset_)

        predictions = pipeline.predict(test_dataset)

        f1_macro = f1_score(test_dataset.y, predictions, average="macro")
        recall_macro = recall_score(test_dataset.y, predictions, average="macro")
        precision_macro = precision_score(test_dataset.y, predictions, average="macro")

        results = pd.concat([results, pd.DataFrame({
            "rep": [j],
            "f1": [f1_macro],
            "recall": [recall_macro],
            "precision": [precision_macro]
        })])
        results.to_csv(f"{name}.csv", index=False)

        tf.keras.backend.clear_session()

def evaluate_pipeline_bert(pipeline_path, name, bert_path, bert_test_path):

    import numpy as np

    def featurize_dataset(dataset, np_bert):
        features = []
        for id_ in dataset.ids:
            features.append(np_bert[id_])
        features = np.array(features)
        dataset._X = features
        return dataset

    import pickle
    with open(bert_path, 'rb') as file:
        np_bert = pickle.load(file)

    import pickle
    with open(bert_test_path, 'rb') as file:
        np_bert_test = pickle.load(file)

    dataset = pd.read_csv("train_dataset.csv")
    labels = dataset.columns[2:]

    validation_dataset = pd.read_csv("validation_dataset.csv")

    train_dataset = pd.concat([dataset, validation_dataset])

    if os.path.exists(f"{name}.csv"):
        results = pd.read_csv(f"{name}.csv")
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
                                       label_names=label_names, ids=train_dataset_.key)

        # Featurize the datasets
        train_dataset_ = featurize_dataset(train_dataset_, np_bert)

        loader = CSVLoader('test_dataset.csv',
                        smiles_field='SMILES',labels_fields=labels, id_field='key')
        test_dataset = loader.create_dataset(sep=",")
        test_dataset._label_names = [str(i) for i, label_ in enumerate(test_dataset._label_names)]

        # Featurize the test dataset
        test_dataset = featurize_dataset(test_dataset, np_bert_test)

        pipeline = Pipeline.load(pipeline_path)
        pipeline.fit(train_dataset_)

        predictions = pipeline.predict(test_dataset)

        f1_macro = f1_score(test_dataset.y, predictions, average="macro")
        recall_macro = recall_score(test_dataset.y, predictions, average="macro")
        precision_macro = precision_score(test_dataset.y, predictions, average="macro")

        results = pd.concat([results, pd.DataFrame({
            "rep": [j],
            "f1": [f1_macro],
            "recall": [recall_macro],
            "precision": [precision_macro]
        })])
        results.to_csv(f"{name}.csv", index=False)

        tf.keras.backend.clear_session()
        
# evaluate_pipeline("npclassifier_pathway_prediction_biosynfoni/trial_12", "np_classifier_biosynfoni")
# evaluate_pipeline("npclassifier_pathway_prediction_attentivefp/trial_14", "np_classifier_attentivefp")
# evaluate_pipeline("npclassifier_pathway_prediction_dmpnn/trial_12", "np_classifier_dmpnn")
# evaluate_pipeline("npclassifier_pathway_prediction_neural_npfp/trial_15", "np_classifier_neural_npfp")
# evaluate_pipeline("npclassifier_pathway_prediction_np_classifier_fp/trial_11", "np_classifier_np_classifier_fp")
# evaluate_pipeline_bert("npclassifier_pathway_prediction_np_bert/trial_14", "np_classifier_np_bert", "np_bert.pkl", "np_bert_test_set.pkl")
# evaluate_pipeline_bert("npclassifier_pathway_prediction_modern_bert/trial_0", "np_classifier_modern_bert", "modern_bert.pkl", "modern_bert_test_set.pkl")
evaluate_pipeline("npclassifier_pathway_prediction_morganfp/trial_2", "np_classifier_morganfp")

