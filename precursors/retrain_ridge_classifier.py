from copy import copy
import os

from deepmol.pipeline import Pipeline
from deepmol.metrics import Metric
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def run(datasets, pipeline_name="Capela_2025"):
    
    folds = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    weighted_f1_scores = []
    weighted_precision_scores = []
    weigthed_recall_score = []
    reps = []
    pipeline = Pipeline.load("ridge_classifier_layered_fingerprints_variance_selector")

    os.makedirs(f"plant_precursor_prediction_{pipeline_name}", exist_ok=True)

    rep = 0
    for tuple_datasets in datasets:
        fold_idx = 0
        for train_dataset, validation_dataset, test_dataset in tuple_datasets:
            labels_names = np.array(train_dataset._label_names)
            train_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]
            validation_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]
            test_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]

            # OPTIMIZE THE PIPELINE
            metric = Metric(f1_score, average="macro")
            
            train_valid_merged = copy(train_dataset)
            train_valid_merged = train_valid_merged.merge([validation_dataset])
            pipeline.fit(train_valid_merged)
            predictions = pipeline.predict(test_dataset)

            train_labels = np.array(train_valid_merged.y)
            valid_labels = np.array(test_dataset.y)

            # Check which labels have at least one '1' in both train and validation datasets
            labels_with_ones = []
            for i in range(train_labels.shape[1]):  # Assuming labels are in one-hot encoded format
                if np.any(train_labels[:, i] == 1) and np.any(valid_labels[:, i] == 1):
                    labels_with_ones.append(i)

            y_pred = predictions[:, labels_with_ones]
            y_true = test_dataset.y[:, labels_with_ones]
            f1 = metric.compute_metric(y_true, y_pred, n_tasks=y_true.shape[1])[0]

            recall = Metric(recall_score, average="macro")
            precision = Metric(precision_score, average="macro")
            recall_score_ = recall.compute_metric(y_true, y_pred, n_tasks=y_true.shape[1])[0]
            precision_score_ = precision.compute_metric(y_true, y_pred, n_tasks=y_true.shape[1])[0]

            wf1 = Metric(f1_score, average="weighted")
            wrecall = Metric(recall_score, average="weighted")
            wprecision = Metric(precision_score, average="weighted")
            wrecall_score_ = wrecall.compute_metric(y_true, y_pred, n_tasks=y_true.shape[1])[0]
            wprecision_score_ = wprecision.compute_metric(y_true, y_pred, n_tasks=y_true.shape[1])[0]
            wf1_score_ = wf1.compute_metric(y_true, y_pred, n_tasks=y_true.shape[1])[0]

            f1_scores.append(f1)
            precision_scores.append(precision_score_)
            recall_scores.append(recall_score_)
            weighted_f1_scores.append(wf1_score_)
            weighted_precision_scores.append(wprecision_score_)
            weigthed_recall_score.append(wrecall_score_)

            f1 = Metric(f1_score)
            recall = Metric(recall_score)
            precision = Metric(precision_score)

            results = pd.DataFrame()
            results["metrics"] = ["f1", "recall", "precision"]
            for i, label in enumerate(labels_names):
                f1_ = f1.compute_metric(y_true[:, i], y_pred[:, i], n_tasks=1)[0]
                recall_ = recall.compute_metric(y_true[:, i], y_pred[:, i], n_tasks=1)[0]
                precision_ = precision.compute_metric(y_true[:, i], y_pred[:, i], n_tasks=1)[0]
                results[label] = [f1_, recall_, precision_]

            results.to_csv(f"plant_precursor_prediction_{pipeline_name}/plant_precursor_{pipeline_name}_fold_{fold_idx}_rep{rep}.csv", index=False)

            folds.append(fold_idx)
            reps.append(rep)

            fold_idx += 1
            # create dataframe with the results
            results = {
                'fold': folds,
                'f1_score': f1_scores,
                'precision_score': precision_scores,
                'recall_score': recall_scores,
                'wF1': weighted_f1_scores,
                'wPrecision': weighted_precision_scores,
                'wRecall': weigthed_recall_score,
                'rep': reps
            }
            df = pd.DataFrame(results)
            # save the dataframe to a csv file
            df.to_csv(f'{pipeline_name}_results.csv', index=False)

        rep += 1

import pickle

# Specify the filename
filename = 'splits.pkl'

# Read the data from the file using pickle
with open(filename, 'rb') as file:
    datasets = pickle.load(file)
run(datasets)