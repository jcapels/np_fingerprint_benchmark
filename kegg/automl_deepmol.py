import os
import shutil
from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
from copy import copy
from sklearn.metrics import f1_score
import optuna
import pandas as pd

import re

from deepmol.pipeline_optimization.objective_wrapper import Objective
from deepmol.pipeline import Pipeline

import numpy as np


class FiveFoldCrossValidation(Objective):
    """
    
    Parameters
    ----------
    objective_steps : callable
        Function that returns the steps of the pipeline for a given trial.
    study : optuna.study.Study
        Study object that stores the optimization history.
    direction : str or optuna.study.StudyDirection
        Direction of the optimization (minimize or maximize).
    save_top_n : int
        Number of best pipelines to save.
    **kwargs
        Additional keyword arguments passed to the objective_steps function.
    """

    def __init__(self, objective_steps, study, direction, save_top_n,
                 trial_timeout=86400, **kwargs):
        """
        Initialize the objective function.
        """
        super().__init__(objective_steps, study, direction, save_top_n, trial_timeout=trial_timeout)
        self.metric = kwargs.pop('metric')
        # the datasets or any additional data can be accessed through kwargs. Here is an example:

        self.datasets = kwargs.pop('datasets')

        # and passed as input of the PipelineOptimization().optimize(data=data). See below in the example of optimization calling.

        self.kwargs = kwargs

    def _run(self, trial: optuna.Trial):
        """
        Call the objective function.
        """
        # train the pipeline 
        trial_id = str(trial.number)
        path = os.path.join(self.save_dir, f'trial_{trial_id}')

        pipeline = Pipeline(steps=self.objective_steps(trial, **self.kwargs), path=path)
        scores = []
        for train, validation, test in self.datasets:
            train_copy = copy(train)
            validation_copy = copy(validation)

            train_copy._label_names = [str(i) for i, label_ in enumerate(train_copy._label_names)]
            validation_copy._label_names = [str(i) for i, label_ in enumerate(train_copy._label_names)]

            if pipeline.steps[-1][1].__class__.__name__ == 'KerasModel':
                pipeline.fit(train_copy, validation_dataset=validation_copy)
            else:
                pipeline.fit(train_copy)

            # Convert labels to numpy arrays if they are not already
            train_labels = np.array(train_copy.y)
            valid_labels = np.array(validation_copy.y)

            # Check which labels have at least one '1' in both train and validation datasets
            labels_with_ones = []
            for i in range(train_labels.shape[1]):  # Assuming labels are in one-hot encoded format
                if np.any(train_labels[:, i] == 1) and np.any(valid_labels[:, i] == 1):
                    labels_with_ones.append(i)

            y_pred = pipeline.predict(validation_copy)
            y_pred = y_pred[:, labels_with_ones]
            y_true = validation_copy.y[:, labels_with_ones]
            score = self.metric.compute_metric(y_true, y_pred, n_tasks=y_true.shape[1])[0]

            # score = pipeline.evaluate(validation_copy, [self.metric])[0][self.metric.name]
            if score is None:
                score = float('-inf') if self.direction == 'maximize' else float('inf')

            scores.append(score)

        score = np.array(scores).mean()
        best_scores = self.study.user_attrs['best_scores']
        min_score = min(best_scores.values()) if len(best_scores) > 0 else float('inf')
        max_score = max(best_scores.values()) if len(best_scores) > 0 else float('-inf')
        update_score = (self.direction == 'maximize' and score > min_score) or (
                self.direction == 'minimize' and score < max_score)

        if len(best_scores) < self.save_top_n or update_score:
            pipeline.save()
            best_scores.update({trial_id: score})

            if len(best_scores) > self.save_top_n:
                if self.direction == 'maximize':
                    min_score_id = min(best_scores, key=best_scores.get)
                    del best_scores[min_score_id]
                    if os.path.exists(os.path.join(self.save_dir, f'trial_{min_score_id}')):
                        shutil.rmtree(os.path.join(self.save_dir, f'trial_{min_score_id}'))
                else:
                    max_score_id = max(best_scores, key=best_scores.get)
                    del best_scores[max_score_id]
                    if os.path.exists(os.path.join(self.save_dir, f'trial_{max_score_id}')):
                        shutil.rmtree(os.path.join(self.save_dir, f'trial_{max_score_id}'))

        self.study.set_user_attr('best_scores', best_scores)
        return score
        

import pickle

# Specify the filename
filename = 'splits.pkl'

# Read the data from the file using pickle
with open(filename, 'rb') as file:
    datasets = pickle.load(file)

# OPTIMIZE THE PIPELINE
po = PipelineOptimization(direction='maximize', study_name='kegg_pathway_prediction_seed43', sampler=optuna.samplers.TPESampler(seed=43),
                          storage='sqlite:///kegg_pathway_prediction_seed43.db')
metric = Metric(f1_score, average="macro")
po.optimize(datasets=datasets, objective_steps='deepchem',
            metric=metric, n_trials=50, data=datasets[0][0], save_top_n=5, trial_timeout=60*60*24,
            objective=FiveFoldCrossValidation)