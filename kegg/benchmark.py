import os
import shutil
from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
from copy import copy
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import optuna

from deepmol.compound_featurization import NPClassifierFP, NeuralNPFP, BiosynfoniKeys, MixedFeaturizer, MorganFingerprint, MHFP

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
    
class ValidationMultiLabel(Objective):
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

        # self.data = kwargs.pop('data')
        self.validation_dataset = kwargs.pop('validation_dataset')
        self.train_dataset = kwargs.pop('train_dataset')

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
        train_copy = copy(self.train_dataset)
        validation_copy = copy(self.validation_dataset)

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

from deepmol.pipeline_optimization._standardizer_objectives import _get_standardizer
from deepmol.pipeline_optimization._deepchem_models_objectives import *

def _optimize(datasets, steps, pipeline_name):
    
    folds = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    weighted_f1_scores = []
    weighted_precision_scores = []
    weigthed_recall_score = []
    reps = []

    os.makedirs(f"kegg_pathway_prediction_{pipeline_name}", exist_ok=True)

    rep = 0
    for tuple_datasets in datasets:
        fold_idx = 0
        for train_dataset, validation_dataset, test_dataset in tuple_datasets:
            labels_names = np.array(train_dataset._label_names)
            train_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]
            validation_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]
            test_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]

            # OPTIMIZE THE PIPELINE
            po = PipelineOptimization(direction='maximize', study_name=f'kegg_pathway_prediction_{pipeline_name}_fold_{fold_idx}_rep{rep}', sampler=optuna.samplers.TPESampler(seed=43),
                                    storage=f'sqlite:///kegg_pathway_prediction.db')
            metric = Metric(f1_score, average="macro")

            po.optimize(data=train_dataset, objective_steps=steps, train_dataset=train_dataset, validation_dataset=validation_dataset,
                        metric=metric, n_trials=20, save_top_n=1, trial_timeout=60*60*1, objective=ValidationMultiLabel)
            
            train_valid_merged = copy(train_dataset)
            train_valid_merged = train_valid_merged.merge([validation_dataset])
            po.best_pipeline.fit(train_valid_merged)
            predictions = po.best_pipeline.predict(test_dataset)

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

            results.to_csv(f"kegg_pathway_prediction_{pipeline_name}/kegg_pathway_prediction_{pipeline_name}_fold_{fold_idx}_rep{rep}.csv", index=False)

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

def optimize_for_dmpnn():

    def dmpnn_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        dmpnn_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = [('standardizer', _get_standardizer(trial))]
        featurizer = DMPNNFeat()
        fnn_layers = trial.suggest_int('fnn_layers', 1, 3)
        dmpnn_kwargs['fnn_layers'] = fnn_layers
        fnn_dropout_p = trial.suggest_float('fnn_dropout_p', 0.0, 0.5, step=0.25)
        dmpnn_kwargs['fnn_dropout_p'] = fnn_dropout_p
        depth = trial.suggest_int('depth_dmpnn', 2, 4)
        dmpnn_kwargs['depth'] = depth
        model = dmpnn_model(dmpnn_kwargs=dmpnn_kwargs, deepchem_kwargs=deepchem_kwargs)
        final_steps.extend([('featurizer', featurizer), ('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)

    # OPTIMIZE THE PIPELINE
    _optimize(datasets, dmpnn_steps, 'dmpnn')
    
def optimize_for_attentivefp():

    def attentivefp_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        attentive_fp_kwargs = {'n_tasks': n_tasks, 'mode': mode, 'n_classes': n_classes, 'batch_size': batch_size}
        
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = [('standardizer', _get_standardizer(trial))]

        use_chirality = trial.suggest_categorical('use_chirality_attentive_fp', [True, False])
        use_partial_charge = trial.suggest_categorical('use_partial_charge_attentive_fp', [True, False])
        # get number of trues in the list
        n_features = 30
        if use_chirality:
            n_features += 2
        if use_partial_charge:
            n_features += 1
        attentive_fp_kwargs['number_atom_features'] = n_features
        featurizer = MolGraphConvFeat(use_edges=True, use_chirality=use_chirality, use_partial_charge=use_partial_charge)
        # model
        num_layers = trial.suggest_int('num_layers_attentive_fp', 1, 5)
        attentive_fp_kwargs['num_layers'] = num_layers
        graph_feat_size = trial.suggest_int('graph_feat_size_attentive_fp', 100, 500, step=100)
        attentive_fp_kwargs['graph_feat_size'] = graph_feat_size
        dropout = trial.suggest_float('dropout_attentive_fp', 0.0, 0.5, step=0.25)
        attentive_fp_kwargs['dropout'] = dropout
        model = attentivefp_model( attentivefp_kwargs=attentive_fp_kwargs,
                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('featurizer', featurizer), ('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)

    # OPTIMIZE THE PIPELINE
    _optimize(datasets, attentivefp_steps, 'attentive_fp')
    
def optimize_for_np_classifier_fp():

    def np_classifier_fp_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = [('standardizer', _get_standardizer(trial))]
        featurizer = NPClassifierFP()
        n_features = len(featurizer.feature_names)
        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                              'batch_size': batch_size}

        dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['dropouts'] = dropouts
        layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
        robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
        bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
        model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('featurizer', featurizer), ('model', model)])
        return final_steps
    

    import pickle

    # Specify the filename
    filename = 'splits.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)

    # OPTIMIZE THE PIPELINE
    _optimize(datasets, np_classifier_fp_steps, 'np_classifier_fp')

def optimize_for_neural_npfp():

    def neural_npfp_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = [('standardizer', _get_standardizer(trial))]

        model_name = trial.suggest_categorical('model', ['aux', 'base', 'ae'])
        featurizer = NeuralNPFP(model_name=model_name)
        n_features = len(featurizer.feature_names)
        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                              'batch_size': batch_size}

        dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['dropouts'] = dropouts
        layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
        robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
        bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
        model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('featurizer', featurizer), ('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)

    # OPTIMIZE THE PIPELINE
    _optimize(datasets, neural_npfp_steps, 'neural_npfp')
    
def optimize_for_biosynfoni():

    def biosynfoni_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = [('standardizer', _get_standardizer(trial))]

        featurizer = BiosynfoniKeys()
        n_features = len(featurizer.feature_names)
        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                              'batch_size': batch_size}

        dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['dropouts'] = dropouts
        layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
        robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
        bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
        model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('featurizer', featurizer), ('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)

    # OPTIMIZE THE PIPELINE
    _optimize(datasets, biosynfoni_steps, 'biosynfoni_keys')

def optimize_for_np_bert():

    def np_bert_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = []

        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': 512, 'n_classes': n_classes,
                                              'batch_size': batch_size}

        dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['dropouts'] = dropouts
        layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
        robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
        bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
        model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits_npbert.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)


    # OPTIMIZE THE PIPELINE
    _optimize(datasets, np_bert_steps, 'np_bert_2')

def optimize_for_modern_bert():

    def modern_bert_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = []

        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': 1024, 'n_classes': n_classes,
                                              'batch_size': batch_size}

        dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['dropouts'] = dropouts
        layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
        robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
        bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
        model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits_modern_bert.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)


    # OPTIMIZE THE PIPELINE
    _optimize(datasets, modern_bert_steps, 'modern_bert_1')

def optimize_biosynfoni_np_classifierfp():

    def biosynfoni_np_classifier_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = [('standardizer', _get_standardizer(trial))]

        featurizer = MixedFeaturizer(featurizers=[BiosynfoniKeys(), NPClassifierFP()])
        n_features = len(featurizer.feature_names)
        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                              'batch_size': batch_size}

        dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['dropouts'] = dropouts
        layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
        robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
        bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
        model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('featurizer', featurizer), ('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)

    # OPTIMIZE THE PIPELINE
    _optimize(datasets, biosynfoni_np_classifier_steps, 'biosynfoni_np_classifier')

def optimize_for_morganfp():

    def morganfp_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = [('standardizer', _get_standardizer(trial))]

        featurizer = MorganFingerprint()
        n_features = len(featurizer.feature_names)
        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                              'batch_size': batch_size}

        dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['dropouts'] = dropouts
        layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
        robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
        bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
        model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('featurizer', featurizer), ('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)

    # OPTIMIZE THE PIPELINE
    _optimize(datasets, morganfp_steps, 'morganfp')

def optimize_for_mhfp():

    def mhfp_steps(trial, data):
        n_tasks = data.n_tasks
        mode = data.mode
        n_classes = len(set(data.y)) if mode == 'classification' else 1
        if isinstance(mode, list):
            if mode[0] == "classification":
                n_classes = len(set(data.y[0]))
            else:
                n_classes = 1

        if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        # , "multitask_irv_classifier_model",
        # "progressive_multitask_classifier_model", "robust_multitask_classifier_model", "sc_score_model"])
            mode = 'classification'
        elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
            # "progressive_multitask_regressor_model", "robust_multitask_regressor_model"])
            mode = 'regression'
        else:
            raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

        batch_size = trial.suggest_categorical("batch_size_deepchem", [8, 16, 32, 64, 128, 256, 512])
        epochs = trial.suggest_int("epochs_deepchem", 10, 200)
        deepchem_kwargs = {"epochs": epochs}
        final_steps = [('standardizer', _get_standardizer(trial))]

        featurizer = MHFP()
        n_features = len(featurizer.feature_names)
        robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                              'batch_size': batch_size}

        dropouts = trial.suggest_float('dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['dropouts'] = dropouts
        layer_sizes = trial.suggest_categorical('layer_sizes_robust_multitask_classifier', [str(cat) for cat in [[50], [100], [500], [200, 100]]])
        robust_multitask_classifier_kwargs['layer_sizes'] = eval(layer_sizes)
        bypass_dropouts = trial.suggest_float('bypass_dropout_robust_multitask_classifier', 0.0, 0.5, step=0.25)
        robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
        model = robust_multitask_classifier_model(
                                              robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                              deepchem_kwargs=deepchem_kwargs)

        final_steps.extend([('featurizer', featurizer), ('model', model)])
        return final_steps


    import pickle

    # Specify the filename
    filename = 'splits.pkl'

    # Read the data from the file using pickle
    with open(filename, 'rb') as file:
        datasets = pickle.load(file)

    # OPTIMIZE THE PIPELINE
    _optimize(datasets, mhfp_steps, 'mhfp')

optimize_for_neural_npfp()
optimize_for_morganfp()
optimize_for_biosynfoni()
optimize_for_np_classifier_fp()
optimize_for_np_bert()
optimize_for_modern_bert()
optimize_for_dmpnn()
optimize_biosynfoni_np_classifierfp()
optimize_for_mhfp()