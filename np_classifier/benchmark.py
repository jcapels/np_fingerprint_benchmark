from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
from copy import copy
import pandas as pd
from sklearn.metrics import f1_score
import optuna
from deepmol.loaders import CSVLoader

from deepmol.pipeline_optimization._standardizer_objectives import _get_standardizer
from deepmol.pipeline_optimization._deepchem_models_objectives import *
from deepmol.compound_featurization import NPClassifierFP, NeuralNPFP, BiosynfoniKeys, MorganFingerprint, MHFP  # Ensure this is the correct module for NPClassifierFP

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
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_dmpnn', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_dmpnn.db')
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=dmpnn_steps,
                metric=metric, n_trials=20, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
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
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_attentivefp', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_attentivefp.db')
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=attentivefp_steps,
                metric=metric, n_trials=20, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
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
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_np_classifier_fp', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_np_classifier_fp.db')
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=np_classifier_fp_steps,
                metric=metric, n_trials=20, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
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
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_neural_npfp', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_neural_npfp.db')
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=neural_npfp_steps,
                metric=metric, n_trials=20, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
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
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_biosynfoni', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_biosynfoni.db')
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=biosynfoni_steps,
                metric=metric, n_trials=20, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
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


    import numpy as np

    def featurize_dataset(dataset, np_bert):
        features = []
        for id_ in dataset.ids:
            features.append(np_bert[id_])
        features = np.array(features)
        dataset._X = features
        return dataset

    dataset = pd.read_csv("train_dataset.csv", nrows=2)
    labels = dataset.columns[2:]
    # LOAD THE DATA
    loader = CSVLoader('train_dataset.csv',
                    smiles_field='SMILES',labels_fields=labels, id_field='key')
    train_dataset = loader.create_dataset(sep=",")
    train_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]

    loader = CSVLoader('validation_dataset.csv',
                    smiles_field='SMILES',labels_fields=labels, id_field='key')
    validation_dataset = loader.create_dataset(sep=",")
    validation_dataset._label_names = [str(i) for i, label_ in enumerate(validation_dataset._label_names)]

    # LOAD THE NP-BERT EMBEDDINGS
    import pickle
    with open('np_bert.pkl', 'rb') as file:
        np_bert = pickle.load(file)

    # Featurize the datasets
    train_dataset = featurize_dataset(train_dataset, np_bert)
    validation_dataset = featurize_dataset(validation_dataset, np_bert)


    # OPTIMIZE THE PIPELINE
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_np_bert', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_np_bert.db', load_if_exists=True)
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=np_bert_steps,
                metric=metric, n_trials=12, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
def optimize_for_modernbert():

    def modernbert_steps(trial, data):
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


    import numpy as np

    def featurize_dataset(dataset, np_bert):
        features = []
        for id_ in dataset.ids:
            features.append(np_bert[id_])
        features = np.array(features)
        dataset._X = features
        return dataset

    dataset = pd.read_csv("train_dataset.csv", nrows=2)
    labels = dataset.columns[2:]
    # LOAD THE DATA
    loader = CSVLoader('train_dataset.csv',
                    smiles_field='SMILES',labels_fields=labels, id_field='key')
    train_dataset = loader.create_dataset(sep=",")
    train_dataset._label_names = [str(i) for i, label_ in enumerate(train_dataset._label_names)]

    loader = CSVLoader('validation_dataset.csv',
                    smiles_field='SMILES',labels_fields=labels, id_field='key')
    validation_dataset = loader.create_dataset(sep=",")
    validation_dataset._label_names = [str(i) for i, label_ in enumerate(validation_dataset._label_names)]

    # LOAD THE NP-BERT EMBEDDINGS
    import pickle
    with open('modern_bert.pkl', 'rb') as file:
        np_bert = pickle.load(file)

    # Featurize the datasets
    train_dataset = featurize_dataset(train_dataset, np_bert)
    validation_dataset = featurize_dataset(validation_dataset, np_bert)


    # OPTIMIZE THE PIPELINE
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_modern_bert', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_modern_bert.db', load_if_exists=True)
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=modernbert_steps,
                metric=metric, n_trials=12, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
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
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_morganfp', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_morganfp.db')
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=morganfp_steps,
                metric=metric, n_trials=20, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
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
    po = PipelineOptimization(direction='maximize', study_name='npclassifier_pathway_prediction_mhfp', sampler=optuna.samplers.TPESampler(seed=43),
                            storage='sqlite:///npclassifier_pathway_prediction_mhfp.db')
    metric = Metric(f1_score, average="macro")

    data = copy(train_dataset)
    data._label_names = [str(i) for i, label_ in enumerate(data._label_names)]

    po.optimize(train_dataset=train_dataset, test_dataset=validation_dataset,objective_steps=mhfp_steps,
                metric=metric, n_trials=20, data=data, save_top_n=5, trial_timeout=60*60*24
                )
    
# optimize_for_dmpnn()
# optimize_for_attentivefp()np_bert
# optimize_for_np_classifier_fp()
# optimize_for_neural_npfp()
# optimize_for_biosynfoni()
# optimize_for_np_bert()
# optimize_for_modernbert()
# optimize_for_morganfp()
optimize_for_mhfp()

