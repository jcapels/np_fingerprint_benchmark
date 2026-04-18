from deepmol.pipeline import Pipeline
from deepmol.compound_featurization import NPClassifierFP
from deepmol.standardizer._utils import simple_standardisation
from deepmol.standardizer import CustomStandardizer
from deepmol.pipeline_optimization._deepchem_models_objectives import *
import pandas as pd

standardizer = CustomStandardizer(params=simple_standardisation)

def train_and_save_pipeline(data):
    n_tasks = data.n_tasks
    mode = data.mode
    n_classes = len(set(data.y)) if mode == 'classification' else 1
    if isinstance(mode, list):
        if mode[0] == "classification":
            n_classes = len(set(data.y[0]))
        else:
            n_classes = 1

    if mode == 'classification' or (len(set(mode)) == 1 and mode[0] == 'classification'):
        mode = 'classification'
    elif mode == 'regression' or (len(set(mode)) == 1 and mode[0] == 'regression'):
        mode = 'regression'
    else:
        raise ValueError("data mode must be either 'classification' or 'regression' or a list of both")

    batch_size = 8
    epochs = 127
    deepchem_kwargs = {"epochs": epochs}
    final_steps = [('standardizer', standardizer)]
    featurizer = NPClassifierFP()
    n_features = len(featurizer.feature_names)
    robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes,
                                            'batch_size': batch_size}

    dropouts = 0.5
    robust_multitask_classifier_kwargs['dropouts'] = dropouts
    layer_sizes = [500]
    robust_multitask_classifier_kwargs['layer_sizes'] = layer_sizes
    bypass_dropouts = 0.5
    robust_multitask_classifier_kwargs['bypass_dropouts'] = bypass_dropouts
    model = robust_multitask_classifier_model(
                                            robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs,
                                            deepchem_kwargs=deepchem_kwargs)

    final_steps.extend([('featurizer', featurizer), ('model', model)])

    pipeline = Pipeline(steps=final_steps, path="plantcyc_robust_multitask_classifier_all_data")
    pipeline.fit(data)
    pipeline.save()

if __name__ == "__main__":
    from deepmol.loaders import CSVLoader

    dataset = pd.read_csv("plant_cyc_pathways_w_labels_dedup.csv", nrows=2)
    labels = dataset.columns[4:]
    # LOAD THE DATA
    loader = CSVLoader('plant_cyc_pathways_w_labels_dedup.csv', 
                    smiles_field='SMILES',labels_fields=labels, id_field="Metabolite")
    dataset = loader.create_dataset(sep=",")
    train_and_save_pipeline(dataset)