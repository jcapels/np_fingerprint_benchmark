# Benchmark of molecular fingerprints for natural product discovery using DeepMol

## Table of Contents

- [Create conda environment](#create-conda-environment)
- [Installation](#installation)
  - [From GitHub](#from-github)
- [Run pipeline](#run-pipeline)


## Create conda environment
    
```bash
conda create -n np_benchmark python=3.11
conda activate np_benchmark

pip install -r requirements.txt
python setup.py install
```

## Installation

### From github

```bash
pip install git+https://github.com/jcapels/np_fingerprint_benchmark.git
```

## Obtain the data

To obtain the data for PlantCyc and KEGG, run the following scripts, respectively: [plantcyc/get_data_from_plantcyc.py](plantcyc/get_data_from_plantcyc.py) and [kegg/get_data_from_kegg.py](kegg/get_data_from_kegg.py)

For the NPClassifier dataset, download the dataset in https://www.dropbox.com/s/y25rl9kuggpzly5/datset_class_all_V1.pkl?dl=0 and paste it in [np_classifier/Data](np_classifier/Data) and run the create_dataset.ipynb notebook. 

The data relative to the precursors is present in this repository but it was taken from https://github.com/jcapels/SMPrecursorPredictor/tree/main/models_and_datasets/final_dataset. 

The data for the masked learning was obtained from the official website of LOTUSDB and COCONUT. These data was placed in the respective data folders in the masked_learning folder. Then, masked_learning/integration_of_data_masked_learning.ipynb should be run.

## Masked Learning

To run and generate each model, you can run the scripts in [masked_learning](masked_learning).

## Supervised Learning and Benchmark

A bash script that builds and runs a containerized Python benchmark using Podman with GPU support.
 
### Prerequisites
 
- Podman installed and configured
- NVIDIA GPU with `nvidia.com/gpu` device available
- A `Dockerfile` in the current directory
- Python script to execute in the container

### Usage
 
```bash
./run.sh <GPU_NUMBER> <FOLDER_NAME> <PYTHON_SCRIPT> [CONTAINER_NAME]
```
 
#### Required Arguments
 
| Argument | Description |
|----------|-------------|
| `GPU_NUMBER` | NVIDIA GPU device ID (e.g., `0`, `1`) |
| `FOLDER_NAME` | Name of the folder containing your data/code (mounted as `/workspace/<FOLDER_NAME>/` in container) |
| `PYTHON_SCRIPT` | Name of the Python script to execute (e.g., `benchmark.py`) |
 
#### Optional Arguments
 
| Argument | Description |
|----------|-------------|
| `CONTAINER_NAME` | Custom name for the container. If omitted, defaults to `<FOLDER_NAME>` |
 
### Examples
 
#### Basic usage (container named after folder)
```bash
./run.sh 0 np_classifier benchmark.py
```
- Uses GPU 0
- Mounts `/home/jcapela/np_fingerprint_benchmark/my_experiment/`
- Runs `benchmark.py` 
- Container is named `my_experiment`
#### Custom container name
```bash
./run.sh 1 my_experiment benchmark.py my_custom_container
```
- Uses GPU 1
- Container is named `my_custom_container` instead of `my_experiment`

### What the Script Does
 
1. **Validates arguments** — ensures 3 or 4 arguments are provided
2. **Sets container name** — uses the optional 4th argument, or defaults to `FOLDER_NAME`
3. **Builds the image** — creates a `fingerprints_benchmark` Docker image from the local `Dockerfile`
4. **Runs the container** — executes with the following configuration:
   - **Volume mount**: Binds `$HOME/np_fingerprint_benchmark/<FOLDER_NAME>/` to `/workspace/<FOLDER_NAME>/` inside the container
   - **GPU access**: Grants access to the specified NVIDIA GPU
   - **Detached mode** (`-d`): Runs in background
   - **Working directory**: Changes to `/workspace/<FOLDER_NAME>/` before executing the script
   - **Output redirection**: Captures all output to `output.txt` in the mounted folder

 
### Troubleshooting
 
#### "Usage" error
Make sure you provide exactly 3 or 4 arguments and that the script has execute permissions:
```bash
chmod +x script.sh
```
 
#### GPU not found
Verify GPU availability:
```bash
podman run --rm --device nvidia.com/gpu=0 nvidia/cuda nvidia-smi
```
 
#### Container already exists
Remove the existing container before running:
```bash
podman rm <CONTAINER_NAME>
```
 
### Permission denied on volume mount
Ensure the folder exists and you have read/write permissions:
```bash
mkdir -p ~/np_fingerprint_benchmark/<FOLDER_NAME>
```
 
### Notes
 
- The script runs containers in **detached mode** — use `podman logs <CONTAINER_NAME>` to view real-time output
- The `Dockerfile` must be in the current directory when running the script
- The volume path is hardcoded to `/home/jcapela/np_fingerprint_benchmark/` — modify the script if you need a different path


## Similarity analysis




