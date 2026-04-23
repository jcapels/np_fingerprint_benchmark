# Benchmarking Molecular Fingerprints for Natural Product Discovery with DeepMol

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Data Acquisition](#data-acquisition)
- [Masked Learning](#masked-learning)
- [Supervised Learning and Benchmark](#supervised-learning-and-benchmark)
- [Similarity Analysis](#similarity-analysis)

---

## Environment Setup

Create and activate a Conda environment with Python 3.11:

```bash
conda create -n np_benchmark python=3.11
conda activate np_benchmark
```

Install dependencies and set up the package:

```bash
pip install -r requirements.txt
python setup.py install
```

---

## Installation

### From GitHub

Install directly from the repository:

```bash
pip install git+https://github.com/jcapels/np_fingerprint_benchmark.git
```

---

## Data Acquisition

### PlantCyc and KEGG
Run the following scripts to obtain data:
- [PlantCyc](plantcyc/get_data_from_plantcyc.py)
- [KEGG](kegg/get_data_from_kegg.py)

### NPClassifier Dataset
1. Download the dataset: [dataset_class_all_V1.pkl](https://www.dropbox.com/s/y25rl9kuggpzly5/datset_class_all_V1.pkl?dl=0)
2. Place it in [np_classifier/Data](np_classifier/Data)
3. Run create_dataset.ipynb`

### Precursors
The precursor data is included in this repository, originally sourced from [SMPrecursorPredictor](https://github.com/jcapels/SMPrecursorPredictor/tree/main/models_and_datasets/final_dataset).

### Masked Learning Data
Data for masked learning was obtained from [LOTUSDB](https://lotus.naturalproducts.net/) and [COCONUT](https://coconut.naturalproducts.net/). Place the data in the respective folders within `masked_learning/` and run `integration_of_data_masked_learning.ipynb`.

---

## Masked Learning

Generate and run models using the scripts in the [masked_learning](masked_learning) directory.

---

## Supervised Learning and Benchmark

### Overview
This section describes how to run a containerized Python benchmark using Podman with GPU support.

### Prerequisites
- Podman installed and configured
- NVIDIA GPU with `nvidia.com/gpu` device available
- `Dockerfile` in the current directory
- Python script to execute in the container

### Usage

```bash
./run.sh <GPU_NUMBER> <FOLDER_NAME> <PYTHON_SCRIPT> [CONTAINER_NAME]
```

#### Required Arguments
   Argument         | Description                                                                 |
 |------------------|-----------------------------------------------------------------------------|
 | `GPU_NUMBER`     | NVIDIA GPU device ID (e.g., `0`, `1`)                                     |
 | `FOLDER_NAME`    | Name of the folder containing your data/code (mounted as `/workspace/<FOLDER_NAME>/`) |
 | `PYTHON_SCRIPT`  | Name of the Python script to execute (e.g., `benchmark.py`)               |

#### Optional Arguments
 | Argument          | Description                                                                 |
 |-------------------|-----------------------------------------------------------------------------|
 | `CONTAINER_NAME`  | Custom name for the container. Defaults to `<FOLDER_NAME>` if omitted.      |

---

#### Example Usage

**Basic Usage (container named after folder):**
```bash
./run.sh 0 np_classifier benchmark.py
```
- Uses GPU 0
- Mounts `/home/jcapela/np_fingerprint_benchmark/np_classifier/`
- Runs `benchmark.py`
- Container named `np_classifier`

**Custom Container Name:**
```bash
./run.sh 1 my_experiment benchmark.py my_custom_container
```
- Uses GPU 1
- Container named `my_custom_container`

---

### Script Workflow
1. **Validates arguments** — Ensures 3 or 4 arguments are provided.
2. **Sets container name** — Uses the optional 4th argument or defaults to `FOLDER_NAME`.
3. **Builds the image** — Creates a `fingerprints_benchmark` Docker image from the local `Dockerfile`.
4. **Runs the container** — Executes with the following configuration:
   - **Volume mount:** Binds `$HOME/np_fingerprint_benchmark/<FOLDER_NAME>/` to `/workspace/<FOLDER_NAME>/`
   - **GPU access:** Grants access to the specified NVIDIA GPU
   - **Detached mode (`-d`):** Runs in the background
   - **Working directory:** Changes to `/workspace/<FOLDER_NAME>/` before executing the script
   - **Output redirection:** Captures all output to `output.txt` in the mounted folder

---

### Troubleshooting
 | Issue                          | Solution                                                                 |
 |--------------------------------|--------------------------------------------------------------------------|
 | **"Usage" error**              | Ensure exactly 3 or 4 arguments are provided and the script is executable: `chmod +x script.sh` |
 | **GPU not found**              | Verify GPU availability: `podman run --rm --device nvidia.com/gpu=0 nvidia/cuda nvidia-smi` |
 | **Container already exists**   | Remove the existing container: `podman rm <CONTAINER_NAME>`             |
 | **Permission denied on volume mount** | Ensure the folder exists and you have read/write permissions: `mkdir -p ~/np_fingerprint_benchmark/<FOLDER_NAME>` |

---

### Notes
- Containers run in **detached mode** — Use `podman logs <CONTAINER_NAME>` to view real-time output.
- The `Dockerfile` must be in the current directory.
- The volume path is hardcoded to `/home/jcapela/np_fingerprint_benchmark/` — Modify the script if a different path is needed.

---

## Similarity Analysis

### Dataset Sample
A sample dataset of natural products is available at [similarity_analysis/30k_sample.csv](similarity_analysis/30k_sample.csv).

### Compute Similarities
Run [compute_similarities.py](similarity_analysis/compute_similarities.py) to calculate cosine and Tanimoto similarities and generate violin plots.

### Biosynthetic Distance
1. Generate reaction chains: [compute_biosynthetic_reactions.py](similarity_analysis/biosynthetic_distance/plantcyc/compute_biosynthetic_reactions.py)
2. Compute biosynthetic distances: [biosynthetic_distance.ipynb](similarity_analysis/biosynthetic_distance/plantcyc/biosynthetic_distance.ipynb)