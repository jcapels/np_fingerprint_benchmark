#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <GPU_NUMBER> <FOLDER_NAME> <PYTHON_SCRIPT>"
    exit 1
fi

# Assign the provided arguments to variables
GPU_NUMBER="$1"
FOLDER_NAME="$2"
PYTHON_SCRIPT="$3"

# Build the container image
podman build . -t fingerprints_benchmark

# Run the container with the specified Python script as the command
podman run -v /home/jcapela/pathway_prediction/scripts/data/"$FOLDER_NAME"/:/workspace/"$FOLDER_NAME"/:z -d --device nvidia.com/gpu="$GPU_NUMBER" --security-opt=label=disable --name="$FOLDER_NAME" fingerprints_benchmark /bin/bash -c "cd /workspace/$FOLDER_NAME/ && python $PYTHON_SCRIPT > output.txt"
