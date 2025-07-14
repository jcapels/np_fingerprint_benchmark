FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3

WORKDIR /workspace

# depending on what you want to test
COPY ./requirements_automl.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

RUN pip install --no-deps git+https://github.com/BioSystemsUM/DeepMol.git@masked_learning

# COPY ./scripts/data/np_classifier/ /workspace

# RUN pip install . --no-deps
WORKDIR /workspace/scripts

CMD bash