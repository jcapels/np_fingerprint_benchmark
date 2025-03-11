FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3

WORKDIR /workspace

# depending on what you want to test
COPY ./requirements_automl.txt /workspace/requirements.txt
RUN pip install -r requirements.txt 

RUN pip install --no-deps deepmol[all]==1.1.11

RUN pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec

# COPY ./scripts/data/np_classifier/ /workspace

# RUN pip install . --no-deps
WORKDIR /workspace/scripts

CMD bash