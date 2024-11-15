FROM python:3.12.7-slim-bullseye

RUN apt update -y &&  \
    apt upgrade -y && \
    apt install make wget -y && \
    apt clean && \
    pip install jupyterlab numpy pandas tensorflow tensorflow_datasets && \
    pip install google matplotlib

    # pip install matplotlib scikit-learn
    # pip install nltk
