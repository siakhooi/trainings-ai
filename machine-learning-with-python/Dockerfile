FROM python:3.12.7-slim-bullseye

RUN apt update -y &&  \
    apt upgrade -y && \
    apt install make curl unzip -y && \
    apt clean && \
    pip install jupyterlab && \
    pip install matplotlib numpy scipy && \
    pip install pandas scikit-learn && \
    pip install basemap
