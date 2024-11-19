FROM python:3.12.7-slim-bullseye

RUN apt update -y &&  \
    apt upgrade -y && \
    apt install make -y && \
    apt clean && \
    pip install Jupyter matplotlib numpy pandas && \
    pip install torch torchaudio torchvision && \
    pip install IPython requests soundfile && \
    pip install Pillow
