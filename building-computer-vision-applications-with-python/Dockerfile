FROM python:3.12.7-slim-bullseye

RUN apt update -y &&  \
    apt upgrade -y && \
    apt install make -y && \
    apt install ffmpeg libsm6 libxext6 -y && \
    apt clean && \
    pip install matplotlib numpy opencv-python


# pandas scikit-learn seaborn tensorflow
