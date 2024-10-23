FROM python:3.12.7-slim-bullseye

RUN apt update -y &&  \
    apt upgrade -y && \
    apt install make curl -y && \
    apt clean && \
    pip install selenium
