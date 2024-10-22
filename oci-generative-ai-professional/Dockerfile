FROM python:3.12.7-slim-bullseye

RUN apt update -y &&  \
    apt upgrade -y && \
    apt install make -y && \
    apt clean && \
    pip install jupyterlab && \
    pip install rank-bm25 scikit-learn && \
    pip install cohere weaviate-client && \
    pip install langchain langchain_community oci  && \
    pip install ads chromadb streamlit && \
    pip install langsmith

# wise
# pydantic
# pypdf
# oracle-ads
# faiss-cpu
# python-multipart
