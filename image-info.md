## trainings-ai-base:latest

FROM python:3.12.7-slim-bullseye

RUN apt update -y &&  \
    apt upgrade -y && \
    apt install make -y && \
    apt clean && \

## python-object-oriented-programming
   # apt install make -y && \

########################################################################################

# python-automation-and-testing
    apt install make curl -y && \
    pip install selenium

# python-essential-training
    pip install jupyterlab multiprocess


# level-up-python
    apt install make unzip -y && \

# artificial-intelligence-foundations-neural-networks

    apt install make -y && \
    pip install matplotlib numpy pandas scikit-learn seaborn tensorflow

# building-computer-vision-applications-with-python
    apt install make -y && \
    apt install ffmpeg libsm6 libxext6 -y && \
    pip install matplotlib numpy opencv-python

# data-analysis-with-python
    apt install make -y && \
    pip install jupyterlab matplotlib numpy pandas scikit-learn scipy seaborn skillsnetwork statsmodels tqdm

# data-visualization-with-python
    apt install make curl -y && \
    pip install folium jupyterlab matplotlib numpy openpyxl pandas pywaffle seaborn wordcloud

# deep-learning-foundations-natural-language-processing-with-tensorflow
    apt install make wget -y && \
    pip install jupyterlab numpy pandas tensorflow tensorflow_datasets && \
    pip install google matplotlib


# deep-learning-getting-started
    apt install make -y && \
    pip install jupyterlab matplotlib pandas scikit-learn tensorflow && \
    pip install nltk

# hands-on-pytorch-machine-learning
    apt install make -y && \
    pip install Jupyter matplotlib numpy pandas && \
    pip install torch torchaudio torchvision && \
    pip install IPython requests soundfile && \
    pip install Pillow

# introduction-to-prompt-engineering-for-generative-ai
    apt install make -y && \
    pip install openai

# machine-learning-foundations-linear-algebra
    apt install make -y && \
    pip install jupyterlab numpy

# machine-learning-with-python
    apt install make curl unzip -y && \
    pip install jupyterlab && \
    pip install matplotlib numpy scipy && \
    pip install pandas scikit-learn && \
    pip install basemap

# oci-generative-ai-professional
    apt install make -y && \
    pip install jupyterlab && \
    pip install rank-bm25 scikit-learn && \
    pip install cohere weaviate-client && \
    pip install langchain langchain_community oci  && \
    pip install ads chromadb streamlit && \
    pip install langsmith

# python-for-data-science
    apt install make -y && \
    pip install ibm_watson jupyterlab matplotlib pandas

# training-neural-networks-in-python
    apt install make -y && \
    pip install numpy

# unit-testing-in-python
  pip -r requirements.txt

# recurrent-neural-networks
    apt install make -y && \
    pip install keras matplotlib numpy pandas sklearn tensorflow
