# trainings-ai-base:latest
# FROM trainings-ai-base:latest

FROM python:3.12.7-slim-bullseye

RUN apt update -y &&  \
    apt upgrade -y && \
    apt install make -y && \
    apt clean && \

## python-object-oriented-programming
   # apt install make -y && \

# python-essential-training
   #  apt install make -y && \
    pip install jupyterlab multiprocess

# level-up-python
#    apt install make -y && \
    apt install unzip -y && \

# python-automation-and-testing
   # apt install make -y && \
    apt install curl -y && \
    pip install selenium

# training-neural-networks-in-python
   # apt install make -y && \
    pip install numpy

## artificial-intelligence-foundations-neural-networks
   # apt install make -y && \
    pip install matplotlib numpy pandas scikit-learn seaborn tensorflow

## recurrent-neural-networks
   # apt install make -y && \
   pip install keras matplotlib numpy pandas sklearn tensorflow

# machine-learning-foundations-linear-algebra
   # apt install make -y && \
    pip install jupyterlab numpy

## pandas-essential-training
   # apt install make -y && \
   RUN apt install wget -y
   RUN pip install jupyterlab pandas




########################################################################################


# data-analysis-with-python
    apt install make -y && \
    pip install jupyterlab matplotlib numpy pandas scikit-learn scipy seaborn skillsnetwork statsmodels tqdm

# python-for-data-science
    apt install make -y && \
    pip install ibm_watson jupyterlab matplotlib pandas

# data-visualization-with-python
    apt install make curl -y && \
    pip install folium jupyterlab matplotlib numpy openpyxl pandas pywaffle seaborn wordcloud

# building-computer-vision-applications-with-python
    apt install make -y && \
    apt install ffmpeg libsm6 libxext6 -y && \
    pip install matplotlib numpy opencv-python


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

# oci-generative-ai-professional
    apt install make -y && \
    pip install jupyterlab && \
    pip install rank-bm25 scikit-learn && \
    pip install cohere weaviate-client && \
    pip install langchain langchain_community oci  && \
    pip install ads chromadb streamlit && \
    pip install langsmith

# deep-learning-foundations-natural-language-processing-with-tensorflow
    apt install make wget -y && \
    pip install jupyterlab numpy pandas tensorflow tensorflow_datasets && \
    pip install google matplotlib


# machine-learning-with-python
    apt install make curl unzip -y && \
    pip install jupyterlab && \
    pip install matplotlib numpy scipy && \
    pip install pandas scikit-learn && \
    pip install basemap

# unit-testing-in-python
  pip -r requirements.txt


# numpy              2.1.3        43M

# pandas           2.2.3         76M
# pytz             2024.2       2.8M
# tzdata           2024.2       2.8M

# matplotlib  8.3M
# fonttools   4.9M
# pillow      4.4M
# kiwisolver  1.5 MB


# selenium           4.25.0       29M

# jupyterlab                 4.3.1           21M
# babel                      2.16.0          31M
# debugpy                    1.8.9           26M
# jedi                       0.19.2          12M
# setuptools                 75.6.0          11M

