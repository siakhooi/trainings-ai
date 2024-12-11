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

# machine-learning-foundations-linear-algebra
   # apt install make -y && \
    pip install jupyterlab numpy

## pandas-essential-training
   # apt install make -y && \
   RUN apt install wget -y
   RUN pip install jupyterlab pandas

## recurrent-neural-networks
   # apt install make -y && \
   pip install keras matplotlib numpy pandas sklearn tensorflow

# deep-learning-model-optimization-and-tuning
RUN pip install matplotlib numpy pandas scikit-learn tensorflow

# introduction-to-generative-adversarial-networks-gans
RUN pip install torch torchvision

# ai-workshop-hands-on-with-gans-using-dense-neural-networks
RUN pip install torch torchvision  matplotlib numpy tqdm


########################################################################################
trainings-ai-data:latest
# FROM trainings-ai-data:latest
FROM trainings-ai-base:latest
RUN pip install matplotlib numpy pandas seaborn

# data-analysis-with-python
#    apt install make -y && \
# RUN pip install matplotlib numpy pandas seaborn
    pip install jupyterlab scikit-learn scipy skillsnetwork statsmodels tqdm

# ai-workshop-build-a-neural-network-with-pytorch-lightning
# FROM trainings-ai-data:latest
RUN pip install scikit-learn
RUN pip install torch torchmetrics lightning pytorch_lightning

# deep-learning-model-optimization-and-tuning
RUN pip install scikit-learn tensorflow


########################################################################################



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

# pandas
# pandas           2.2.3         76M
# pytz             2024.2       2.8M
# tzdata           2024.2       2.8M

# matplotlib (19MB)
# matplotlib  8.3M
# fonttools   4.9M
# pillow      4.4M
# kiwisolver  1.5MB

# seaborn
# matplotlib (8.3 MB)
# fonttoolsl (4.9 MB)
# kiwisolver (1.5 MB)
# pillow     (4.4 MB)


# selenium           4.25.0       29M

# jupyterlab (101MB)
# jupyterlab                 4.3.1           21M
# babel                      2.16.0          31M
# debugpy                    1.8.9           26M
# jedi                       0.19.2          12M
# setuptools                 75.6.0          11M


# scikit-learn
# scikit_learn 12.9 MB
# scipy        40.8 MB

# tensorflow              (688.4 MB)
# tensorflow              (615.5 MB)
# grpcio                    (5.9 MB)
# h5py                      (5.4 MB)
# keras                     (1.2 MB)
# libclang                 (24.5 MB)
# ml_dtypes                 (2.2 MB)
# numpy                    (19.2 MB)
# tensorboard               (5.5 MB)
# setuptools                (1.2 MB)
# tensorboard_data_server   (6.6 MB)
# pygments                  (1.2 MB)

# torch                 (3004.7 MB)
# torch                  (906.4 MB)
# nvidia_cublas_cu12     (363.4 MB)
# nvidia_cuda_cupti_cu12  (13.8 MB)
# nvidia_cuda_nvrtc_cu12  (24.6 MB)
# nvidia_cudnn_cu12      (664.8 MB)
# nvidia_cufft_cu12      (211.5 MB)
# nvidia_curand_cu12      (56.3 MB)
# nvidia_cusolver_cu12   (127.9 MB)
# nvidia_cusparse_cu12   (207.5 MB)
# nvidia_nccl_cu12       (188.7 MB)
# nvidia_nvjitlink_cu12   (21.1 MB)
# sympy                    (6.2 MB)
# triton                 (209.6 MB)
# networkx                 (1.7 MB)
# setuptools               (1.2 MB)

# torchmetrics          (3004.7 MB)
# torch                  (906.4 MB)
# nvidia_cublas_cu12     (363.4 MB)
# nvidia_cuda_cupti_cu12  (13.8 MB)
# nvidia_cuda_nvrtc_cu12  (24.6 MB)
# nvidia_cudnn_cu12      (664.8 MB)
# nvidia_cufft_cu12      (211.5 MB)
# nvidia_curand_cu12      (56.3 MB)
# nvidia_cusolver_cu12   (127.9 MB)
# nvidia_cusparse_cu12   (207.5 MB)
# nvidia_nccl_cu12       (188.7 MB)
# nvidia_nvjitlink_cu12   (21.1 MB)
# sympy                    (6.2 MB)
# triton                 (209.6 MB)
# networkx                 (1.7 MB)
# setuptools               (1.2 MB)

# lightning
# torch                  (906.4 MB)
# nvidia_cublas_cu12     (363.4 MB)
# nvidia_cuda_cupti_cu12  (13.8 MB)
# nvidia_cuda_nvrtc_cu12  (24.6 MB)
# nvidia_cudnn_cu12      (664.8 MB)
# nvidia_cufft_cu12      (211.5 MB)
# nvidia_curand_cu12      (56.3 MB)
# nvidia_cusolver_cu12   (127.9 MB)
# nvidia_cusparse_cu12   (207.5 MB)
# nvidia_nccl_cu12       (188.7 MB)
# nvidia_nvjitlink_cu12   (21.1 MB)
# sympy                    (6.2 MB)
# triton                 (209.6 MB)
# aiohttp                  (1.7 MB)
# networkx                 (1.7 MB)
# setuptools               (1.2 MB)

# pytorch_lightning

# torch                  (906.4 MB)
# nvidia_cublas_cu12     (363.4 MB)
# nvidia_cuda_cupti_cu12  (13.8 MB)
# nvidia_cuda_nvrtc_cu12  (24.6 MB)
# nvidia_cudnn_cu12      (664.8 MB)
# nvidia_cufft_cu12      (211.5 MB)
# nvidia_curand_cu12      (56.3 MB)
# nvidia_cusolver_cu12   (127.9 MB)
# nvidia_cusparse_cu12   (207.5 MB)
# nvidia_nccl_cu12       (188.7 MB)
# nvidia_nvjitlink_cu12   (21.1 MB)
# sympy                    (6.2 MB)
# triton                 (209.6 MB)
# aiohttp                  (1.7 MB)
# networkx                 (1.7 MB)
# setuptools               (1.2 MB)
