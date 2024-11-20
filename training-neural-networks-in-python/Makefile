# https://www.linkedin.com/learning/training-neural-networks-in-python-17058600
# https://github.com/LinkedInLearning/training-neural-networks-in-python-3215347


IMAGE_NAME=siakhooi/training-neural-networks-in-python:202411
docker-build:
	docker build . -t ${IMAGE_NAME}
docker-run:
	docker run --rm -it --network host -w /working -v $$(pwd):/working  ${IMAGE_NAME} bash
docker-push:
	docker push ${IMAGE_NAME}
docker-pull:
	docker pull ${IMAGE_NAME}
docker-rmi:
	docker rmi ${IMAGE_NAME}

run-02:
	python3 02-MLP-Perceptron.py
run-03:
	python3 03-MLP.py
run-04:
	python3 04-MLP-backpropagation.py
run-05:
	python3 05-SDR.py
run-06:
	python3 06-SDRNN_7to1.py
run-07:
	python3 07-SDRNN_7to7.py
run-08:
	python3 08-SDRNN_7to10.py
