# https://www.linkedin.com/learning/hands-on-pytorch-machine-learning

IMAGE_NAME=siakhooi/hands-on-pytorch-machine-learning:202403
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

clean:
	rm -rf 0*.png

run-01-04-dataset:
	python 01-04-1-dataset-exploration-fashion-mnist.py
run-01-04-data:
	python 01-04-2-data-exploration-tutors.py
run-02-01-pytorch-sensors:
	python 02-01-pytorch-sensors.py
run-02-02-pytorch-basic-operations:
	python 02-02-pytorch-basic-operations.py
run-02-03-pytorch-numpy-bridge:
	python 02-03-pytorch-numpy-bridge.py
run-02-04-autograd:
	python 02-04-autograd.py
run-02-05-advanced-autograd:
	python 02-05-advanced-autograd.py
run-03-02-torchvision:
	python 03-02-torchvision.py
run-04-02-torchaudio-speech-recognition:
	python 04-02-torchaudio-speech-recognition.py
run-05-02-torchtext-classification:
	python 05-02-torchtext-classification.py

