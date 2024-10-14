# https://cognitiveclass.ai/courses/python-for-data-science

SHELL := /bin/bash

IMAGE_NAME=siakhooi/python-for-data-science:202311
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

run-plot-sin:
	python ./numpy-plot-sin.py
