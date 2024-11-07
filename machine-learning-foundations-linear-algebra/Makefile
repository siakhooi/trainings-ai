# https://www.linkedin.com/learning/machine-learning-foundations-linear-algebra

SHELL := /bin/bash

IMAGE_NAME=siakhooi/machine-learning-foundations-linear-algebra:202402
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

run:
	python ch2-v2-vector-arithmetic.py
	python ch3-v1-dot-product-of-vectors.py
	python ch3-v2-scalar-and-vector-projection.py
	python ch5-v2-gaussion-elimination-and-finding-the-inverse-matrix.py
	python ch5-v3-inverse-and-determinant.py
