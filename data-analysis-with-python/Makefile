# https://cognitiveclass.ai/courses/data-analysis-python

IMAGE_NAME=siakhooi/data-analysis-with-python:202311
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

download-files:
	mkdir -p resources
	curl -L -o resources/imports-85.data \
	https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
