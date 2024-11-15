# https://www.linkedin.com/learning/deep-learning-getting-started

IMAGE_NAME=siakhooi/deep-learning-getting-started:202403
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

run4:
	python code-04-iris-classification.py

run5:
	python code-05-spam-classification.py

run6:
	python code-06-incident-root-cause-analysis.py
