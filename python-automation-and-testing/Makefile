# https://www.linkedin.com/learning/python-automation-and-testing
# https://selenium-python.readthedocs.io/installation.html

IMAGE_NAME=siakhooi/python-automation-and-testing:202410
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
	curl -LO https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
	tar xf geckodriver-v0.35.0-linux64.tar.gz

web:
	(cd localhost && python -m http.server)
