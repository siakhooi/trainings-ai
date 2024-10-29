# https://www.linkedin.com/learning/introduction-to-prompt-engineering-for-generative-ai-2023

IMAGE_NAME=siakhooi/introduction-to-prompt-engineering-for-generative-ai:202309
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
