# Generative AI Professional

IMAGE_NAME=siakhooi/oci-generative-ai-professional:202403
docker-build:
	docker build . -t ${IMAGE_NAME}
docker-push:
	docker push ${IMAGE_NAME}
docker-pull:
	docker pull ${IMAGE_NAME}
docker-run:
	docker run --rm -it --network host -w /working -v $$(pwd):/working  ${IMAGE_NAME} bash
docker-rmi:
	docker rmi ${IMAGE_NAME}

# https://dashboard.cohere.com/
# https://docs.cohere.com/reference/key