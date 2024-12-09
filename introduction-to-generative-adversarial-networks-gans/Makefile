# https://www.linkedin.com/learning/introduction-to-generative-adversarial-networks-gans
# https://github.com/LinkedInLearning/generative-ai-foundations-generative-adversarial-networks-gans-4380315


IMAGE_NAME=siakhooi/introduction-to-generative-adversarial-networks-gans:202412
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
	rm -f *.png
run:
	python gan.py
