# https://www.linkedin.com/learning/artificial-intelligence-foundations-neural-networks-22853427
# https://github.com/LinkedInLearning/artificial-intelligence-foundations-neural-networks-4381282

IMAGE_NAME=siakhooi/artificial-intelligence-foundations-neural-networks:202403
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
run-04:
	python 04-neural-network-using-keras.py

run-05:
	python 05-manually-tune-hyperparameters.py
