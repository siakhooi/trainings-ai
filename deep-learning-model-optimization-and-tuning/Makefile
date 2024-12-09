# https://www.linkedin.com/learning/deep-learning-model-optimization-and-tuning

IMAGE_NAME=siakhooi/deep-learning-model-optimization-and-tuning:202412
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
run-2-2:
	python c2-2-tuning-batch-size-and-epoch.py
run-2-3:
	python c2-3-tuning-layers.py
run-2-4:
	python c2-4-tuning-nodes.py
run-2-5:
	python c2-5-tuning-activation-functions.py
run-2-6:
	python c2-6-weights-initialization.py
run-3-2:
	python c3-2-batch-normalization.py
run-3-4:
	python c3-4-optimizers.py
run-3-6:
	python c3-6-learning-rates.py
run-4-3:
	python c4-3-regularization.py
run-4-5:
	python c4-5-dropout.py
run-5-2:
	python c5-2-acquire-and-process-data.py
