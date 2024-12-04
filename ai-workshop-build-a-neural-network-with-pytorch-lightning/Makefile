# https://www.linkedin.com/learning/ai-workshop-build-a-neural-network-with-pytorch-lightning

IMAGE_NAME=siakhooi/ai-workshop-build-a-neural-network-with-pytorch-lightning:202412
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
run-01:
	python c01-using-pytorch.py
run-02:
	python c02-regression-model-using-pytorch-lightning.py
run-03:
	python c03-classification-model-using-pytorch-lightning.py
