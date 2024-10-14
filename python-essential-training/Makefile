# https://www.linkedin.com/learning/python-essential-training-18764650
# https://github.com/LinkedInLearning/python-ET-4314028

IMAGE_NAME=siakhooi/python-essential-training:202410
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

jupyterlab:
	jupyter lab --allow-root

c11:
	python 'Ex_Files_Python_EssT/Exercise Files/exercise_files/11_1_writefile.py' --output x -t x

