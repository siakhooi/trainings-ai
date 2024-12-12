# https://www.linkedin.com/learning/ai-workshop-hands-on-with-gans-with-deep-convolutional-networks

IMAGE_NAME=siakhooi/ai-workshop-hands-on-with-gans-with-deep-convolutional-networks:202412
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
	rm -rf ch*.jpg
clean-all: clean
	rm -rf anime_classification
unzip:
	mkdir anime_classification
	cd anime_classification && unzip ../'Ex_Files_AI_Workshop_Hands_on_GANs/Exercise Files/datasets/bad_fake_images.zip'
	cd anime_classification && unzip ../'Ex_Files_AI_Workshop_Hands_on_GANs/Exercise Files/datasets/good_fake_images.zip'
	cd anime_classification && unzip ../'Ex_Files_AI_Workshop_Hands_on_GANs/Exercise Files/datasets/real_images.zip'
	cd anime_classification && unzip ../'Ex_Files_AI_Workshop_Hands_on_GANs/Exercise Files/datasets/anime_images.zip'
	rm -rf anime_classification/__MACOSX

run-1-4:
	python ch1-4.py
run-1-5:
	python ch1-5.py
run-1-6:
	python ch1-6.py
run-2-3:
	python ch2-3.py
run-2-4:
	python ch2-4.py
run-2-5:
	python ch2-5-training-discriminator-on-bad-fakes.py
run-2-6:
	python ch2-6-training-discriminator-on-good-fakes.py
run-3-3:
	python ch3-3.py
run-3-4:
	python ch3-4.py
run-3-5:
	python ch3-5.py
run-3-6:
	python ch3-6.py
