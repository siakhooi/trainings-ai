# https://www.linkedin.com/learning/ai-workshop-hands-on-with-gans-using-dense-neural-networks

IMAGE_NAME=siakhooi/ai-workshop-hands-on-with-gans-using-dense-neural-networks:202412
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
	rm -rf *.png output
run-1:
	python ch1.py
run-2-1:
	python ch2-1-discriminator.py
run-2-2:
	python ch2-2-generator.py
run-3-1:
	mkdir -p output
	python ch3-1.py
run-3-2:
	mkdir -p output
	python ch3-2.py
