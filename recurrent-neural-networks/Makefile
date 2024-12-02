# https://www.linkedin.com/learning/recurrent-neural-networks
# https://github.com/LinkedInLearning/recurrent-neural-networks-2478142

IMAGE_NAME=siakhooi/recurrent-neural-networks:202411
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
run-03:
	python c03-predict-stock-price.py
run-05:
	python c05-forecast-service-loads.py
prep-07:
	rm -f glove.6B.50d.txt
	unzip glove.6B.50d.txt.zip
run-07:
	python c07-spam-detector.py
