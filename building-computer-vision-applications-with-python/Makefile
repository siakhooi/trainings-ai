# https://www.linkedin.com/learning/building-computer-vision-applications-with-python

IMAGE_NAME=siakhooi/building-computer-vision-applications-with-python:202403
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
	rm -f 0*.jpg 0*.png horsie.jpg stitched.bmp
run-01:
	python 01-04-testing-your-environment.py
run-02-01:
	python 02-01-image-representation.py
run-02-02:
	python 02-02-color-encoding.py
run-02-03:
	python 02-03-image-file-management.py
run-02-04:
	python 02-04-resolution.py
run-02-05:
	python 02-05-rotations-and-flips.py
run-02-07:
	python 02-07-manipulate-some-pictures.py
run-03-01:
	python 03-01-average-grayscale.py
run-03-02:
	python 03-02-weighted-grayscale.py
run-03-03:
	python 03-03-converting-grayscale-to-black-and-white.py
run-03-04:
	python 03-04-adaptive-thresholding.py
run-03-06:
	python 03-06-removing-color.py
run-04-02:
	python 04-02-average-filters.py
run-04-03:
	python 04-03-median-filters.py
run-04-04:
	python 04-04-gaussian-filters.py
run-04-05:
	python 04-05-edge-detection-filters.py
run-04-07:
	python 04-07-create-your-own-convolution-filter.py
run-05-02:
	python 05-02-downscaling-pictures.py
run-05-04:
	python 05-04-upscaling-pictures.py
run-05-06:
	python 05-06-resize-some-pictures.py
run-06-02:
	python 06-02-stitching-images.py
run-06-05:
	python 06-05-stitch-two-pictures-together.py
run-07-02:
	python 07-02-erosion-and-dilation.py
run-07-03:
	python 07-03-open-and-close.py
run-07-05:
	python 07-05-help-a-robot.py
