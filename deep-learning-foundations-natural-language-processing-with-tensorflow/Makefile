# https://www.linkedin.com/learning/deep-learning-foundations-natural-language-processing-with-tensorflow
# https://projector.tensorflow.org
# https://raw.githubusercontent.com/dswh/cml_example2/refs/heads/master/data/headlines.json
# https://github.com/dswh

IMAGE_NAME=siakhooi/deep-learning-foundations-natural-language-processing-with-tensorflow:202403
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

download-files:
#	wget --no-check-certificate https://storage.googleapis.com/wdd-2-node.appspot.com/x1.json -o x1.json
	wget --no-check-certificate https://raw.githubusercontent.com/dswh/cml_example2/refs/heads/master/data/headlines.json -o x1.json
	wget --no-check-certificate https://raw.githubusercontent.com/dswh/lil_nlp_with_tensorflow/main/sonnets.txt -O /tmp/sonnet.txt

run-01-02:
	python 01-02-word-encodings.py
run-01-03:
	python 01-03-creating-sequences-of-tokens.py
run-01-04:
	python 01-04-padding-the-sequences.py
run-01-06:
	python 01-06-sentiment-analysis.py
run-02-02:
	python 02-02-word-embeddings-for-sentiment-analysis.py
run-02-03:
	python 02-03-projecting-vectors-using-tensorflow.py
run-02-04:
	python 02-04-classifying-text.py
run-02-06:
	python 02-06-text-classification.py
run-03-02:
	python 03-02-implements-long-short-term-memory.py
run-03-03:
	python 03-03-improving-text-classifier-with-cnn.py
run-03-05:
	python 03-05-yelp-review-classifier.py
run-04-02:
	python 04-02-text-generation.py
run-04-04:
	python 04-04-poetry-generation.py
