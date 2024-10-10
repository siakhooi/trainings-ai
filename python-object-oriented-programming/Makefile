# https://github.com/LinkedinLearning/python-object-oriented-programming-4413110
# https://docs.python.org/3/reference/datamodel.html
SHELL := /bin/bash

IMAGE_NAME=siakhooi/python-object-oriented-programming:202410
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

run-1-1:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 1/definition_start.py'
run-1-2:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 1/instance_start.py'
run-1-3:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 1/typecheck_start.py'
run-1-4:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 1/class_start.py'
run-1-5:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 1/challenge.py'
	python 'python-object-oriented-programming-4413110-main/Finished/Ch 1/challenge.py'

run-2-1:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 2/inheritance_start.py'
run-2-2:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 2/abstract_start.py'
run-2-3:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 2/multiple_start.py'
run-2-4:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 2/interface_start.py'
run-2-5:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 2/composition_start.py'
run-2-6:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 2/challenge.py'
	python 'python-object-oriented-programming-4413110-main/Finished/Ch 2/challenge.py'
	diff \
	<(python 'python-object-oriented-programming-4413110-main/Start/Ch 2/challenge.py') \
	<(python 'python-object-oriented-programming-4413110-main/Finished/Ch 2/challenge.py')

run-3-1:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 3/magicstr_start.py'
run-3-2:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 3/magiceq_start.py'
run-3-3:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 3/magicattr_start.py'
run-3-4:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 3/magiccall_start.py'
run-3-5:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 3/challenge.py'
	python 'python-object-oriented-programming-4413110-main/Finished/Ch 3/challenge.py'
	diff \
	<(python 'python-object-oriented-programming-4413110-main/Start/Ch 3/challenge.py') \
	<(python 'python-object-oriented-programming-4413110-main/Finished/Ch 3/challenge.py')

run-4-1:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 4/dataclass_start.py'
run-4-2:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 4/postinit_start.py'
run-4-3:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 4/datadefault_start.py'
run-4-4:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 4/immutable_start.py'
run-4-5:
	python 'python-object-oriented-programming-4413110-main/Start/Ch 4/challenge.py'
	python 'python-object-oriented-programming-4413110-main/Finished/Ch 4/challenge.py'
	diff \
	<(python 'python-object-oriented-programming-4413110-main/Start/Ch 4/challenge.py') \
	<(python 'python-object-oriented-programming-4413110-main/Finished/Ch 4/challenge.py')

