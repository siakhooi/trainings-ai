# https://www.linkedin.com/learning/unit-testing-in-python
# https://docs.pytest.org/en/latest
# https://github.com/coding-geographies/dockerized-pytest-course

IMAGE_NAME=siakhooi/unit-testing-in-python:202411

docker-build:
	docker compose build
docker-run:
	docker compose run test sh

docker-retag:
	docker tag dockerized-pytest-course-master-test:latest ${IMAGE_NAME}
docker-push:
	docker push ${IMAGE_NAME}
docker-pull:
	docker pull ${IMAGE_NAME}
docker-rmi:
	docker rmi dockerized-pytest-course-master-test:latest
	docker rmi ${IMAGE_NAME}

pytest1:
	pytest
	pytest -k test_fitness_log
	pytest -h
	pytest --cov

chapter2-video2:
	pytest -k map
chapter2-video3:
	pytest -k except
chapter2-video4:
	pytest -k test_chp2_challenge
chapter2-video6:
	pytest -k test_happy_path_start
chapter2-video7:
	pytest -k test_sad_path_start

chapter3-video1:
	pytest -k test_fixtures_start
# chapter3-video2:
# 	pytest -k test_factory_start
chapter3-video3:
	pytest -k test_parametrize_start
chapter3-video4:
	pytest -k test_param_challenge

chapter4-video1:
	pytest -k test_conftest_start
chapter4-video5:
	pytest -k conftest_start

challenges:
	pytest --cov scripts
