# https://www.linkedin.com/learning/pandas-essential-training-24082178
# https://github.com/LinkedInLearning/pandas-essential-training-new-dataset-dupe-4493047
# https://seaborn.pydata.org
# https://pandastutor.com

IMAGE_NAME=siakhooi/pandas-essential-training:202411
docker-build:
	docker build . -t ${IMAGE_NAME}
docker-run:
	docker run --rm -it --network host -w /working -v $$(pwd):/working  ${IMAGE_NAME} bash
docker-run-lab:
	docker run --rm -it --network host -w /working -v $$(pwd):/working  ${IMAGE_NAME} jupyter-lab --allow-root
docker-push:
	docker push ${IMAGE_NAME}
docker-pull:
	docker pull ${IMAGE_NAME}
docker-rmi:
	docker rmi ${IMAGE_NAME}

download-files:
	curl -LO https://github.com/jonfernandes/pandas_essential/raw/main/olympics_1896_2004.csv
	wget https://github.com/jonfernandes/pandas_essential/raw/main/olympics_2008.csv

clean:
	rm -f *.png
run-c2-1:
	python chapter2-1-overview.py
run-c2-2:
	python chapter2-2-series.py
run-c2-4:
	python chapter2-4-challange.py
run-c2-5:
	python chapter2-5-6-columns.py
run-c2-7:
	python chapter2-7-filters.py
run-c2-8:
	python chapter2-8-string.py
run-c2-9:
	python chapter2-9-sorting.py
run-c3-1:
	python chapter3-1-dtypes.py
run-c3-2:
	python chapter3-2-dtypes-memory.py
run-c3-3:
	python chapter3-3-read-and-dtypes.py
run-c3-5:
	python chapter3-5-index.py
run-c3-6:
	python chapter3-6-best-practices.py
run-c3-7:
	python chapter3-7-create-series-and-dataframes.py
run-c3-8:
	python chapter3-8-dates.py
run-c3-9:
	python chapter3-9-combine-dataframes.py
run-c3-10:
	python chapter3-10-combine-datasets.py
run-c3-11:
	python chapter3-11-missing-data-fillna.py
run-c3-12:
	python chapter3-12-missing-data-dropna.py
run-c3-13:
	python chapter3-13-duplicates.py
run-c3-14:
	python chapter3-14-validating-data.py
run-c3-15:
	python chapter3-15-further-data-validation.py
run-c4-1:
	python chapter4-1-plot.py
run-c4-2:
	python chapter4-2-seaborn-colormap.py
run-c4-3:
	python chapter4-3-groupby.py
run-c4-4:
	python chapter4-4-staking-unstaking.py
run-c4-6:
	python chapter4-6-challenges.py
