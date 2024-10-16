# https://cognitiveclass.ai/courses/data-visualization-python

#SHELL := /bin/bash

IMAGE_NAME=siakhooi/data-visualization-with-python:202312
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
	mkdir -p resources
	curl -L -o resources/Canada.xlsx \
		https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx
	curl -L -o resources/alice_novel.txt \
		https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt
	curl -L -o resources/alice_mask.png \
		'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png'
	curl -L -o resources/Police_Department_Incidents_-_Previous_Year__2016_.csv \
		'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Police_Department_Incidents_-_Previous_Year__2016_.csv'
	curl -L -o 'resources/world_countries.json' \
		'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json'
clean:
	rm -f *.png *.html
prepare:
	python lab1-prepare.py
	python lab2-prepare.py

# Syllabus
# Module 1 - Introduction to Visualization Tools
# Introduction to Data Visualization
# Introduction to Matplotlib
# Basic Plotting with Matplotlib
# Dataset on Immigration to Canada
# Line Plots

# Module 2 - Basic Visualization Tools
# Area Plots
# Histograms
# Bar Charts

# Module 3 - Specialized Visualization Tools
# Pie Charts
# Box Plots
# Scatter Plots

# Module 4 - Extra Visualization Tools
# Waffle Charts
# Word Clouds
# Seaborn and Regression Plots

# Module 5 - Creating Maps and Visualizing Geospatial Data
# Introduction to Folium
# Maps with Markers
# Choropleth Maps
