SHELL := /bin/bash

IMAGE_NAME=siakhooi/level-up-python:202410
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

run-01:
	python 01-find-prime-factors.py
	python 'level-up-python-3210418-main/src/01 Find Prime Factors/factor.py'
run-02:
	python 02-find-palindrome.py
	python 'level-up-python-3210418-main/src/02 Identify a Palindrome/palindrome.py'
run-03:
	python 03-sort-a-string.py
	python 'level-up-python-3210418-main/src/03 Sort a String/sort_words.py'
run-04:
	python 04-find-all-list-items.py
	python 'level-up-python-3210418-main/src/04 Find All List Items/index_all.py'
run-05:
	python 05-play-the-waiting-game.py
#	python 'level-up-python-3210418-main/src/05 Play the Waiting Game/waiting_game.py'
run-06:
	python 06-save-a-dictionary.py
	python 'level-up-python-3210418-main/src/06 Save a Dictionary/dictionary.py'
run-07:
	python 07-schedule-a-function.py
	python 'level-up-python-3210418-main/src/07 Schedule a Function/schedule_function.py'
run-08:
	python 08-send-an-email.py
	python 'level-up-python-3210418-main/src/08 Send an Email/send_email.py'
run-09:
	python 09-simulate-dice.py
	python 'level-up-python-3210418-main/src/09 Simulate Dice/roll_dice.py'
run-10:
	python 10-count-unique-words.py
	cd 'level-up-python-3210418-main/src/10 Count Unique Words/' && python 'count_words.py'
run-11:
	python 11-generate-a-password.py
	cd 'level-up-python-3210418-main/src/11 Generate a Password/' && python 'diceware.py'
run-12:
	python 12-merge-csv-files.py
	cd 'level-up-python-3210418-main/src/12 Merge CSV Files/' && python 'merge_csv.py'
run-13:
	python 13-solve-a-sudoku.py
	python 'level-up-python-3210418-main/src/13 Solve a Sudoku/sudoku.py'
run-14:
	python 14-build-a-zip-archive.py
	unzip -t my_stuff.zip
	cd 'level-up-python-3210418-main/src/14 Build a Zip Archive/' && python 'zip_all.py'
	unzip -t 'level-up-python-3210418-main/src/14 Build a Zip Archive/my_stuff.zip'

run-15:
	python 15-download-sequential-files.py
	python 'level-up-python-3210418-main/src/15 Download Sequential Files/download_files.py'
	diff images images1
