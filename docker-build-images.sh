#!/bin/bash

readonly base_tag=trainings-ai-base:latest
readonly base_file=Dockerfile.base

tempDir=$(mktemp -d --tmpdir "$(basename "$0")-XXXXXXXXXX")

build_base() {
  base_Dockerfile=$(realpath "$base_file")
  docker build -t $base_tag -f "$base_Dockerfile" "$tempDir"
}
build_image() {
  local training=$1

  (
    cd "$training" || {
      echo "Directory not found: $training"
      exit
    }
    make docker-build
    make docker-push
    make docker-rmi
  )
}

# build_base
# build_image python-automation-and-testing
# build_image python-object-oriented-programming
# build_image python-essential-training
# build_image level-up-python
# build_image machine-learning-foundations-linear-algebra
# build_image training-neural-networks-in-python
# build_image artificial-intelligence-foundations-neural-networks
# build_image recurrent-neural-networks

# to repush

# build_image pandas-essential-training
