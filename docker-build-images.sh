#!/bin/bash

readonly base_tag=trainings-ai-base:latest
readonly base_file=Dockerfile.base
readonly data_tag=trainings-ai-data:latest
readonly data_file=Dockerfile.data

tempDir=$(mktemp -d --tmpdir "$(basename "$0")-XXXXXXXXXX")

build_base() {
  base_Dockerfile=$(realpath "$base_file")
  docker build -t $base_tag -f "$base_Dockerfile" "$tempDir"
}
build_data(){
  data_Dockerfile=$(realpath "$data_file")
  docker build -t $data_tag -f "$data_Dockerfile" "$tempDir"

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

################################################################################
# build_base

# build_image python-automation-and-testing
# build_image python-object-oriented-programming
# build_image python-essential-training
# build_image level-up-python
# build_image training-neural-networks-in-python
# build_image machine-learning-foundations-linear-algebra
# build_image pandas-essential-training
# build_image artificial-intelligence-foundations-neural-networks
# build_image recurrent-neural-networks
# build_image introduction-to-generative-adversarial-networks-gans
# build_image ai-workshop-hands-on-with-gans-using-dense-neural-networks

################################################################################
# build_data

# build_image data-analysis-with-python

# to repush

# build_image ai-workshop-build-a-neural-network-with-pytorch-lightning

# build_image deep-learning-model-optimization-and-tuning
