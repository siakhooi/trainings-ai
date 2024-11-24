#!/bin/bash

readonly base_tag=trainings-ai-base:latest
readonly base_file=Dockerfile.base

tempDir=$(mktemp -d --tmpdir "$(basename "$0")-XXXXXXXXXX")

base_Dockerfile=$(realpath "$base_file")
docker build -t $base_tag -f "$base_Dockerfile" "$tempDir"
