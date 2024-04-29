#!/bin/bash

# see https://github.com/max-mapper/gifify-docker

# Exit on error
set -e


# Define the input and output files
path="data/Karioitahi_09Feb2022/"
input_file="132MSDCF-28mm-f4.mp4"
output_file="132MSDCF-28mm-f4.gif"

cd "$path"
docker run -it --rm -v $(pwd):/data maxogden/gifify "$input_file" -o "$output_file" --from 0 --to 5 --resize 1000:-1