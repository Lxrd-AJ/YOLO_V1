#!/bin/bash

./data/download_voc.sh

# if training.py is not using a logger
# nohup python -u training.py > training.log &
nohup python -u training.py &