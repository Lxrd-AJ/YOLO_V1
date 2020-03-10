#!/bin/bash

./data/download_voc.sh

nohup python -u training.py > training.log &