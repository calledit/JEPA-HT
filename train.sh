#!/bin/bash
python train.py "$@"
code=$?
if [ $code -ne 0 ] && [ $code -ne 130 ]; then
    curl -s -d "train.py crashed! (exit $code)" ntfy.sh/3090_training_me_124
fi
