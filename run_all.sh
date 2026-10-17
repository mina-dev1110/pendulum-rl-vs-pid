#!/usr/bin/env bash
set -e
python src/train_sac.py --seeds 0 1 2 --steps 20000
python src/evaluate.py --seeds 0 1 2 --episodes 100
