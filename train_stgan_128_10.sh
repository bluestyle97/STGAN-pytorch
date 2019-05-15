#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 screen python main.py --config ./configs/train_stgan_128_10.yaml
