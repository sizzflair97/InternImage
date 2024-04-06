#!/bin/bash
CUDA_VISIBLE_DEVICES=0

python image_demo.py \
  data/test_image\
  configs/dacon/upernet_internimage_xl_512x1024_80k_mapillary.py  \
  ${@:1} \
  --opacity 0.5