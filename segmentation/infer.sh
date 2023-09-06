#!/bin/bash
CUDA_VISIBLE_DEVICES=0

python image_demo.py \
  data/test_image \
  configs/dacon/upernet_internimage_xl_512x1024_160k_dacon.py  \
  work_dirs/upernet_internimage_xl_512x1024_160k_dacon/latest.pth \
  --opacity 0.5