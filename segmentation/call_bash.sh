#!/bin/bash

srun --cpus-per-task 14 -p gpu2 --gres=gpu:a10:4 --pty bash