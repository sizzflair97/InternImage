#!/bin/bash

srun --cpus-per-task 56 -p gpu2 --gres=gpu:a10:4 --pty bash