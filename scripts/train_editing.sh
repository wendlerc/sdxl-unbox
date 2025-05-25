#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/train_sae.py SAE/config_editing_down.json &
CUDA_VISIBLE_DEVICES=0 python scripts/train_sae.py SAE/config_editing_up.json &
CUDA_VISIBLE_DEVICES=0 python scripts/train_sae.py SAE/config_editing_mid.json &
CUDA_VISIBLE_DEVICES=0 python scripts/train_sae.py SAE/config_editing_up0.json &
