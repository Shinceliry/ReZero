#!/usr/bin/env bash

speech_dir=/home/dataset/sakaya/rezero_speech/train-clean-100
noise_dir=/home/dataset/sakaya/rezero_noise
val_dir=/mlnas/sakaya/rezero_simulation/spherical/first_mic/S2N1R1/val
region_type=spherical
project_name=DRezero
run_name=onthefly_RMSNorm
config=/home/sakaya/InvisibleMic/src/ADCRezero/config/D303535.yaml

env CUDA_VISIBLE_DEVICES=$1 taskset -c $2-$3  uv run src/ADCRezero/train/train_onthefly.py --speech_dir $speech_dir --noise_dir $noise_dir --project_name $project_name --val_dir $val_dir --region_type $region_type --run_name $run_name \
                                                                    --config $config