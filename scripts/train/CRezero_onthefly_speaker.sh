#!/usr/bin/env bash

speech_dir=/home/dataset/sakaya/rezero_speech/train-clean-100
noise_dir=/home/dataset/sakaya/rezero_noise
val_dir=/mlnas/sakaya/rezero_simulation/conical/first_speaker/S2N1R1/val
region_type=conical
project_name=CRezero
run_name=onthefly_speaker_RMSNorm

env CUDA_VISIBLE_DEVICES=$1 taskset -c $2-$3  uv run src/ADCRezero/train/train_onthefly.py --speech_dir $speech_dir --noise_dir $noise_dir --project_name $project_name --val_dir $val_dir --region_type $region_type --first_positioning "speaker"  --run_name $run_name \