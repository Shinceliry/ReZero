#!/usr/bin/env bash

speech_dir=/mlnas/sakaya/Dataset/Raw/LibriSpeech/train-clean-100
noise_dir=/mlnas/sakaya_a6000/rezero_noise
val_dir=/mlnas/sakaya/rezero_simulation/conical/first_speaker/S2N1R1/val
region_type=conical
decision_query_region=angle_limit
project_name=CRezero
run_name=onthefly_angle_limit

env CUDA_VISIBLE_DEVICES=$1 taskset -c $2-$3  uv run src/ADCRezero/train/train_onthefly.py --speech_dir $speech_dir --noise_dir $noise_dir --project_name $project_name --val_dir $val_dir --region_type $region_type --decision_query_region $decision_query_region  --run_name $run_name