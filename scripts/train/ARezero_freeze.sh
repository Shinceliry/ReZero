#!/usr/bin/env bash

train_dir=/mlnas/sakaya/rezero_simulation/angular/first_mic/2N1R1/train
val_dir=/mlnas/sakaya/rezero_simulation/angular/first_mic/S2N1R1/val
region_type=angular
project_name=ARezero
training_seed=$1
run_name="freeze_first_mic_rms10_seed${training_seed}"

env CUDA_VISIBLE_DEVICES=$2 taskset -c $3-$4 uv run src/ADCRezero/train/train_freeze.py --train_dir $train_dir --project_name $project_name --val_dir $val_dir --region_type $region_type --run_name $run_name --training_seed $training_seed