#!/usr/bin/env bash

train_dir=/mlnas/sakaya/rezero_simulation/conical/first_speaker/S2N1R1/train
val_dir=/mlnas/sakaya/rezero_simulation/conical/first_speaker/S2N1R1/val
region_type=conical
project_name=CRezero
run_name=freeze_first_speaker_lr1e-04
config=src/ADCRezero/config/lr1e-04.yaml

env CUDA_VISIBLE_DEVICES=$2 taskset -c $3-$4 uv run src/ADCRezero/train/train_freeze.py --train_dir $train_dir --project_name $project_name --val_dir $val_dir --region_type $region_type  --run_name $run_name --config $config