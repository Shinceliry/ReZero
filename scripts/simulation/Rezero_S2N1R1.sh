# usr/bin/env bash
region_type=$1
first_positioning=$2
gpu_id=$3
start_cpu_id=$4
end_cpu_id=$5

train_speech_dir=/home/dataset/sakaya/rezero_speech/train-clean-100
val_speech_dir=/home/dataset/sakaya/rezero_speech/train-clean-360
test_speech_dir=/home/dataset/sakaya/rezero_speech/train-clean-360
noise_dir=/home/dataset/sakaya/rezero_noise

output_base_dir=/home/dataset/sakaya/rezero_simulation/$region_type/first_$first_positioning/S2N1R1_limit_mic_z_spk_elve
train_output_dir=$output_base_dir/train
val_output_dir=$output_base_dir/val
test_output_dir=$output_base_dir/test

# config=src/ADCRezero/config/D303535.yaml

# # test
# env CUDA_VISIBLE_DEVICES=$gpu_id taskset -c $start_cpu_id-$end_cpu_id uv run src/ADCRezero/data/simulation_fram_rir.py \
#     --speech_dir $test_speech_dir \
#     --noise_dir $noise_dir \
#     --output_dir $test_output_dir \
#     --region_type $region_type \
#     --iterations 3000 \
#     --mode "test" \
#     --first_positioning $first_positioning \
#     --decision_query_region "angle_limit" \
#     --plot_3d \
#     --limit_mic_z \
#     --elevation_limit \
#     # --rms_norm \
#     # --config $config \

env CUDA_VISIBLE_DEVICES=$gpu_id taskset -c $start_cpu_id-$end_cpu_id uv run src/ADCRezero/test/generation_dataset_metadata.py --dataset_dir $test_output_dir --output_dir $output_base_dir

# # train
# env CUDA_VISIBLE_DEVICES=$gpu_id taskset -c $start_cpu_id-$end_cpu_id uv run src/ADCRezero/data/simulation_fram_rir.py \
#     --speech_dir $train_speech_dir \
#     --noise_dir $noise_dir \
#     --output_dir $train_output_dir \
#     --region_type $region_type \
#     --iterations 100 \
#     --mode "train" \
#     --first_positioning $first_positioning \
#     --decision_query_region "angle_limit" \
#     --rms_norm \
#     --limit_mic_z \
#     --elevation_limit

# env CUDA_VISIBLE_DEVICES=$gpu_id taskset -c $start_cpu_id-$end_cpu_id uv run src/ADCRezero/test/generation_dataset_metadata.py --dataset_dir $train_output_dir --output_dir $output_base_dir

# # val
# env CUDA_VISIBLE_DEVICES=$gpu_id taskset -c $start_cpu_id-$end_cpu_id uv run src/ADCRezero/data/simulation_fram_rir.py \
#     --speech_dir $val_speech_dir \
#     --noise_dir $noise_dir \
#     --output_dir $val_output_dir \
#     --region_type $region_type \
#     --iterations 1000 \
#     --mode "val" \
#     --first_positioning $first_positioning \
#     --decision_query_region "angle_limit" \
#     --rms_norm \
#     --limit_mic_z \
#     --elevation_limit \
#     # --config $config \

# env CUDA_VISIBLE_DEVICES=$gpu_id taskset -c $start_cpu_id-$end_cpu_id uv run src/ADCRezero/test/generation_dataset_metadata.py --dataset_dir $val_output_dir --output_dir $output_base_dir