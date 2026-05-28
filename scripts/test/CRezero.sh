model_path=/home/sakaya/InvisibleMic/models/CRezero/onthefly_speaker_2026-01-24-09:40:39/model_iter_240000.pth
test_dir=/home/dataset/sakaya/rezero_simulation/conical/first_speaker/S2N1R1_limit_mic_z_spk_elve/test
region_type=conical

env CUDA_VISIBLE_DEVICES=$1 taskset -c $2-$3 uv run src/ADCRezero/test/test.py --CRezero_path $model_path --test_dir $test_dir --region_type $region_type --evaluation --inference 