model_path=/mlnas/sakaya/checkpoints/model_iter_210000.pth
test_dir=/home/dataset/sakaya/rezero_simulation/spherical/first_mic/S2N1R1_limit_mic_z_spk_elve/test
region_type=spherical

env CUDA_VISIBLE_DEVICES=$1 taskset -c $2-$3 uv run src/ADCRezero/test/test.py --DRezero_path $model_path --test_dir $test_dir --region_type $region_type --inference --evaluation