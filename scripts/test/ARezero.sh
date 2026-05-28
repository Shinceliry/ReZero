model_path=/home/sakaya/InvisibleMic/models/ARezero/onthefly_RMSNorm_2025-12-15-20:50:28/model_iter_240000.pth
test_dir=/mlnas/sakaya/rezero_simulation/angular/first_mic/S2N1R1/test
region_type=angular

env CUDA_VISIBLE_DEVICES=$1 taskset -c $2-$3 uv run src/ADCRezero/test/test.py --ARezero_path $model_path --test_dir $test_dir --region_type $region_type --evaluation --inference 