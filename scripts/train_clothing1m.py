import os

string = "python train.py --gpus 1 --dataset clothing1M --noise_type noisy --noise_rate 0.38 --pretrained True --config_file configs/clothing1M.yaml --save_dir exps/clothing1M/clothing1m --ema"

os.system(string)
