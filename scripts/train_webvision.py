import os

string = "python train.py --gpus 1 --dataset WebVision --config_file configs/webvision.yaml --save_dir exps/WebVision/webvision --ema"

os.system(string)
