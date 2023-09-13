import os

string = "python eval.py --gpus 0 --dataset WebVision --config_file configs/webvision.yaml --resume ckpt/webvision.pth.tar"
os.system(string)
