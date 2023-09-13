import os

string = "python eval.py --gpus 0 --dataset clothing1M --config_file configs/clothing1M.yaml --resume ckpt/clothing1m.pth.tar"
os.system(string)
