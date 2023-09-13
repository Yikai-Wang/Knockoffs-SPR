import os

options = [
    ('sym', 0.2),
    ('sym', 0.4),
    ('sym', 0.6),
    ('sym', 0.8),
    ('asy', 0.2),
    ('asy', 0.3),
    ('asy', 0.4)
]

for noise_type, noise_rate in options:
    string = "python eval.py --gpus 0 --dataset CIFAR100 --config_file configs/cifar100.yaml --resume ckpt/CIFAR100/{}-{}.pth.tar".format(noise_type, noise_rate)
    os.system(string)
