import argparse
import yaml 

def config():
    parser = argparse.ArgumentParser(description='Knockoffs-SPR')

    # Experiment
    parser.add_argument('--overwrite', default=False, help='Overwrite existed log path, only used for debugging')
    parser.add_argument("--tqdm", type=int, default=0, help='Monitoring the training process')
    parser.add_argument('--config_file', type=str, default=None, help='predefined configs')

    # Data
    parser.add_argument('--dataset', type=str, default="CIFAR10", metavar='DATA', help='Dataset: CIFAR10, CIFAR100, Clothing1M, WebVision')
    parser.add_argument('--root', type=str, default="../data", help='Data root')
    parser.add_argument('--noise_type', type=str, default='clean', help='Noise type: clean, symmetric, asymmetric')
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Noise rate')
    parser.add_argument('--save_dir', type=str, default=None, help='Dirs of saving the logs')


    # Backbone
    parser.add_argument('--gpus', type=str, default='0', help='Indexes of gpus to use')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker for loading data')
    parser.add_argument('--grad_bound', type=float, default=0., help='Gradient norm bound')
    parser.add_argument('--seed', type=int, default=233, help='Choices of random seed to reproduce our experiments')
    parser.add_argument('--backbone', type=str, default=None, help='Choice of backbones: conv2, res18, res50, vgg')
    parser.add_argument('--optimizer', type=str, default=None, help='Optimizers: sgd, adam')
    parser.add_argument('--momentum', type=str, default=None, help='Momentums for optimizers')
    parser.add_argument('--nesterov', type=str, default=False, help='Nesterov for SGD')
    parser.add_argument('--pretrained', default=False, help='Use pretrained model')
    parser.add_argument('--ssl_pretrained', default=None, help='Use pretrained model')
    parser.add_argument('--resume', type=str, default=None, help='Path of pretrained model, default value is set as None to train from scratch')
    parser.add_argument('--lr', type=float, default=None, help='Initial learning rate')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler algorithm for learning rate decay')
    parser.add_argument('--milestones', default=None, help='Learning rate epochs, only used for step lr algorithm')
    parser.add_argument('--gamma', type=float, default=None, help='Learning rate decay parameter, only used for step lr algorithm')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay parameter in optimizer')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--start_epoch', type=int, default=None, help='Starting epoch of training, only used when pretrained model is utilized')
    parser.add_argument('--epochs', type=int, default=None, help='Total training epochs')
    parser.add_argument('--warmup', type=int, default=None, help='Warm up epochs')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--beta', default=None, type=float)

    # SPR
    parser.add_argument('--spr', type=int, default=1, help='Use SPR for training')
    parser.add_argument('--spr_mode', type=str, default='knockoffs')
    parser.add_argument('--num_classes_sub', type=int, default=None, help='Classes in each group')
    parser.add_argument('--num_examples_sub', type=int, default=None, help='Examples of each class in each group')
    parser.add_argument('--permute_strategy', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--reduce_alg', type=str, default=None, help='Reduced alg for features')


    # Cutmix
    parser.add_argument('--augment', type=str, default=None, help='Flag of using augmentation for dataloader')
    parser.add_argument('--ssl_prob', type=float, default=None, help='Probability of using CutMix strategy')

    args = parser.parse_args()

    # make sure the setting of noise rate and noise type are consistent
    assert not (args.noise_rate == 0.0) ^ (args.noise_type == 'clean'), 'Contradictory setting of noise tyoe and noise rate'

    # load the configs from the yaml file if it is not specified in the command line
    assert args.config_file is not None, 'Please load a config file!'
    predefined = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
    for k, v in predefined.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    return args