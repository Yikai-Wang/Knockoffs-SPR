import os

import torch
import torch.nn as nn

from config import config
from dataset import DatasetGenerator
from utils import evaluate, evaluate_top5, set_seed

args = config()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = 'cuda:0' if torch.cuda.is_available()  else 'cpu'

if args.seed is not None:
    set_seed(args.seed)

data_loader = DatasetGenerator(data_path=os.path.join(args.root, args.dataset),
                               num_of_workers=args.num_workers,
                               seed=args.seed,
                               train_batch_size=args.batch_size,
                               noise_type=args.noise_type,
                               dataset=args.dataset,
                               noise_rate=args.noise_rate,
                               augment=args.augment,
                               ).getDataLoader()
test_loader = data_loader['test_dataset']
if args.dataset == 'WebVision':
    test_loader_imagenet = data_loader['test_imagenet']

if args.backbone == 'res18' and 'CIFAR' in args.dataset:
    from models.resnet_cifar import resnet18
    model = resnet18(num_classes=args.num_classes, show=True)
    nFeat = 512
elif args.backbone == 'inception':
    from models.inception import InceptionResNetV2
    model = InceptionResNetV2(num_classes=args.num_classes, show=True)
    nFeat = 1536
elif args.backbone == 'res50':
    from models.resnet import resnet50
    model = resnet50(num_classes=args.num_classes, show=True)
    nFeat = 2048
    if args.pretrained:
        from torchvision.models.resnet import ResNet50_Weights
        state_dict = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
        state_dict = {k:v for k,v in state_dict.items() if 'fc' not in k}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print('Loading ImageNet pretrained model')
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)
else:
    raise NameError

if args.resume is not None:
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys()):
        if 'ema_model' in key:
            state_dict[key.replace('ema_model.', '')] = state_dict[key]
            del state_dict[key]
        else:
            del state_dict[key]
    model.load_state_dict(state_dict)
    epoch = checkpoint['epoch']
    if args.start_epoch is None:
        args.start_epoch = epoch + 1
else:
    raise NotImplementedError

if len(args.gpus) > 1:
    model = nn.DataParallel(model)
model = model.to(device)

if args.dataset == 'WebVision':
    top1, top5 = evaluate_top5(test_loader, model, device)
    top1_imagenet, top5_imagenet = evaluate_top5(test_loader_imagenet, model, device)
    print('Test Accuracy of WebVision: {:.2f}({:.2f}) ImageNet: {:.2f}({:.2f})'.format(100 * top1, 100 * top5, 100 * top1_imagenet, 100 * top5_imagenet))
else:
    test_acc = evaluate(test_loader, model, device)
    print('Test Accuracy of {}: {:.2f}'.format(args.dataset, 100*test_acc))
