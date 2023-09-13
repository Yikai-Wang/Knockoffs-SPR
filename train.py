import os
import os.path as osp
import sys
from collections import defaultdict
from time import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
from ema_pytorch import EMA

from config import config
from dataset import DatasetGenerator
from models.kspr import kspr_parallel, KSPR
from utils import Logger, evaluate, evaluate_top5, rand_bbox, save_checkpoint, set_seed, load_ssl_pretrain, get_statistics

args = config()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = 'cuda:0' if torch.cuda.is_available()  else 'cpu'

if args.seed is not None:
    set_seed(args.seed)

if os.path.exists(args.save_dir) and args.overwrite:
    os.system('rsync -a {0} logs/trashes/ && rm -r {0}'.format(args.save_dir))
    print('Existing log folder, move it to trashes!')

data_loader = DatasetGenerator(data_path=os.path.join(args.root, args.dataset),
                               num_of_workers=args.num_workers,
                               seed=args.seed,
                               train_batch_size=args.batch_size,
                               noise_type=args.noise_type,
                               dataset=args.dataset,
                               noise_rate=args.noise_rate,
                               augment=args.augment,
                               ).getDataLoader()
train_loader, test_loader = data_loader['train_dataset'], data_loader['test_dataset']
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
    if args.ssl_pretrained:
        print('Load SSL pretrained model from {}'.format(args.resume))
        args.start_epoch = 0
        load_ssl_pretrain(model, args.resume)
    else:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        if args.start_epoch is None:
            args.start_epoch = epoch + 1
else:
    args.start_epoch = 0

if len(args.gpus) > 1:
    model = nn.DataParallel(model)

model = model.to(device)

if args.ema:
    ema = EMA(model, beta=0.999, update_after_step = 100,update_every = 10,)

if args.dataset == 'clothing1M':
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
        momentum=args.momentum, weight_decay=args.weight_decay, 
        nesterov=args.nesterov)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
        weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
        weight_decay= args.weight_decay)

if args.scheduler == 'cos':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, 
        eta_min=0.0, last_epoch=-1)
elif args.scheduler == 'step':
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, 
        gamma=args.gamma)

criterion = nn.CrossEntropyLoss(reduction='none')

sys.stdout = Logger(args.save_dir+'.txt', 'a')

print(args)

best_acc = 0
ema_best_acc = 0
if args.spr:
    ep_stats = {}
    ep_stats['label'] = np.array(train_loader.dataset.mislabeled_targets).astype(int)
    num_train = len(ep_stats['label'])
    ep_stats['logit'] = np.zeros((num_train, args.num_classes))
    ep_stats['feature'] = np.zeros((num_train, nFeat))
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        ep_stats['gt'] = np.array(train_loader.dataset.targets).astype(int)
        gt_indicator = np.array(ep_stats['label'] == ep_stats['gt']).astype(np.uint8)

    if (args.ssl_pretrained or args.pretrained) and args.start_epoch >= args.warmup:
        with torch.no_grad():
            for img, target, gt, index in train_loader:
                logit, feature = model(img.to(device))
                feature = feature.detach().cpu().numpy()
                logit = logit.detach().cpu().numpy()
                for batch_idx, true_idx in enumerate(index):
                    ep_stats['logit'][true_idx] = logit[batch_idx]
                    ep_stats['feature'][true_idx] = feature[batch_idx]
        clean_set, q_list = kspr_parallel(
            X=ep_stats['feature'], 
            y=ep_stats['label'], 
            y_permute=ep_stats['logit'],
            num_class=args.num_classes, 
            clean_set=None, 
            num_classes_sub=args.num_classes_sub,
            num_examples_sub=args.num_examples_sub,
            permute_strategy=args.permute_strategy,
            reduce_alg=args.reduce_alg,
            threshold=args.threshold,
            spr_mode=args.spr_mode,
        )
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            TP, FP, TN, FN, Pr, Re, FDR = get_statistics(clean_set, gt_indicator)
            print('Initial TP:{:.2f} FP:{:.2f} TN:{:.2f} FN:{:.2f} Pr:{:.2f} Re:{:.2f} FDR:{:.2f} q:{:.2f}({:.2f}) NoS:{}'.format(
                TP, FP, TN, FN, Pr, Re, FDR, 100 * np.mean(q_list), 100 * np.std(q_list), len(list(clean_set))))
    else:
        clean_set = None

csv_file = open(args.save_dir+'.csv', 'w', newline='')
if args.dataset in ['clothing1M', 'CIFAR10', 'CIFAR100']:
    print('Epoch LR Loss Best Acc Time TP FP TN FN Pr Re FDR q NoS')
    filenames = ['Epoch', 'LR', 'Loss', 'Best', 'Acc', 'Time', 
        'TP', 'FP', 'TN', 'FN', 'Pr', 'Re', 'FDR', 'q', 'NoS']
    
elif args.dataset == 'WebVision':
    print('Epoch\tLR\tLoss\tBest\tAcc\tTop5:\tIN@1\tIN@5\tTime\tq\tNoS')
    filenames = [
        'Epoch', 'LR', 'Loss', 'Best', 'Acc', 'top5', 
        'imagenet_top1','imagenet_top5','Time', 'q', 'NoS'
    ]
else:
    print('Epoch\tLR\tLoss\tBest\tAcc\tTime\tq\tNoS')
    filenames = [
        'Epoch', 'LR', 'Loss', 'Best', 'Acc', 'Time', 'q', 'NoS'
    ]
if args.ema:
    if args.dataset == 'WebVision':
        filenames += ['ema_Best', 'ema_Acc', 'ema_top5', 'ema_imagenet_top1', 'ema_imagenet_top5']
    else:
        filenames += ['ema_Best', 'ema_Acc']
    
csv_writer = csv.DictWriter(csv_file, fieldnames=filenames)
csv_writer.writeheader()

kspr = KSPR(num_class=args.num_classes, 
            reduce_dim=args.num_classes,
            permute_strategy=args.permute_strategy,
            reduce_alg=args.reduce_alg,
            threshold=args.threshold)
for ep in range(args.start_epoch, args.epochs):
    start = time()
    model.train()
    count_info = defaultdict(float)
    if args.tqdm:
        train_loader = tqdm(train_loader, ncols=0)
    for batch in train_loader:
        if args.augment:
            x, x1, y, idx = batch
            
        else:
            x, y, idx = batch

        model.zero_grad()
        optimizer.zero_grad()

        if (not args.ssl_pretrained and not ep and not args.pretrained) or not args.augment or ep < args.warmup:
            x, y = x.to(device), y.to(device)
            logit, feature = model(x)
            loss = criterion(logit, y)
        else:
            r = np.random.rand(1)
            if r >= args.ssl_prob:
                x, y = x.to(device), y.to(device)
                logit, feature = model(x)
                if args.spr:
                    weight = torch.zeros_like(y)
                    for i in range(len(weight)):
                        if int(idx[i]) in clean_set:
                            weight[i] = 1.0
                else:
                     weight = torch.ones_like(y)
                loss = criterion(logit, y) * weight

            else:
                onehot = F.one_hot(y, num_classes=args.num_classes)
                with torch.no_grad():
                    x_gpu = x.to(device)
                    logit, feature = model(x_gpu)
                    logit = logit.detach().cpu()
                    feature = feature.detach().cpu()
                    del x_gpu 
                    x_gpu = x1.to(device)
                    logit1, _ = model(x_gpu)
                    del x_gpu 
                    logit1 = logit1.detach().cpu()
                    p = (logit.softmax(1)+logit1.softmax(1)) / 2
                    p = p**(1/0.5)
                    y_u = p / p.sum(1, keepdim=True)
                    
                labeled = []
                unlabeled = []
                if not args.spr:
                    bs = len(idx)
                    _permutation = np.random.permutation(bs)
                    labeled = _permutation[:int(bs//2)].tolist()
                    unlabeled = _permutation[int(bs//2):].tolist()
                else:
                    for i, id in enumerate(idx):
                        if int(id) in clean_set:
                            labeled.append(i)
                        else:
                            unlabeled.append(i)
                   
                n_labeled = len(labeled)
                orded_idx = labeled + unlabeled
                l = np.random.beta(args.beta, args.beta)
                l = max(l, 1-l)

                rand_idx = torch.randperm(x1.shape[0])
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), l)
                mixed_x = x[orded_idx]
                mixed_y = y[orded_idx]
                mixed_y[n_labeled:] = y_u.argmax(1)[orded_idx][n_labeled:]
                mixed_x[:, :, bbx1:bbx2, bby1:bby2] = mixed_x[rand_idx, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (mixed_x.size()[-1] * mixed_x.size()[-2]))
                target_a = mixed_y
                target_b = mixed_y[rand_idx]
                # compute output
                mixed_logit, _ = model(mixed_x.to(device))
                loss = criterion(mixed_logit, target_a.to(device)) * lam + criterion(mixed_logit, target_b.to(device)) * (1. - lam)
            
            
        loss = loss.mean()
        loss.backward()
        if args.grad_bound:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        if args.ema:
            ema.update()

        if args.spr:
            feature = feature.detach().cpu().numpy()
            if ep < args.warmup or (not args.ssl_pretrained and not ep) or r >= args.ssl_prob:
                logit_batch = logit.detach().cpu().numpy()
            else:
                logit_batch = y_u.cpu().numpy()
            for batch_idx, true_idx in enumerate(idx):
                ep_stats['logit'][true_idx] = logit_batch[batch_idx]
                ep_stats['feature'][true_idx] = feature[batch_idx]

        count_info['loss'] += loss.item()
        count_info['num_batches'] += 1

    lr = scheduler.get_last_lr()[0]
    scheduler.step()
    
    if args.spr and (ep >= args.warmup - 1):
        clean_set, q_list = kspr_parallel(
            X=ep_stats['feature'], 
            y=ep_stats['label'], 
            y_permute=ep_stats['logit'],
            num_class=args.num_classes, 
            clean_set=clean_set, 
            num_classes_sub=args.num_classes_sub,
            num_examples_sub=args.num_examples_sub,
            permute_strategy=args.permute_strategy,
            reduce_alg=args.reduce_alg,
            threshold=args.threshold,
            spr_mode=args.spr_mode,
        )
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            TP, FP, TN, FN, Pr, Re, FDR = get_statistics(clean_set, gt_indicator)

    torch.cuda.empty_cache()

    if args.dataset == 'WebVision':
        top1, top5 = evaluate_top5(test_loader, model, device)
        top1_imagenet, top5_imagenet = evaluate_top5(test_loader_imagenet, model, device)
        test_acc = top1
    else:
        test_acc = evaluate(test_loader, model, device)

    torch.cuda.empty_cache()

    if args.ema:
        if args.dataset == 'WebVision':
            ema_top1, ema_top5 = evaluate_top5(test_loader, ema, device)
            ema_top1_imagenet, ema_top5_imagenet = evaluate_top5(test_loader_imagenet, ema, device)
            ema_test_acc = ema_top1
        else:
            ema_test_acc = evaluate(test_loader, ema, device)

    torch.cuda.empty_cache()

    

    if test_acc > best_acc:
        best_acc = test_acc
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        save_checkpoint({
            'epoch': ep,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(args.save_dir, 'best_model.pth.tar'))
    if args.ema:
        if ema_test_acc > ema_best_acc:
            ema_best_acc = ema_test_acc
            ema_state_dict = ema.module.state_dict() if hasattr(ema, "module") else ema.state_dict()
            save_checkpoint({
                'epoch': ep,
                'model_state_dict': ema_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, osp.join(args.save_dir, 'best_ema_model.pth.tar'))
    
    end = time()  

    print_info = '{}/{} {:.4f} {:.3f} '.format(
        ep, args.epochs, lr, count_info['loss'] / count_info['num_batches'])
    csv_row = {'Epoch':ep, 'LR':lr, 
        'Loss':count_info['loss'] / count_info['num_batches']}
    
    if args.dataset == 'WebVision':
        print_info += '{:.2f} {:.2f}({:.2f}) {:.2f}({:.2f}) '.format(
            100 * best_acc, 100 * test_acc, 100 * top5, 
            100 * top1_imagenet, 100 * top5_imagenet)
        csv_row.update({'Best':100 * best_acc, 'Acc':100 * test_acc, 'top5':100 * top5, 
        'imagenet_top1':100 * top1_imagenet, 'imagenet_top5':100 * top5_imagenet})
    else:
        print_info += '{:.2f} {:.2f} '.format(100 * best_acc, 100 * test_acc)
        csv_row.update({'Best':100 * best_acc, 'Acc':100 * test_acc})
    
    if args.ema:
        if args.dataset == 'WebVision':
            print_info += '{:.2f} {:.2f}({:.2f}) {:.2f}({:.2f}) '.format(
                100 * ema_best_acc, 100 * ema_test_acc, 100 * ema_top5, 
                100 * ema_top1_imagenet, 100 * ema_top5_imagenet)
            csv_row.update({'ema_Best':100 * ema_best_acc, 'ema_Acc':100 * ema_test_acc, 'ema_top5':100 * ema_top5, 
            'ema_imagenet_top1':100 * ema_top1_imagenet, 'ema_imagenet_top5':100 * ema_top5_imagenet})
        else:
            print_info += '{:.2f} {:.2f} '.format(100 * ema_best_acc, 100 * ema_test_acc)
            csv_row.update({'ema_Best':100 * ema_best_acc, 'ema_Acc':100 * ema_test_acc})


    cost_time = '{:2d}m{:2d}s '.format(int((end-start)//60), int((end-start)%60))
    print_info += cost_time
    csv_row.update({'Time':cost_time})

    if args.dataset in ['CIFAR10', 'CIFAR100'] and (ep >= args.warmup - 1) and args.spr:
        print_info += '{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
                TP, FP, TN, FN, Pr, Re, FDR
            )
        csv_row.update({'TP':TP, 'FP':FP, 'TN':TN, 'FN':FN, 'Pr':Pr, 'Re':Re, 'FDR':FDR})
    if args.spr and (ep >= args.warmup - 1):
        q_statistics = ' {:.2f}({:.2f})'.format(100 * np.mean(q_list), 100 * np.std(q_list))
        print_info += q_statistics
        print_info += ' {}'.format(len(list(clean_set)))
        csv_row.update({'q':q_statistics})
        csv_writer.writerow(csv_row)
    print(print_info)