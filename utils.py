
import errno
import os
import os.path as osp
import random
import shutil
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def evaluate(loader, model, device):
    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z, _ = model(x)
            pred = torch.argmax(z, 1)
            total += y.size(0)
            correct += (pred==y).sum().item()

    acc = float(correct) / float(total)
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def evaluate_top5(loader, model, device):
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0 / batch_size))
            return res
    top1 = AverageMeter('Acc@1', ':6.4f')
    top5 = AverageMeter('Acc@5', ':6.4f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z, _ = model(x)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(z, y, topk=(1, 5))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

        # TODO: this should also be done with the ProgressMeter
        # # print(' * Acc@1 {top1.avg:.4f} Acc@5 {top5.avg:.4f}'
        #       .format(top1=top1, top5=top5))

    return top1.avg, top5.avg
    
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None, mode='a'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_checkpoint(state, fpath='checkpoint.pth.tar', is_best=False):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def load_ssl_pretrain(model, checkpoint):
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
        selected_ckpt = {k.split('encoder.module.')[1].replace('shortcut','downsample'):v for k,v in checkpoint.items() if 'encoder.module' in k and 'fc' not in k}
        
    elif 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
        selected_ckpt = {k.split('encoder.')[1]:v for k,v in checkpoint.items() if 'encoder' in k and 'fc' not in k}
        
    
    missing, unexpected = model.load_state_dict(selected_ckpt, strict=False)
    print('Model missing keys:\n', missing)
    print('Model unexpected keys:\n', unexpected)

def get_statistics(clean_set, gt_indicator):
    clean_set_indicator = set(clean_set)
    num = len(gt_indicator)

    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    for i in range(num):
        if i not in clean_set_indicator:
            if gt_indicator[i] == 0:
                true_negative += 1
            else:
                false_negative += 1
        else:
            if gt_indicator[i] == 0:
                false_positive += 1
            else:
                true_positive += 1
    
    TP = 100 * true_positive/num
    FP = 100 * false_positive/num
    TN = 100 * true_negative/num
    FN = 100 * false_negative/num
    Pr = 100 * true_positive / (true_positive + false_positive)
    Re = 100 * true_positive / (true_positive + false_negative)
    FDR = 100 * false_positive/ (false_positive + true_positive)
    return TP, FP, TN, FN, Pr, Re, FDR

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

def consistency_loss(logits_s, logits_w, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        select = max_probs.ge(p_cutoff).long()
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long()

    else:
        assert Exception('Not Implemented consistency_loss')

def logit2permute(logit, y):
    y_permute = []
    y_pred =  logit.argsort(axis=-1)
    for ind, label in enumerate(y):
        if y_pred[ind,-1] == label:
            y_permute.append(y_pred[ind,-2])
        else:
            y_permute.append(y_pred[ind,-1])
    return np.array(y_permute)