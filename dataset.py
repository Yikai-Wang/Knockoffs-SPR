import os
from copy import deepcopy
from os import listdir

import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

import simsiam.loader
from randaugment import RandAugment


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

def mislabel(y, noise_type, noise_rate, noise_pairs=None, seed=None):
    np.random.seed(seed)
    mislabeled_y = deepcopy(y)
    nb_classes = len(np.unique(mislabeled_y))
    n_samples = len(mislabeled_y)
    if noise_type == 'symmetric':
        class_index = [np.where(np.array(mislabeled_y) == i)[0] for i in range(nb_classes)]
        mislabeled_idx = []
        for d in range(nb_classes):
            n_img = len(class_index[d])
            n_mislabeled = int(noise_rate * n_img)
            mislabeled_class_index = np.random.choice(class_index[d], n_mislabeled, replace=False)
            mislabeled_idx.extend(mislabeled_class_index)
            print("Class:{} Images:{} Mislabeled:{}".format(d, n_img, n_mislabeled))
        print("Total:{} Mislabeled:{}".format(n_samples, len(mislabeled_idx)))
        for i in mislabeled_idx:
            mislabeled_y[i] = other_class(n_classes=nb_classes, current_class=mislabeled_y[i])
    elif noise_type == 'asymmetric':
        total_mislabeld = 0
        for s, t in noise_pairs:
            class_index = np.where(np.array(mislabeled_y) == s)[0]
            n_img = len(class_index)
            n_mislabeled = int(noise_rate * n_img)
            mislabeled_class_index = np.random.choice(class_index, n_mislabeled, replace=False)
            total_mislabeld += n_mislabeled
            for i in mislabeled_class_index:
                mislabeled_y[i] = t
            print("Class:{} Images:{} Mislabeled:{}".format(s, n_img, n_mislabeled))
        print("Total:{} Mislabeled:{}".format(n_samples, total_mislabeld))
    return mislabeled_y

class DatasetGenerator():
    def __init__(self,
                 train_batch_size=128,
                 eval_batch_size=256,
                 data_path='data/',
                 seed=233,
                 num_of_workers=4,
                 noise_type='clean',
                 dataset=None,
                 noise_rate=0.0,
                 augment=False,
                 ssl=False,
                 ):
        self.seed = seed
        np.random.seed(seed)
        self.augment = augment
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_path = data_path
        self.num_of_workers = num_of_workers
        self.noise_rate = noise_rate
        self.dataset = dataset
        self.noise_type = noise_type
        self.ssl = ssl
        self.data_loaders = self.loadData()

    def getDataLoader(self):
        return self.data_loaders
    
    def getTrainSet(self):
        return self.train_dataset

    def loadData(self):    
        if self.dataset == 'CIFAR10':
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            if self.ssl:
                train_transform = simsiam.loader.TwoCropsTransform(
                    transforms.Compose([
                        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                    ]))
                train_dataset = CIFAR10(
                    root=self.data_path,
                    train=True,
                    transform=train_transform,
                    download=True,
                    seed=self.seed,
                    noise_type=self.noise_type,
                    noise_rate=self.noise_rate,
                    augment=self.augment,
                    )
            else:
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

                train_dataset = CIFAR10(
                    root=self.data_path,
                    train=True,
                    transform=train_transform,
                    download=True,
                    seed=self.seed,
                    noise_type=self.noise_type,
                    noise_rate=self.noise_rate,
                    augment=self.augment,
                    )

                test_dataset = datasets.CIFAR10(
                    root=self.data_path,
                    train=False,
                    transform=test_transform,
                    download=False)

        elif self.dataset == 'CIFAR100':
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]

            if self.ssl:
                train_transform = simsiam.loader.TwoCropsTransform(
                    transforms.Compose([
                        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                    ]))
                train_dataset = CIFAR100(
                    root=self.data_path,
                    train=True,
                    transform=train_transform,
                    download=True,
                    seed=self.seed,
                    noise_type=self.noise_type,
                    noise_rate=self.noise_rate,
                    augment=self.augment,
                    )
            else:
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

                train_dataset = CIFAR100(
                    root=self.data_path,
                    train=True,
                    transform=train_transform,
                    download=True,
                    seed=self.seed,
                    noise_type=self.noise_type,
                    noise_rate=self.noise_rate,
                    augment=self.augment,
                    )

                test_dataset = datasets.CIFAR100(
                    root=self.data_path,
                    train=False,
                    transform=test_transform,
                    download=False)
        
        elif self.dataset == 'WebVision':
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]

            if self.ssl:
                train_transform = simsiam.loader.TwoCropsTransform(
                    transforms.Compose([
                        transforms.RandomResizedCrop(299, scale=(0.2, 1.)),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(MEAN, STD)
                    ]))
                train_dataset = WebVision(
                    root=self.data_path,
                    train=True,
                    transform=train_transform,
                    augment=self.augment,
                    )
            else:
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)])

                strong_transform = transforms.Compose([
                    RandAugment(3, 5),
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)])

                test_transform = transforms.Compose([
                    transforms.Resize(320),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)])

                train_dataset = WebVision(
                    root=self.data_path,
                    train=True,
                    transform=train_transform,
                    strong_transform=strong_transform,
                    augment=self.augment,
                    )

                test_dataset = WebVision(
                    root=self.data_path,
                    train=False,
                    transform=test_transform)
                imagenet_val = ImageNetVal(transform=test_transform)
        
        elif self.dataset == 'clothing1M':
            MEAN = (0.485, 0.456, 0.406)
            STD = (0.229, 0.224, 0.225)

            if self.ssl:
                train_transform = simsiam.loader.TwoCropsTransform(
                    transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(MEAN, STD)
                    ]))
                train_dataset = Clothing1M(
                    root=self.data_path,
                    train=True,
                    transform=train_transform,
                    augment=self.augment,
                    mode=self.noise_type,
                    )
            else:
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ])

                test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)])

                train_dataset = Clothing1M(
                    root=self.data_path,
                    train=True,
                    transform=train_transform,
                    augment=self.augment,
                    mode=self.noise_type,
                    )

                test_dataset = Clothing1M(
                    root=self.data_path,
                    train=False,
                    transform=test_transform)
        else:
            raise("Unknown Dataset")

        data_loaders = {} 
        self.train_dataset = train_dataset
        print("Num of train %d" % (len(train_dataset)))
        if not self.ssl:
            data_loaders['train_dataset'] = DataLoader(
                dataset=train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=self.num_of_workers)

            data_loaders['test_dataset'] = DataLoader(
                dataset=test_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_of_workers)
            if self.dataset == 'WebVision':
                data_loaders['test_imagenet'] = DataLoader(
                    dataset=imagenet_val,
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=self.num_of_workers)

            print("Num of test %d" % (len(test_dataset)))

        return data_loaders

class CIFAR10(datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 noise_type=None, noise_rate=0.2, seed=0, augment=False):
        super(CIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.targets = self.targets
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.augment = augment
        if noise_rate > 0:
            if noise_type == 'asymmetric':
                noise_pairs = [(9, 1), (2, 0), (3, 5), (5, 3), (4, 7)] # source -> target
                self.mislabeled_targets = mislabel(self.targets, noise_type, noise_rate, noise_pairs=noise_pairs, seed=seed)
            elif noise_type == 'symmetric':
                self.mislabeled_targets = mislabel(self.targets, noise_type, noise_rate, seed=seed)
            actual_noise = (np.array(self.mislabeled_targets) != np.array(self.targets)).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
        else:
            self.mislabeled_targets = self.targets

    
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.mislabeled_targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_origin = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img_origin)
            if self.augment:
                img1 = self.transform(img_origin)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train:
            if self.augment:
                return img, img1, target, index
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100(datasets.CIFAR100):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 noise_type=None, noise_rate=0.2, seed=0, augment=False):
        super(CIFAR100, self).__init__(root, train, transform, target_transform, download)
        self.targets = self.targets
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.augment = augment
        if noise_rate > 0:
            if noise_type == 'asymmetric':
                # mislabelling appears within the same superclass
                sub_classes = [
                    [4, 30, 55, 72, 95], [1, 32, 67, 73, 91], 
                    [54, 62, 70, 82, 92], [9, 10, 16, 28, 61], 
                    [0, 51, 53, 57, 83], [22, 39, 40, 86, 87], 
                    [5, 20, 25, 84, 94], [6, 7, 14, 18, 24], 
                    [3, 42, 43, 88, 97], [12, 17, 37, 68, 76], 
                    [23, 33, 49, 60, 71], [15, 19, 21, 31, 38], 
                    [34, 63, 64, 66, 75], [26, 45, 77, 79, 99], 
                    [2, 11, 35, 46, 98], [27, 29, 44, 78, 93], 
                    [36, 50, 65, 74, 80], [47, 52, 56, 59, 96], 
                    [8, 13, 48, 58, 90], [41, 69, 81, 85, 89],
                ]
                noise_pairs = []
                for sub in sub_classes:
                    for i in range(len(sub)-1):
                        # source -> target
                        noise_pairs.append((sub[i],sub[i+1]))
                    noise_pairs.append((sub[-1], sub[0]))
                self.mislabeled_targets = mislabel(self.targets, noise_type, noise_rate, noise_pairs=noise_pairs, seed=seed)
            elif noise_type == 'symmetric':
                self.mislabeled_targets = mislabel(self.targets, noise_type, noise_rate, seed=seed)
            actual_noise = (np.array(self.mislabeled_targets) != np.array(self.targets)).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
        else:
            self.mislabeled_targets = self.targets
    
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.mislabeled_targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_origin = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img_origin)
            if self.augment:
                img1 = self.transform(img_origin)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train:
            if self.augment:
                return img, img1, target, index
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class WebVision(Dataset):
    def __init__(self, root, train=True, transform=None, strong_transform=None, target_transform=None, augment=False, num_class=50):
        self.root = root.lower()
        self.transform = transform
        self.strong_transform = strong_transform
        self.target_transform = target_transform
        self.augment = augment
        self.train = train
        num_images = {i:0 for i in range(num_class)}
        if self.train:
            with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
                lines = f.readlines()
            if num_class == 1000:
                with open(os.path.join(self.root, 'info/train_filelist_flickr.txt')) as f:
                    lines += f.readlines()
            self.data = []
            self.mislabeled_targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.data.append(img)
                    self.mislabeled_targets.append(target)
                    num_images[target] += 1
        else:
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()
            self.data = []
            self.targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.data.append(img)
                    self.targets.append(target)


    
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.mislabeled_targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.train:
            img_origin = Image.open(os.path.join(self.root, img)).convert('RGB')
        else:
            img_origin = Image.open(os.path.join(self.root, 'val_images_256/', img)).convert('RGB')
            

        if self.transform is not None:
            img = self.transform(img_origin)
            if self.augment == 'cutmix':
                img1 = self.transform(img_origin)
            elif self.augment == 'strong':
                img1 = self.strong_transform(img_origin)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            if self.augment:
                return img, img1, target, index
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class Clothing1M(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, augment=False, mode='noisy'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment
        self.train = train
        self.mode = mode

        if self.train:
            self.data = []
            self.mislabeled_targets = []
            if mode == 'noisy':
                train_labels = {}
                with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
                    lines = f.read().splitlines()
                    for l in lines:
                        entry = l.split()           
                        img_path = '%s/'%self.root+entry[0]
                        train_labels[img_path] = int(entry[1])   
                with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                    lines = f.read().splitlines()
                    for l in lines:
                        img_path = '%s/'%self.root+l
                        self.data.append(img_path)
                        self.mislabeled_targets.append(train_labels[img_path])  
            elif mode == 'clean':
                train_labels = {}
                with open('%s/clean_label_kv.txt'%self.root,'r') as f:
                    lines = f.read().splitlines()
                    for l in lines:
                        entry = l.split()           
                        img_path = '%s/'%self.root+entry[0]
                        train_labels[img_path] = int(entry[1])   
                with open('%s/clean_train_key_list.txt'%self.root,'r') as f:
                    lines = f.read().splitlines()
                    for l in lines:
                        img_path = '%s/'%self.root+l
                        self.data.append(img_path)
                        self.mislabeled_targets.append(train_labels[img_path])  
                with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                    lines = f.read().splitlines()
                    for l in lines:
                        img_path = '%s/'%self.root+l
                        self.data.append(img_path)
                        self.mislabeled_targets.append(train_labels[img_path])  
            else:
                raise NameError                
        else:
            test_labels = {}
            with open('%s/clean_label_kv.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = '%s/'%self.root+entry[0]
                    test_labels[img_path] = int(entry[1])  
            self.data = []
            self.targets = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l
                    self.data.append(img_path)
                    self.targets.append(test_labels[img_path])


    
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.mislabeled_targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img_origin = Image.open(img).convert('RGB')
            

        if self.transform is not None:
            img = self.transform(img_origin)
            if self.augment:
                img1 = self.transform(img_origin)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            if self.augment:
                return img, img1, target, index
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class ImageNetVal(Dataset):
        def __init__(self, transform, root_dir='../data/imagenet/', num_class=50):
            class2index_path = root_dir + 'meta.mat'
            meta = sio.loadmat(class2index_path)
            imagenet_class2index = {}
            for i in range(len(meta['synsets'])):
                imagenet_class2index[meta['synsets'][i][0][1][0]]=meta['synsets'][i][0][0][0][0]
            synsets_path = os.path.join('../data/webvision/info/synsets.txt')
            webvision_classes = open(synsets_path, 'r').read().splitlines()
            webvision_classes = [l.split()[0] for l in webvision_classes[:num_class]]
            selected_labels = [imagenet_class2index[c] for c in webvision_classes]
            imagenet2webvision = {c:i for i,c in enumerate(selected_labels)}
            self.path = root_dir + 'val/'
            self.val_data = []
            self.transform = transform
            paths = os.listdir(self.path)
            paths.sort()
            with open(root_dir+'ILSVRC2012_validation_ground_truth.txt', 'r') as f:
                lines = f.read().splitlines()
                lines = [int(l.split()[0]) for l in lines]
                for i, l in enumerate(lines):
                    if l in selected_labels:
                        self.val_data.append((os.path.join(self.path, paths[i]), imagenet2webvision[l]))    
                    
        def __getitem__(self, index):
            img, target = self.val_data[index]
            image = Image.open(img).convert('RGB')   
            img = self.transform(image) 
            return img, int(target)
        
        def __len__(self):
            return len(self.val_data)
