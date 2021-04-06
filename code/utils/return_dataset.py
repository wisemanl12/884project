import os
import os.path
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def xray_dataset_frompath(datapath, num_labels):
    normal_files = glob.glob(join(datapath, 'NORMAL', "*.jpeg"))
    normal_label = [0]*len(normal_files)
    pneumonia_files = glob.glob(join(datapath, 'PNEUMONIA', "*.jpeg"))
    pneumonia_label = [1]*len(pneumonia_files)
    if num_labels==3:
        for i in range(0,len(pneumonia_files)):
            tt = pneumonia_files[i].split('_')[2]
            if tt[0]=='v':
                pneumonia_label[i] = 2
    image_list = normal_files+pneumonia_files
    label_list = normal_label+pneumonia_label

    image_list = np.array(image_list)
    label_list = np.array(label_list)
    return image_list, label_list


class DatasetXRay(object):
    def __init__(self, datapath, num_labels, transform=None, target_transform=None):
        imgs, labels = xray_dataset_frompath(datapath, num_labels)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


def return_dataset(args):    
    train_path = os.path.join(args.datapath,'train')
    val_path = os.path.join(args.datapath,'test')
    test_path = os.path.join(args.datapath,'val')    

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    target_size = 256
    if args.net == 'inception_v3':
      target_size = 300
      crop_size = 299
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(target_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = DatasetXRay(train_path, args.num_labels, transform=data_transforms['train'])
    val_dataset = DatasetXRay(val_path, args.num_labels, transform=data_transforms['val'])
    test_dataset = DatasetXRay(test_path, args.num_labels, transform=data_transforms['test'])                                      

    if args.num_labels==2:
        class_list = [0,1]
    else:
        class_list = [0,1,2]
    print("%d classes in this dataset" % len(class_list))

    train_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, num_workers=3, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=3, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=3, shuffle=True, drop_last=False)

    return train_loader, val_loader, test_loader, class_list
