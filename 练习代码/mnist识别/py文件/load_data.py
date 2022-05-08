import os
import random
import cv2

from torch.utils.data import Dataset


def image_root(image_root, txt):
    f = open(txt, 'wt')
    
    for (label, filename) in enumerate(sorted(os.listdir(image_root), reverse = False)):
        #print(label, filename)
        if os.path.isdir(os.path.join(image_root, filename)):
            for imagename in os.listdir(os.path.join(image_root, filename)):
                name, ext = os.path.splitext(imagename)
                ext = ext[1:]
                if ext == 'jpg' or ext == 'png' or ext == 'bmp':
                    f.write('%s %d\n'%(os.path.join(image_root, filename, imagename), label))


def shuffle_split(totalFile, trainFile, valFile):
    with open(totalFile, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    num = len(lines)
    trainNum = int(num * 0.8)
    with open(trainFile, 'w') as f:
        f.writelines(lines[:trainNum])
    with open(valFile, 'w') as f1:
        f1.writelines(lines[trainNum:])




class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    
    
