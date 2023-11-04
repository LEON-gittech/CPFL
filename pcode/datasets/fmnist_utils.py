from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST
import copy

class FMNISTPair(FashionMNIST):
    """FMNIST Dataset"""
    def __getitem__(self, index):
        img, target = self.data[index].numpy(), self.targets[index].numpy()
        tmp = np.zeros((32, 32, 3), dtype=np.uint8)
        for i in range(3):
            tmp[:,:,i] = np.resize(img, (32, 32)).astype(np.uint8)
        img = Image.fromarray(tmp)
        del tmp
        

        img_norm = test_transform(img) #这里不能直接返回 img，而是要返回归一化后的 img
        if self.target_transform is not None:
                target = self.target_transform(target)
        if self.train:
            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)
                return pos_1, pos_2, img_norm, target
            else:
                return img_norm, target
        else:
            return img_norm, target

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

if __name__ == "__main__":
    data = FashionMNIST(root="/home/leon/workspace/pFedSD/data", train=True, download=True)