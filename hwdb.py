import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class HWDB(object):
    def __init__(self, path, transform):
        traindir = os.path.join(path, 'train')
        testdir = os.path.join(path, 'test')
        self.trainset = datasets.ImageFolder(traindir, transform)
        self.testset = datasets.ImageFolder(testdir, transform)
        self.train_size = len(self.trainset)
        self.test_size = len(self.testset)
        self.num_classes = len(self.trainset.classes)
        self.class_to_idx = self.trainset.class_to_idx

    def get_loader(self, batch_size=100, num_workers = 0, pin_memory = False):
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        return trainloader, testloader