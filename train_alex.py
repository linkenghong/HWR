from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from PIL import Image
from model import HWDB_GoogLeNet, HWDB_AlexNet
from hwdb import HWDB

def valid(epoch, model, test_loarder):
    print("epoch %d 开始验证..." % epoch)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loarder:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('correct number: ', correct)
        print('totol number:', total)
        acc = 100 * correct / total
        print('第%d个epoch的识别准确率为：%.04f%%' % (epoch, acc))


def train(epoch, model, criterion, optimizer, train_loader, save_iter=100):
    print("epoch %d 开始训练..." % epoch)
    model.train()
    sum_loss = 0.0
    total = 0
    correct = 0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        if (i + 1) % save_iter == 0:
            batch_loss = sum_loss / save_iter
            acc = 100 * correct / total
            print('epoch: %d, batch: %d loss: %.03f, acc: %.04f'
                  % (epoch, i + 1, batch_loss, acc))
            total = 0
            correct = 0
            sum_loss = 0.0

if __name__ == "__main__":
    # 超参数
    epochs = 10000
    batch_size = 500
    lr = 0.0001
    degrees = 15

    data_path = r'data'
    pth_path = r'checkpoints/'
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)

    with open('char_dict_new_new', 'rb') as f:
        class_dict = pickle.load(f)
    num_classes = len(class_dict)

    transform = transforms.Compose([
        transforms.Resize((108, 108)),
        # transforms.RandomRotation(degrees, resample=Image.BICUBIC, expand=False, center=None, fill=255),
        transforms.ToTensor(),
    ])

    dataset = HWDB(path=data_path, transform=transform)
    print("训练集数据:", dataset.train_size)
    print("测试集数据:", dataset.test_size)
    trainloader, testloader = dataset.get_loader(batch_size, num_workers = 8, pin_memory = True)

    model = HWDB_AlexNet(num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    # model.load_state_dict(torch.load('checkpoints/handwriting_iter_68.pth'))

    #####################################################################################
    # 减少种类后
    with open('pretrained_dict', 'rb') as f:
        pretrained_dict = pickle.load(f)

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    #####################################################################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(69, epochs):
        train(epoch, model, criterion, optimizer, trainloader)
        valid(epoch, model, testloader)
        print("epoch%d 结束, 正在保存模型..." % epoch)
        torch.save(model.state_dict(), pth_path + 'HWDB_AlexNet_iter_%d.pth' % epoch)