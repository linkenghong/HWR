import pickle
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont
from model import HWDB_GoogLeNet, HWDB_AlexNet, HWDB_MobileNet, HWDB_SegNet


def image_prediction(img, net_num = 825, topk = 30, labels = None, net_type = 'G'):
    with open('char_dict_new_new', 'rb') as f:
        char_dict = pickle.load(f)
    index_char_dict = {index: key for key, index in char_dict.items()}
    num_classes = len(char_dict)
    # torch.cuda.set_device(1)

    g_transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
    ])
    m_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    a_transform = transforms.Compose([
        transforms.Resize((108, 108)),
        transforms.ToTensor(),
    ])
    pth_dir = r'./checkpoints'

    if net_type == 'G':
        input = g_transform(img)
        input = input.unsqueeze(0)
        net = HWDB_GoogLeNet(num_classes)
        net_path = os.path.join(pth_dir, 'HWDB_GoogLeNet_iter_' + str(net_num) +'.pth')
    elif net_type == 'M':
        input = m_transform(img)
        input = input.unsqueeze(0)
        net = HWDB_MobileNet(num_classes)
        net_path = os.path.join(pth_dir, 'handwriting_iter_' + str(net_num) +'.pth')
    elif net_type == 'A':
        input = a_transform(img)
        input = input.unsqueeze(0)
        net = HWDB_AlexNet(num_classes)
        net_path = os.path.join(pth_dir, 'HWDB_AlexNet_iter_' + str(net_num) +'.pth')
    elif net_type == 'S':
        input = g_transform(img)
        input = input.unsqueeze(0)
        net = HWDB_SegNet(num_classes)
        net_path = os.path.join(pth_dir, 'HWDB_SegNet_iter_' + str(net_num) +'.pth')

    if torch.cuda.is_available():
        net = net.cuda()
        input = input.cuda()
    net.load_state_dict(torch.load(net_path))
    net.eval()

    output = net(input)
    output_sm = nn.functional.softmax(output)

    topk_list = torch.topk(output_sm, topk)[1].tolist()[0]
    topk_char = [index_char_dict[ind] for ind in topk_list]
    conf, pred = torch.max(output_sm.data, 1)
    pred_char = index_char_dict[pred.item()]
    conf = conf.item()

    if labels == None:
        return pred_char, conf, topk_char
    else:
        label_confs = []
        for label in labels:
            try:
                label_conf = output_sm.tolist()[0][char_dict[label]]
                label_confs.append(label_conf)
            except:
                label_confs.append(0)
        return pred_char, conf, topk_char, label_confs







