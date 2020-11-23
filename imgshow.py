from torchvision.utils import save_image
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import sys
from alfred.utils.log import logger as logging

def imgshow(char_index):
    train_dir = r'./data/train'
    path = os.path.join(train_dir, char_index)
    img_path_list = glob.glob(os.path.join(path, '*.png'))

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    count = 1
    plt.figure()

    for p in img_path_list:
        if count > 4:
            break
        plt.subplot(2,2,count)
        input = Image.open(p).convert('RGB')
        input = transform(input)
        plt.imshow(input.permute(1,2,0))
        count += 1

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        logging.error('send a pattern like this: {}'.format('06203'))
    else:
        p = sys.argv[1]
        logging.info('show img from: {}'.format(p))
        imgshow(p)