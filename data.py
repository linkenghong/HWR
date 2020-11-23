import numpy as np
import os
import pickle
import glob
import struct
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor

def read_from_gnt_dir(dir):
    def read_from_gnt(gnt):
        header_size = 10
        with open(gnt, 'rb') as f:
            while True:
                header = np.fromfile(f, dtype='uint8', count=header_size)
                if not header.size:
                    break
                sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
                tagcode = header[5] + (header[4] << 8)
                width = header[6] + (header[7] << 8)
                height = header[8] + (header[9] << 8)
                if header_size + width * height != sample_size:
                    break
                image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                yield image, tagcode
    
    all_gnt_files = glob.glob(os.path.join(dir, '*.gnt'))
    for gnt in all_gnt_files:
        for image, tagcode in read_from_gnt(gnt):
            yield image, tagcode


def save_char_list(*dirs):
    char_set = set()
    num = 1
    for dir in dirs:        
        for _, tagcode in read_from_gnt_dir(dir):
            try:
                label = struct.pack('>H', tagcode).decode('gb2312')
                char_set.add(label)
            except Exception as e:        # 数据集中包含不在gb2312标准中的字符，所以要跳过
                continue
                
    char_list = list(char_set)
    char_dict = dict(zip(sorted(char_list), range(len(char_list))))
    print(len(char_dict))
    print("char_dict=", char_dict)

    with open('char_dict', 'wb') as f:
        pickle.dump(char_dict, f)
    
    with open('characters.txt', 'w') as f:
        for char in char_dict.keys():
            f.write(char + '\n')


def gnt_to_img(gnt_dir, img_dir):
    def save_img(tagcode, image, counter):
        label = struct.pack('>H', tagcode).decode('gb2312')
        img = Image.fromarray(image)
        dir_name = os.path.join(img_dir, '%0.5d' % char_dict[label])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        img.convert('RGB').save(dir_name + '/' + str(counter) + '.png')
        print("thread: {}, counter=".format(threading.current_thread().name), counter)

    counter = 0 
    thread_pool = ThreadPoolExecutor(4)  # 定义4个线程执行此任务
    for image, tagcode in read_from_gnt_dir(gnt_dir):
        thread_pool.submit(save_img, tagcode, image, counter)
        counter += 1
    thread_pool.shutdown()


if __name__ == "__main__":
    HWDB0_train_gnt_dir = r'./data/HWDB1.0trn_gnt'
    HWDB0_test_gnt_dir = r'./data/HWDB1.0tst_gnt'
    HWDB2_train_gnt_dir = r'./data/HWDB1.2trn_gnt'
    HWDB2_test_gnt_dir = r'./data/HWDB1.2tst_gnt'
    train_img_dir = r'./data/train'
    test_img_dir = r'./data/test'

    if not os.path.exists('char_dict'):
        save_char_list(HWDB0_train_gnt_dir, HWDB2_train_gnt_dir)
    else:
        with open('char_dict', 'rb') as f:
            char_dict = pickle.load(f)
    
    if not os.path.exists(train_img_dir):
        os.mkdir(train_img_dir)
    if not os.path.exists(test_img_dir):
        os.mkdir(test_img_dir)
    
    train_thread_1 = threading.Thread(target=gnt_to_img, args=(HWDB0_train_gnt_dir, train_img_dir)).start()
    train_thread_2 = threading.Thread(target=gnt_to_img, args=(HWDB2_train_gnt_dir, train_img_dir)).start()
    test_thread_1 = threading.Thread(target=gnt_to_img, args=(HWDB0_test_gnt_dir, test_img_dir)).start()
    test_thread_2 = threading.Thread(target=gnt_to_img, args=(HWDB2_test_gnt_dir, test_img_dir)).start()
    train_thread_1.join()
    train_thread_2.join()
    test_thread_1.join()
    test_thread_2.join()
    

