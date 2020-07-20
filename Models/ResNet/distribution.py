# -*- coding: utf-8 -*-
# Personal implemetation
import os
import codecs
import shutil
import random
from PIL import Image

data_dir='/home/aistudio/data/data2815'
# 解压文件

os.system('cd /home/aistudio/data/data2815 && unzip -q flower_photos.zip')

train_rate=0.8


class_list = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)) and not x.endswith('data') and not x.startswith('.')]
class_list.sort()

train_data_dir=os.path.join(data_dir,'train_data')
eval_data_dir=os.path.join(data_dir,'eval_data')
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
if not os.path.exists(eval_data_dir):
    os.makedirs(eval_data_dir)

train_list=codecs.open(os.path.join(data_dir,'train_list.txt'),'w')
eval_list=codecs.open(os.path.join(data_dir,'eval_list.txt'),'w')

with codecs.open(os.path.join(data_dir,'label_list.txt'),'w') as label_list:
    label_id=0
    for class_dir in class_list:
        one_class_data_dir=os.path.join(data_dir,class_dir)
        label_list.write('{}\t{}\n'.format(label_id,class_dir))
        for single_data in os.listdir(one_class_data_dir):
            single_data_path=os.path.join(one_class_data_dir,single_data)
            try:
                # check if every file is an image , it can be modified
                single_data_file = Image.open(single_data_path)
                if random.uniform(0,1)<= train_rate:
                    shutil.copyfile(single_data_path,os.path.join(train_data_dir,single_data))
                    train_list.write('{0}\t{1}\n'.format(os.path.join(train_data_dir,single_data),label_id))
                else:
                    shutil.copyfile(single_data_path,os.path.join(eval_data_dir,single_data))
                    eval_list.write('{0}\t{1}\n'.format(os.path.join(eval_data_dir,single_data),label_id))
            except Exception:
                print('exist img or other files that can\'t open')
        label_id += 1


train_list.close()
eval_list.close()