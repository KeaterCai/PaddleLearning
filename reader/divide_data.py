```
The script divides the data into training set and evaluation set according to a rate.
The information of the training set and  the evaluation set is saved in a txt file
Meanwhile, the information of classes is generated in a txt file, 
```
import os
import codecs
import shutil
import random
from PIL import Image

all_data_dir='/home/aistudio/data/data504/vegetables'
# unzip
if not os.path.exists(all_data_dir):
    os.system('cd /home/aistudio/data/data504 && unzip -q vegetables.zip')
train_rate=0.8


class_list = [x for x in os.listdir(all_data_dir)]
class_list.sort()

train_img_dir=os.path.join(all_data_dir,'train_imgs')
eval_img_dir=os.path.join(all_data_dir,'eval_imgs')
if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)
if not os.path.exists(eval_img_dir):
    os.makedirs(eval_img_dir)

train_labels=codecs.open(os.path.join(all_data_dir,'train_labels.txt'),'w')
eval_labels=codecs.open(os.path.join(all_data_dir,'eval_labels.txt'),'w')

with codecs.open(os.path.join(all_data_dir,'label_list.txt'),'w') as label_list:
    label_id=0
    for class_dir in class_list:
        img_dir_pre=os.path.join(all_data_dir,class_dir)
        if not os.path.isdir(img_dir_pre):
            continue
	# filter the hidden dirs which are useless
        if class_dir[0] is '.':
            continue
        label_list.write('{}\t{}\n'.format(label_id,class_dir))
        for img in os.listdir(img_dir_pre):
            print(img)
            img_dir=os.path.join(img_dir_pre,img)
            try:
                img_file = Image.open(img_dir)
                if random.uniform(0,1)<= train_rate:
                    shutil.copyfile(img_dir,os.path.join(train_img_dir,img))
                    train_labels.write('{0}\t{1}\n'.format(os.path.join(train_img_dir,img),label_id))
                else:
                    shutil.copyfile(img_dir,os.path.join(eval_img_dir,img))
                    eval_labels.write('{0}\t{1}\n'.format(os.path.join(eval_img_dir,img),label_id))
            except Exception:
                print('exist img or other files that can\'t open')
        label_id += 1


train_labels.close()
eval_labels.close()
                

