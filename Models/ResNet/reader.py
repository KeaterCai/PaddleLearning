# -*- coding: utf-8 -*-
from PIL import Image,ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import math
import codecs
from config import handle_params, read_params
# from Config.py import handle_params, read_params

def resize_img(img, size): # size CHW
    '''
    force to resize the img
    '''
    img=img.resize((size[1],size[2]),Image.BILINEAR)
    return img

def random_crop(img,size,scale=[0.08,1.0],ratio=[3./4 , 4./3]): # size CHW
    '''
    image randomly croped
    scale is the rate of area changing
    ratio is the rate of higth and  
    output image has been resized to the setting input size
    valuable passed from Config.py
    size information
    '''
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    
    bound = min((float(img.size[0])/img.size[1])/(aspect_ratio**2),(float(img.size[1])/img.size[0])*(aspect_ratio**2))
    scale_max=min(scale[1],bound)
    scale_min=min(scale[0],bound)
    
    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size*aspect_ratio)
    h = int(target_size/aspect_ratio)
    
    i = np.random.randint(0,img.size[0] - w + 1)
    j = np.random.randint(0,img.size[1] - h + 1)
    
    img=img.crop((i,j,i+w,j+h))
    img=img.resize((size[1],size[2]),Image.BILINEAR)
    return img

def random_crop_scale(img,size,scale=[0.8,1.0],ratio=[3./4,4./3]): # size CHW
    '''
    image randomly croped
    scale rate is the mean element
    First scale rate be generated, then based on the rate, ratio can be generated with limit
    the range of ratio should be large enough, otherwise in order to successfully crop, the ratio will be ignored
    valuable passed from Config.py
    size information
    
    '''
    scale[1]=min(scale[1],1.)
    scale_rate=np.random.uniform(*scale)
    target_area = img.size[0]*img.size[1]*scale_rate
    target_size = math.sqrt(target_area)
    bound_max=math.sqrt(float(img.size[0])/img.size[1]/scale_rate)
    bound_min=math.sqrt(float(img.size[0])/img.size[1]*scale_rate)
    aspect_ratio_max=min(ratio[1],bound_min)
    aspect_ratio_min=max(ratio[0],bound_max)
    if aspect_ratio_max < aspect_ratio_min:
        aspect_ratio = np.random.uniform(bound_min,bound_max)
    else:
        aspect_ratio = np.random.uniform(aspect_ratio_min,aspect_ratio_max)
    
    w = int(aspect_ratio * target_size)
    h = int(target_size / aspect_ratio)
    
    i = np.random.randint(0,img.size[0] - w + 1)
    j = np.random.randint(0,img.size[1] - h + 1)
    img = img.crop((i,j,i+w,j+h))
    img=img.resize((size[1],size[2]),Image.BILINEAR)
    return img

def rotate_img(img,angle=[-14,15]):
    '''
    rotate the img
    '''
    angle = np.random.randint(*angle)
    img= img.rotate(angle)
    return img

def random_brightness(img,prob,delta):
    '''
    adjust the image brightnass
    valuable passed from Config.py     prob,delta
    valuable passed from Config.py
    '''
    # probability
    brightness_prob = np.random.uniform(0,1)
    if brightness_prob < prob:
        brightness_delta = np.random.uniform(-delta,+delta)+1
        img=ImageEnhance.Brightness(img).enhance(brightness_delta)
    return img

def random_contrast(img,prob,delta):
    '''
    adjust the image contrast
    valuable passed from Config.py     prob,delta
    valuable passed from Config.py
    '''
    # probability
    contrast_prob = np.random.uniform(0,1)
    if contrast_prob < prob:
        contrast_delta = np.random.uniform(-delta,+delta)+1
        img=ImageEnhance.Contrast(img).enhance(contrast_delta)
    return img

def random_saturation(img,prob,delta):
    '''
    adjust the image 
    valuable passed from Config.py     prob,delta
    valuable passed from Config.py
    '''
    # probability
    saturation_prob = np.random.uniform(0,1)
    if saturation_prob < prob:
        saturation_delta = np.random.uniform(-delta,+delta)+1
        img=ImageEnhance.Color(img).enhance(saturation_delta)
    return img


def random_hue(img,prob,delta):
    '''
    adjust the image 
    valuable passed from Config.py     prob,delta
    valuable passed from Config.py
    '''
    # probability
    hue_prob = np.random.uniform(0,1)
    if hue_prob < prob :
        hue_delta = np.random.uniform(-delta,+delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:,:,0]=img_hsv[:,:,0]+hue_delta
        img=Image.fromarray(img_hsv,mode='HSV').convert('RGB')
    return img
    
    # params 使用引用传递
def distort_color(img,params):
    """
    概率的图像增强
    :param img:(param dict包含4种参数)
    :return:
    """
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob < 0.35:
        img = random_brightness(img,params['brightness_prob'],params['brightness_delta'])
        img = random_contrast(img,params['contrast_prob'],params['contrast_delta'])
        img = random_saturation(img,params['saturation_prob'],params['saturation_delta'])
        img = random_hue(img,params['hue_prob'],params['hue_delta'])
    elif prob < 0.7:
        img = random_brightness(img,params['brightness_prob'],params['brightness_delta'])
        img = random_saturation(img,params['saturation_prob'],params['saturation_delta'])
        img = random_hue(img,params['hue_prob'],params['hue_delta'])
        img = random_contrast(img,params['contrast_prob'],params['contrast_delta'])
    return img

def custom_img_reader(input_list,mode='train'): # input_list txt
    with codecs.open(input_list) as flist:
        lines=[line.strip() for line in flist]
    def reader():
        # shuffle the data
        np.random.shuffle(lines) ## 这个操作会在内存上直接随机lines 不用赋值
        if mode == 'train':
            for line in lines:
                img_path,label = line.split()
                img=Image.open(img_path)
                try:
                    if img.mode!='RGB':
                        img=img.convert('RGB')
                    if handle_params['need_distort']:
                        img = distort_color(img,handle_params)
                    if handle_params['need_rotate']:
                        img = rotate_img(img)
                    if handle_params['need_crop']:
                        img = random_crop(img,handle_params['input_size'])
                    if handle_params['need_flip']:
                        prob = np.random.randint(0,2)
                        if prob == 0:
                            img=img.transpose(Image.FLIP_LEFT_RIGHT)
                    img=np.array(img).astype(np.float32)
                    img -= handle_params['mean_rgb']
                    img=img.transpose((2,0,1)) # 三个维度的关系HWC to CHW 机器学习用的通道数在最前面
                    img *= 0.007843
                    label = np.array([label]).astype(np.int64)
                    yield img, label
                except Exception as e:
                    print('Exception occured\n')
                    pass
        if mode == 'eval':
            for line in lines:
                img_path,label = line.split()
                img=Image.open(img_path)
                if img.mode!='RGB':
                    img=img.convert('RGB')
                img=resize_img(img,handle_params['input_size'])
                img=np.array(img).astype(np.float32)
                img -= handle_params['mean_rgb']
                img=img.transpose((2,0,1))
                img *= 0.007843
                label = np.array([label]).astype(np.int64)
                yield img,label
        if mode == 'test':
            for line in lines:
                img_path = line
                img=Image.open(img_path)
                if img.mode!='RGB':
                    img=img.convert('RGB')
                img=resize_img(img,handle_params['input_size'])
                img=np.array(img).astype(np.float32)
                img -= handle_params['mean_rgb']
                img=img.transpose((2,0,1))
                img *= 0.007843
                yield img
    return reader

if __name__ == '__main__':
    test_file_path='/home/aistudio/data/data2815/sunflowers/20406385204_469f6749e2_n.jpg'
    test_img=Image.open(test_file_path)
    plt.subplot(1,2,1)
    plt.imshow(test_img)
    test_img=random_contrast(test_img,handle_params['contrast_prob'],handle_params['contrast_delta'])
    test_img=random_saturation(test_img,handle_params['saturation_prob'],handle_params['saturation_delta'])
    test_img=random_hue(test_img,handle_params['hue_prob'],handle_params['hue_delta'])
    test_img=random_brightness(test_img,handle_params['brightness_prob'],handle_params['brightness_delta'])
    test_img=random_crop(test_img,handle_params['input_size'])
    plt.subplot(1,2,2)
    plt.imshow(test_img)
    input_list='/home/aistudio/data/data2815/train_list.txt'
    reader = custom_img_reader(input_list)
    print(next(reader()))