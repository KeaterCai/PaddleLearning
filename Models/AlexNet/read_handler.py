import matplotlib.pyplot as plt
import numpy as np
import codecs
# math.sqrt比**号开更快，但是为同一个数量级
import math 
from PIL import Image, ImageEnhance
from config import train_params

def resize_img(img, size): # size CHW
    '''
    force to resize the img
    '''
    img=img.resize((size[1],size[2]),Image.BILINEAR)
    return img

def random_crop(img, scale=[0.08,1.0],ratio=[3./4 , 4./3]):
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
    img=img.resize((train_params['img_size'][1],train_params['img_size'][2]),Image.BILINEAR)
    return img

def random_crop_scale(img,scale=[0.8,1.0],ratio=[3./4,4./3]):
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
    img=img.resize((train_params['img_size'][1],train_params['img_size'][2]),Image.BILINEAR)
    return img

def rotate_img(img,angle=[-14,15]):
    '''
    rotate the img
    '''
    angle = np.random.randint(*angle)
    img= img.rotate(angle)
    return img

def random_brightness(img):
    '''
    adjust the image brightnass
    valuable passed from Config.py     prob,delta
    valuable passed from Config.py
    '''
    # probability
    prob = np.random.uniform(0,1)
    if prob < train_params['img_process_method']['brightness_prob']:
        brightness_delta= train_params['img_process_method']['brightness_delta']
        delta = np.random.uniform(-brightness_delta,+brightness_delta)+1
        img=ImageEnhance.Brightness(img).enhance(delta)
    return img

def random_contrast(img):
    '''
    adjust the image contrast
    valuable passed from Config.py     prob,delta
    valuable passed from Config.py
    '''
    # probability
    prob = np.random.uniform(0,1)
    if prob < train_params['img_process_method']['contrast_prob']:
        contrast_delta= train_params['img_process_method']['contrast_delta']
        delta = np.random.uniform(-contrast_delta,+contrast_delta)+1
        img=ImageEnhance.Contrast(img).enhance(delta)
    return img

def random_saturation(img):
    '''
    adjust the image 
    valuable passed from Config.py     prob,delta
    valuable passed from Config.py
    '''
    # probability
    prob = np.random.uniform(0,1)
    if prob < train_params['img_process_method']['saturation_prob']:
        saturation_delta= train_params['img_process_method']['saturation_delta']
        delta = np.random.uniform(-saturation_delta,+saturation_delta)+1
        img=ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    '''
    adjust the image 
    valuable passed from Config.py     prob,delta
    valuable passed from Config.py
    '''
    # probability
    prob = np.random.uniform(0,1)
    if prob < train_params['img_process_method']['hue_prob']:
        hue_delta= train_params['img_process_method']['hue_delta']
        delta = np.random.uniform(-hue_delta,+hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:,:,0]=img_hsv[:,:,0]+delta
        img=Image.fromarray(img_hsv,mode='HSV').convert('RGB')
    return img
    
def distort_color(img):
    '''
    randomly Apply different distort order
    '''
    prob = np.random.uniform(0,1)
    if prob< 0.35:
        img=random_brightness(img)
        img=random_contrast(img)
        img=random_saturation(img)
        img=random_hue(img)
    elif prob < 0.7:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img

def custom_img_reader(file_list,mode='train'):
    with codecs.open(file_list) as flist:
        lines=[line.strip() for line in flist]
    def reader():
        # shuffle the data
        np.random.shuffle(lines) ## 这个操作会在内存上直接随机lines 不用赋值
        if mode == 'train':
            for line in lines:
                img_path,label = line.split()
                img=Image.open(img_path)
                plt.imshow(img)
                try:
                    if img.mode!='RGB':
                        img=img.convert('RGB')
                    if train_params['img_process_method']['is_distort']:
                        img = distort_color(img)
                    if train_params['img_process_method']['is_rotate']:
                        img = rotate_img(img)
                    if train_params['img_process_method']['is_crop']:
                        img = random_crop(img)
                    if train_params['img_process_method']['is_flip']:
                        prob = np.random.randint(0,2)
                        if prob == 0:
                            img=img.transpose(Image.FLIP_LEFT_RIGHT)
                    img=np.array(img).astype(np.float32)
                    img -= train_params['mean_rgb']
                    img=img.transpose((2,0,1)) # 三个维度的关系HWC to CHW 机器学习用的通道数在最前面
                    img *= 0.007843
                    x_data = img.astype(np.float32)
                    y_data = np.array([label]).astype(np.int64)
                    yield x_data,y_data
                except Exception as e:
                    print('x\n')
                    pass
        if mode == 'eval':
            for line in lines:
                img_path,label = line.split()
                img=Image.open(img_path)
                if img.mode!='RGB':
                    img=img.convert('RGB')
                img=resize_img(img,train_params['img_size'])
                img=np.array(img).astype(np.float32)
                img -= train_params['mean_rgb']
                img=img.transpose((2,0,1))
                img *= 2./255               
                yield img,int(label)
        if mode == 'test':
            for line in lines:
                img_path = line
                img=Image.open(img_path)
                if img.mode!='RGB':
                    img=img.convert('RGB')
                img=resize_img(img,train_params['img_size'])
                img=np.array(img).astype(np.float32)
                img -= train_params['mean_rgb']
                img=img.transpose((2,0,1))
                img *= 2./255
                yield img
    return reader

if __name__=='__main__':
    
    ## 测试方法部分的代码正确性
    test_img_path='/home/aistudio/data/data504/vegetables/train_imgs/1515827042897.jpg'
    img=Image.open(test_img_path)
    plt.subplot(1,2,1)
    plt.imshow(img)
    img=rotate_img(img) # distort(img) random_crop(img) random_crop_scale
    plt.subplot(1,2,2)
    plt.imshow(img)   
    img=img.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(img)
    img=np.array(img).astype(np.float32)

    img -= train_params['mean_rgb']
    img=img.transpose((2,0,1)) # 三个维度的关系HWC to CHW 机器学习用的通道数在最前面
    img *= 0.007843 
    ## 测试reader部分的代码正确性
    img_list_path='/home/aistudio/data/data504/vegetables/train_labels.txt'
    reader = custom_img_reader(img_list_path)
    print(next(reader())[0].shape)
     
                    
                    
            
        