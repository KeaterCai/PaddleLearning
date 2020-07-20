# -*- coding: utf-8 -*-
import os
import logging
import codecs
# The dir of all logs
__log_dir='/home/aistudio/logs'
mode = 'train'

read_params={
    'input_num':-1,  # -1, 以及空字典和空list 都需要初始化后使用
    'class_num':-1,
    # 'class_list':[],
    'data_dir':'/home/aistudio/data/data2815',
    'train_list':'train_list.txt',
    'label_list':'label_list.txt',
    'label_dict':{},
    'eval_list':'eval_list.txt',
    'test_list':'',
    'mode':mode   
}
handle_params={
    'input_size':[3,224,224],
    "need_distort": True,  # 是否启用图像颜色增强
    "need_rotate": True,   # 是否需要增加随机角度
    "need_crop": True,      # 是否要增加裁剪
    "need_flip": True,      # 是否要增加水平随机翻转
    "hue_prob": 0.5,
    "hue_delta": 18,
    "contrast_prob": 0.5,
    "contrast_delta": 0.5,
    "saturation_prob": 0.5,
    "saturation_delta": 0.5,
    "brightness_prob": 0.5,
    "brightness_delta": 0.125,
    'mean_rgb':[127.5,127.5,127.5],
    'mode':mode
}
def init_read_params():
    # check mode setting
    if read_params['mode'] == 'train':
        input_list=os.path.join(read_params['data_dir'],read_params['train_list'])
    elif read_params['mode'] == 'eval':
        input_list=os.path.join(read_params['data_dir'],read_params['eval_list'])
    elif read_params['mode'] == 'test':
        input_list=os.path.join(read_params['data_dir'],read_params['eval_list'])
    else:
        raise ValueError('mode must be one of train, eval or test')
    
    label_list=os.path.join(read_params['data_dir'],read_params['label_list'])
    
    class_num=0
    with codecs.open(label_list,encoding='utf-8') as l_list:
        lines=[line.strip() for line in l_list]
        for line in lines:
            parts=line.strip().split()
            read_params['label_dict'][parts[1]]=int(parts[0])
            class_num += 1
        read_params['class_num']=class_num
    with codecs.open(input_list,encoding='utf-8') as t_list:
        lines =[line.strip() for line in t_list]
        read_params['input_num']=len(lines)

def init_log_config():
    # syntax mode has  already been checked
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    if not os.path.exists(__log_dir):
        os.mkdir(__log_dir)
    log_path=os.path.join(__log_dir,mode + '.log')
    sh=logging.StreamHandler() # 输出到命令行的Handler
    fh=logging.FileHandler(log_path,mode='w')
    fh.setLevel(logging.DEBUG)
    formatter= logging.Formatter('{asctime:s} - {filename:s} [line:{lineno:d}] - {levelname:s} : {message:s}',style='{')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

init_read_params()
logger = init_log_config()
if __name__ == '__main__':
    print(read_params['input_num'])
    print(read_params['label_dict'])
    logger.info('Times out')