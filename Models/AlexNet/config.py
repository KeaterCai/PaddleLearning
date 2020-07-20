import os
import logging
import codecs

__log_dir='/home/aistudio/logs'

train_params = {
    "img_size": [3,224,224],
    "class_num": -1,
    "image_num": -1,
    "label_dict":{},
    "data_dir": '/home/aistudio/data/data504/vegetables',
    'train_list': 'train_labels.txt',
    'eval_list': 'eval_labels.txt',
    'label_list': 'label_list.txt',
    "continue_train": False,
    "pretrained": False,
    'pretrained_dir': '',
    'mode':'train',
    'num_epoch': 10,
    'batch_size': 64,
    'mean_rgb':[127.5,127.5,127.5],
    'use_gpu':False,
    'img_process_method': {
        'is_distort':True,
        'is_rotate':True,
        'is_crop':True,
        'is_flip':True,
        'hue_prob':0.5,
        'hue_delta':18,
        'contrast_prob':0.5,
        'contrast_delta':0.5,
        'saturation_prob':0.5,
        'saturation_delta':0.5,
        'brightness_prob':0.5,
        'brightness_delta':0.125
    },
    'early_stop': { # 没发现用到的地方
        'sample_frequency':50,
        'successsive_limit':3,
        'good_acc': 0.92
    },
    'learning_strategy':{ # 有的参数在没有用到
        'name':'cosins_decay',
        'epochs':[40,80,100],
        'steps':[0.1,0.01,0.001,0.0001]
    },
    'learning_rate':0.0125
}
def init_train_parameters():
    label_list=os.path.join(train_params['data_dir'],train_params['label_list'])
    train_list=os.path.join(train_params['data_dir'],train_params['train_list'])
    class_num=0
    with codecs.open(label_list,encoding='utf-8') as l_list:
        lines=[line.strip() for line in l_list]
        for line in lines:
            parts=line.strip().split()
            train_params['label_dict'][parts[1]]=int(parts[0])
            class_num += 1
        train_params['class_num']=class_num
    with codecs.open(train_list,encoding='utf-8') as t_list:
        lines =[line.strip() for line in t_list]
        train_params['image_num']=len(lines)

    
def init_log_config():
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    if not os.path.exists(__log_dir):
        os.mkdir(__log_dir)
    log=os.path.join(__log_dir,'train_log.log')
    sh=logging.StreamHandler() # 输出到命令行的Handler
    fh=logging.FileHandler(log,mode='w')
    fh.setLevel(logging.DEBUG)
    formatter= logging.Formatter('{asctime:s} - {filename:s}[line:{lineno:d}] - {levelname:s} : {message:s}',style='{')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

logger = init_log_config()
init_train_parameters()

if __name__=='__main__':
    logger.info('successfully log')
    print(train_params)

    