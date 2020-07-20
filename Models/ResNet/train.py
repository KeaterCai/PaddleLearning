import math
import os
import paddle.fluid as fluid
import numpy as np
from reader import custom_img_reader
from config import read_params,logger
from network import resnet50
learning_params = {
    'learning_rate': 0.0001,
    'batch_size': 64,
    'step_per_epoch':-1,
    'num_epoch': 80,
    'epochs':[10,30],  # Applies piecewise decay to the initial learning rate.
    'lr_decay':[1,0.1,0.01],
    'use_GPU':True,
    'pretrained':False,
    'pretrain_params_path':'',
    'save_params_path':''
}

place = fluid.CPUPlace() if not learning_params['use_GPU'] else fluid.CUDAPlace(0)

def get_momentum_optimizer(parameter_list):
    '''
    piecewise decay to the initial learning rate
    '''
    batch_size = learning_params['batch_size']
    step_per_epoch = int(math.ceil(read_params['input_num'] / batch_size)) # 
    learning_rate = learning_params['learning_rate']

    boundaries = [i * step_per_epoch for i in learning_params['epochs']]
    values = [i * learning_rate for i in learning_params['lr_decay']]
    learning_rate = fluid.layers.piecewise_decay(boundaries,values)
    optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9,parameter_list=parameter_list)
    return optimizer

def get_adam_optimizer(parameter_list):
    '''
    piecewise decay to the initial learning rate
    '''
    batch_size = learning_params['batch_size']
    step_per_epoch = int(math.ceil(read_params['input_num'] / batch_size)) # 
    learning_rate = learning_params['learning_rate']

    boundaries = [i * step_per_epoch for i in learning_params['epochs']]
    values = [i * learning_rate for i in learning_params['lr_decay']]
    learning_rate = fluid.layers.piecewise_decay(boundaries,values)
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate,parameter_list=parameter_list)
    return optimizer

with fluid.dygraph.guard(place):
    resnet = resnet50(False,num_classes=5)
    optimizer = get_adam_optimizer(resnet.parameters())
    if learning_params['pretrained']:
        params,_ = fluid.load_dygraph(learning_params['pretrain_params_path'])
        resnet.set_dict(params)

with fluid.dygraph.guard(place):
    resnet.train()
    train_list = os.path.join(read_params['data_dir'],read_params['train_list'])
    train_reader = fluid.io.batch(custom_img_reader(train_list,mode='train'),batch_size=learning_params['batch_size'])
    train_loader = fluid.io.DataLoader.from_generator(capacity=3,return_list=True,use_multiprocess=False)
    train_loader.set_sample_list_generator(train_reader,places=place)
    eval_list = os.path.join(read_params['data_dir'],read_params['eval_list'])
    eval_reader = fluid.io.batch(custom_img_reader(eval_list,mode='eval'),batch_size=learning_params['batch_size'])

    for epoch_id in range(learning_params['num_epoch']):
        

        for batch_id,single_step_data in enumerate((train_loader())):
            img = fluid.dygraph.to_variable(single_step_data[0])
            label = fluid.dygraph.to_variable(single_step_data[1])
            predict = resnet(img)
            predict = fluid.layers.softmax(predict)
            acc = fluid.layers.accuracy(input=predict,label=label)
            loss = fluid.layers.cross_entropy(predict,label)
            avg_loss = fluid.layers.mean(loss)
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            resnet.clear_gradients()
            if batch_id % 10 == 0:
                
                try:
                    single_eval_data = next(eval_reader())
                    img_eval = np.array([x[0] for x in single_eval_data])
                    label_eval = np.array([x[1] for x in single_eval_data])
                    img_eval = fluid.dygraph.to_variable(img_eval)
                    label_eval = fluid.dygraph.to_variable(label_eval)
                    eval_predict = resnet(img_eval)
                    eval_predict = fluid.layers.softmax(eval_predict)
                    eval_acc = fluid.layers.accuracy(input=eval_predict,label=label_eval)
                    logger.info('Loss epoch {} step {}: {}  acc{} eval_acc {}'.format(epoch_id,batch_id, avg_loss.numpy(),acc.numpy(),eval_acc.numpy()))
                except Exception:  
                    logger.info('Loss epoch {} step {}: {}  acc{} '.format(epoch_id,batch_id, avg_loss.numpy(),acc.numpy()))

            
        logger.info('Final loss:{}'.format(avg_loss.numpy()))
    fluid.save_dygraph(resnet.state_dict(),learning_params['save_params_path'])