import paddle.fluid as fluid
from network import Alex_net
import numpy as np
import read_handler
import math
import os
from config import train_params,logger

momentum_rate = 0.9
l2_decay=1.2e-4

def get_optimizer(params,parameter_list):
    learning_rate=params['learning_rate']
    image_num=params['image_num']
    
    batch_size=params['batch_size']
    step = int(math.ceil(float(image_num)/batch_size))
    num_epoch=params['num_epoch']
    
    var_learning_rate=fluid.layers.cosine_decay(learning_rate=learning_rate,step_each_epoch=step,epochs=num_epoch)
    
    optimizer=fluid.optimizer.MomentumOptimizer(var_learning_rate,momentum_rate,parameter_list=parameter_list,
        regularization=fluid.regularizer.L2Decay(l2_decay))
    return optimizer

num_epoch=train_params['num_epoch']
batch_size=train_params['batch_size']

def train():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        alexnet=Alex_net('alexnet',train_params['class_num'])
        optimizer = get_optimizer(train_params,alexnet.parameters())
        train_list = os.path.join(train_params['data_dir'],train_params['train_list'])
        train_reader=fluid.io.batch(read_handler.custom_img_reader(train_list),batch_size=batch_size)
        train_loader=fluid.io.DataLoader.from_generator(capacity=5,return_list=True)
        train_loader.set_sample_list_generator(train_reader,places=fluid.CPUPlace())
        for i in range(num_epoch):
            for batch_id, data in enumerate(train_loader()):
                
                img = fluid.dygraph.to_variable(data[0])
                label = fluid.dygraph.to_variable(data[1])
                
                predict,acc =alexnet(img,label)
                loss = fluid.layers.cross_entropy(predict,label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                alexnet.clear_gradients()
                
                if batch_id % 10 == 0:
                    logger.info('Loss epoch {} step {}: {}  acc{}'.format(i,batch_id,avg_loss.numpy(),acc.numpy()))
    logger.info('Final loss:{}'.format(avg_loss.numpy()))

if __name__=='__main__':
    train()                
                
            