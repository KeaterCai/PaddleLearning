#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, you need to use the persistence path as the following:
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[63]:


a = 2
# reset the values and list out the corresponding Commands
get_ipython().run_line_magic('reset', '')
get_ipython().run_line_magic('lsmagic', '')


# In[4]:


import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random


# In[5]:


batch_size=25
buf_size=100
def load_data():
    train_data=fluid.io.batch(paddle.dataset.uci_housing.train(),
        batch_size=batch_size)
    # shuffle buffer_size is not allowed to use syntax for passing
    test_data=fluid.io.batch(
        fluid.io.shuffle(paddle.dataset.uci_housing.test(),buf_size),
        batch_size=batch_size)
    return train_data,test_data


# In[6]:


# Setting Layers
class Regressor(fluid.dygraph.Layer):
    def __init__(self):
        super(Regressor,self).__init__()
        # Layer Setting
        self.fc=Linear(input_dim=13,output_dim=1,act=None)

    def forward(self,inputs):
        x=self.fc(inputs)
        return x


# In[7]:


train_data=paddle.dataset.uci_housing.train();
sampledata=next(train_data())
print(sampledata)
times=fluid.io.shuffle(paddle.dataset.uci_housing.test(),buf_size)
time=next(times())
print(times)
data1,data2=load_data()
print(data1)


# In[8]:


# 训练环境setting
with fluid.dygraph.guard():
    model=Regressor()
    model.train()
    train_data,test_data=load_data()
    opt=fluid.optimizer.SGD(learning_rate=0.001,parameter_list=model.parameters())


# In[54]:


with dygraph.guard(fluid.CPUPlace()):
    EPOCH_NUM=70
    global all_loss
    all_loss=[]
    for epoch_id in range(EPOCH_NUM):
        for iter_id,batch in enumerate(train_data()):

            # batch上面是tuple的特殊结构需要转换一下
            length=len(batch)
            shape=batch[0][0].shape
            x=np.zeros((length,shape[0]),dtype=np.float32)
            y=np.zeros((length,1),dtype=np.float32)
            for i in range(length):
                x[i]=batch[i][0]
                y[i]=batch[i][1]
            
            # 转换成ndarray的结构
            features=dygraph.to_variable(x)
            prices=dygraph.to_variable(y)

            predicts=model(features)

            loss=fluid.layers.square_error_cost(predicts,label=prices)
            avg_loss=fluid.layers.mean(loss)
            if iter_id%10==0:
                print('epoch:{},iter:{},loss:{}'.format(epoch_id,iter_id,avg_loss.numpy()))
            avg_loss.backward()

            opt.minimize(avg_loss)

            all_loss.append(avg_loss.numpy())
            
    fluid.save_dygraph(model.state_dict(),'LR_model')
    print("Sucessfully saved")
            
        


# # Plot
# 
# 打印cost的变化曲线

# In[46]:


# plot the cost 
import matplotlib.pyplot as plt
temp_loss=np.array(all_loss)
iters=np.arange(len(all_loss))
plt.plot(iters,all_loss)
plt.savefig('/home/aistudio/data/times.png')


# In[47]:


def load_single_data_for_predict():
    test_data=fluid.io.shuffle(paddle.dataset.uci_housing.test(),buf_size)
    test_single_data=next(test_data())
    print(test_single_data)
    return test_single_data

            


# In[50]:



test_data=fluid.io.shuffle(paddle.dataset.uci_housing.test(),buf_size)
print(next(a()))


# In[56]:


with dygraph.guard():
    para_dict,_= fluid.load_dygraph('/home/aistudio/LR_model')
    model.set_dict(para_dict)
    model.eval()
    
    test_data,label=load_single_data_for_predict()

    # 注意这句的用法
    test_data=test_data.astype(np.float32)
    print(test_data.dtype)
    test_data=dygraph.to_variable(test_data)
    print(test_data)
    results = model(test_data)
    print(results)
    


# In[11]:


get_ipython().system('reset')
get_ipython().system('ffmpeg -version')


# In[ ]:


def layer_setting()


# In[21]:


def main():
    train_data,test_data=load_data()
    for id,sample in enumerate(train_data()):
        print(id,sample)
    print(test_data())
    with fluid.dygraph.guard():
        y1=np.array(test_data()).astype('float32')
        y = fluid.dygraph.to_variable(y1, zero_copy=False)

main()

