# coding=gbk

from model import *
from utils import *
import numpy as np
from keras.optimizers import Adam


#---------------------------------设置参数-------------------------------------
steps = 1000
epochs = 3000
batch_size = 32
steps_per_epoch = 625
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
data_path = "datasets/"
save_path = "save_models/"
#-----------------------------------------------------------------------------


#-------------------------------加载数据生成器----------------------------------
data_generator = data_gen(
    data_path,
    batch_size=batch_size,
    target_size=(64, 64),
    seed=10)
#-----------------------------------------------------------------------------


#--------------------------------搭建模型--------------------------------------
unet = UNet(
    steps=steps,
    width=[64,128,256],
    depth=2,
    activation='swish',
    pooling='aver_pooling',
    upsample='up_sample'
    )
unet.build([(None,64,64,3), (None,)])
unet.summary()
unet.compile(
    optimizer=Adam(0.0001),
    loss='mse',
    metrics=['acc']
    )
unet.load_weights(save_path+'unet.h5')
#-----------------------------------------------------------------------------


#---------------------------------训练模型-------------------------------------
# 定义一个线性variance schedule
alpha = 1 - 0.02 * np.arange(1, steps + 1) / steps
bar_alpha = np.cumprod(alpha)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    for step in range(steps_per_epoch):
        x, y = get_batch_train_data(data_generator, steps, bar_alpha)
        loss, acc = unet.train_on_batch(x, y)
        epoch_loss += loss/steps_per_epoch
        epoch_acc += acc/steps_per_epoch

    print(f'epoch:{epoch}\tloss:{epoch_loss}\tacc:{epoch_acc}')

    if epoch%10 == 0:
        unet.save_weights(save_path+f'unet_{epoch}.h5')
#----------------------------------------------------------------------------