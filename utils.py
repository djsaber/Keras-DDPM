# coding=gbk

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def data_gen(path, batch_size, target_size, **kwargs):
    '''创建图片生成器'''

    def prep_fn(img):
        img = img / 255.0
        img = (img - 0.5) * 2
        return img

    img_gen = ImageDataGenerator(
        preprocessing_function=prep_fn,
        horizontal_flip=True,
        )

    gen = img_gen.flow_from_directory(
        path,
        batch_size=batch_size,
        target_size=target_size,
        **kwargs)

    return gen


def get_batch_train_data(data_generator, steps, bar_alpha):

    # 采样图片和高斯噪音
    batch_imgs = next(data_generator)[0]
    batch_noise = np.random.randn(*batch_imgs.shape)
    # 随机采样step  [0, steps-1]
    batch_steps = np.random.choice(steps, batch_imgs.shape[0])
    # 每步对应的bar_alpha
    batch_bar_alpha = bar_alpha[batch_steps][:, None, None, None]
    # 对图片添加噪声
    batch_noisy_imgs = batch_imgs*np.sqrt(batch_bar_alpha) + batch_noise*np.sqrt(1-batch_bar_alpha)
    
    batch_noisy_imgs = batch_noisy_imgs.astype('float32')
    batch_noise = batch_noise.astype('float32')
    x = [batch_noisy_imgs, batch_steps]
    y = batch_noise
    
    return x, y
    




