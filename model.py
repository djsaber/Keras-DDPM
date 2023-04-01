# coding=gbk

from keras.layers import Layer
from keras.models import Model
from keras.layers import LayerNormalization, Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers import Conv2DTranspose, UpSampling2D, Input
import keras.activations as activations
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm

class ResidualBlock(Layer):
    '''�в�ģ��
    ������
        - width��������
        - activation�������
    '''
    def __init__(
        self, 
        width,
        activation,
        **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.activation = activations.get(activation)

    def build(self, input_shape):
        if input_shape[-1] != self.width:
            self.res_conv = self.conv1 = Conv2D(self.width, kernel_size=1, padding="same")
        self.normalize = LayerNormalization()
        self.conv1 = Conv2D(self.width, kernel_size=3, padding="same")
        self.conv2 = Conv2D(self.width, kernel_size=3, padding="same")
        super().build(input_shape)

    def call(self, inputs):
        if inputs.shape[-1] == self.width:
            res = inputs
        else:
            res = self.res_conv(inputs)
        x = self.normalize(inputs)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x+res


class DownBlock(Layer):
    '''�²���ģ��
    ������
        - width��ģ����
        - depth��ģ�����
        - activation�������
        - pooling���ػ�����, 'max_pooling', 'aver_pooling'
    '''
    def __init__(
        self, 
        width, 
        depth, 
        activation='swish', 
        pooling='aver_pooling', 
        **kwargs
        ):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.activation = activation
        self.pooling = pooling

    def build(self, input_shape):
        self.res_blocks = [
            ResidualBlock(self.width, self.activation)
            for _ in range(self.depth)]
        if self.pooling == 'max_pooling':
            self.pool = MaxPooling2D()
        elif self.pooling == 'aver_pooling':
            self.pool = AveragePooling2D()
        super().build(input_shape)

    def call(self, inputs):
        x = inputs
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.pool(x), x


class UpBlock(Layer):
    '''�ϲ���ģ��
    ������
        - width��ģ����
        - depth��ģ�����
        - activation�������
        - upsample���ϲ�������, 'up_sample', 'de_conv'
    '''
    def __init__(self, width, depth, activation, upsample, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.activation = activation
        self.upsample = upsample

    def build(self, input_shape):
        self.res_blocks = [
            ResidualBlock(self.width, self.activation)
            for _ in range(self.depth)]
        if self.upsample == 'up_sample':
            self.up = UpSampling2D()
        elif self.upsample == 'de_conv':
            self.up = Conv2DTranspose(input_shape[0][-1], 3, 2, 'same')
        super().build(input_shape)

    def call(self, inputs):
        x, skip = inputs
        x = self.up(x)
        x = K.concatenate([x, skip])
        for res_block in self.res_blocks:
            x = res_block(x)
        return x


class Sinusoidal_Embedding(Layer):
    '''ȥ�벽��stpe�ı���㣬��int����Ƕ��Ϊ��h,w,d��
    ������
        - input_dim������ά��
        - embedding_dims������ά��
        - output_size�������״
    '''
    def __init__(
        self, 
        input_dim,
        embedding_dim,
        output_size,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.frequencies = K.exp(
            tf.linspace(
                0., 
                tf.math.log(K.cast(input_dim, 'float32')),
                embedding_dim // 2)
            )
        self.angular_speeds = 2.0 * K.np.pi * self.frequencies
        self.up_sample = UpSampling2D(output_size)

    def call(self, inputs):
        inputs = K.reshape(inputs, (K.shape(inputs)[0],1, 1, 1))
        inputs = K.cast(inputs, dtype='float32')
        embeddings = tf.concat(
            [tf.sin(self.angular_speeds * inputs), 
            tf.cos(self.angular_speeds * inputs)], axis=3
            )
        return self.up_sample(embeddings)


class UNet(Model):
    '''UNet����
    ������
        - steps��ȥ�벽��
        - width�����²���ģ��Ŀ��, [128, 256, 512]
        - depth�����²���ģ������
        - activation�������, 'swish'
        - pooling���²�������, 'max_pooling', 'aver_pooling'
        - upsample���ϲ�������, 'up_sample', 'de_conv'
    '''
    def __init__(
        self, 
        steps,
        width, 
        depth,
        activation,
        pooling,
        upsample,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.steps = steps
        self.width = width
        self.depth = depth
        self.activation = activation
        self.pooling = pooling
        self.upsample = upsample

    def build(self, input_shape):
        x_shape, t_shape = input_shape
        self.t_emb = Sinusoidal_Embedding(
            input_dim = self.steps, 
            embedding_dim = 32,
            output_size = x_shape[1:-1])
        self.down_blocks = [
            DownBlock(width, self.depth, self.activation, self.pooling)
            for width in self.width]
        self.mid = ResidualBlock(2*self.width[-1], self.activation)
        self.up_blocks = [
            UpBlock(width, self.depth, self.activation, self.upsample)
            for width in reversed(self.width)]
        self.out = ResidualBlock(x_shape[-1], self.activation)
        super().build(input_shape)
        self.call([Input(shape[1:]) for shape in input_shape])

    def call(self, inputs):
        x, t = inputs
        x = tf.concat([x, self.t_emb(t)], axis=-1)
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)
        x = self.mid(x)
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block([x, skip])
        x = self.out(x)
        return x