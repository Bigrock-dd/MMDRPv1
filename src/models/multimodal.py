import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from .loss import PCC_RMSE

class CBAM(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(filters, ratio)
        self.spatial_attention = SpatialAttention()
        
    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        self.dense1 = Dense(filters // ratio, activation='relu')
        self.dense2 = Dense(filters, activation='sigmoid')
        
    def call(self, inputs):
        avg_x = self.avg_pool(inputs)
        max_x = self.max_pool(inputs)
        avg_x = self.dense1(avg_x)
        max_x = self.dense1(max_x)
        avg_x = self.dense2(avg_x)
        max_x = self.dense2(max_x)
        x = avg_x + max_x
        return tf.multiply(inputs, tf.expand_dims(x, 1))

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        
    def call(self, inputs):
        avg_x = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_x = tf.reduce_max(inputs, axis=-1, keepdims=True)
        x = tf.concat([avg_x, max_x], axis=-1)
        x = self.conv(x)
        return tf.multiply(inputs, x)

class MultiModalNet:
    """多模态神经网络模型"""
    
    def __init__(self, shell_input_shape, gnn_input_shape, lr=0.001, dropout=0.2):
        self.shell_input_shape = shell_input_shape
        self.gnn_input_shape = gnn_input_shape
        self.lr = lr
        self.dropout = dropout
        self.model = None

    def channel_attention(self, inputs, ratio=8):
        channel = inputs.shape[-1]
        
        # 平均池化
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(avg_pool)
        avg_pool = Dense(channel, kernel_initializer='he_normal', use_bias=False)(avg_pool)
        
        # 最大池化
        max_pool = GlobalMaxPooling2D()(inputs)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(max_pool)
        max_pool = Dense(channel, kernel_initializer='he_normal', use_bias=False)(max_pool)
        
        # 融合
        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        
        return Multiply()([inputs, cbam_feature])

    def spatial_attention(self, inputs):
        # 计算空间注意力
        avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(inputs)
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        
        # 空间注意力卷积
        cbam_feature = Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
        
        return Multiply()([inputs, cbam_feature])

    def cbam_block(self, inputs):
        # 通道注意力
        x = self.channel_attention(inputs)
        # 空间注意力
        x = self.spatial_attention(x)
        return x
        
    def build(self):
        """构建多模态模型"""
        # 使用Adam优化器
        optimizer = Adam(learning_rate=self.lr)
        
        # 壳层特征分支
        shell_input = Input(shape=self.shell_input_shape, name='shell_input')
        
        # 将1D输入转换为2D以使用CBAM
        x1 = Reshape((self.shell_input_shape[0], 1, 1))(shell_input)
        
        # 第一个卷积块
        x1 = Conv2D(32, (3, 1), padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = self.cbam_block(x1)
        
        # 第二个卷积块
        x1 = Conv2D(64, (3, 1), padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = self.cbam_block(x1)
        
        # Flatten并进行全连接
        x1 = Flatten()(x1)
        x1 = Dense(256, activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(self.dropout)(x1)

        # GNN特征分支
        gnn_input = Input(shape=self.gnn_input_shape, name='gnn_input')
        x2 = Dense(128, activation='relu')(gnn_input)
        x2 = BatchNormalization()(x2)
        x2 = Dense(256, activation='relu')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(self.dropout)(x2)

        # 特征融合
        combined = Concatenate()([x1, x2])
        
        # 融合层
        z = Dense(256, activation='relu')(combined)
        z = BatchNormalization()(z)
        z = Dropout(self.dropout)(z)
        
        z = Dense(128, activation='relu')(z)
        z = BatchNormalization()(z)
        z = Dropout(self.dropout)(z)
        
        z = Dense(64, activation='relu')(z)
        z = BatchNormalization()(z)
        output = Dense(1, activation='linear')(z)

        # 构建模型
        self.model = tf.keras.Model(inputs=[shell_input, gnn_input], outputs=output)
        
        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: PCC_RMSE(y_true, y_pred, alpha=0.1),
            metrics=['mse']
        )
        
        # 打印模型结构
        self.model.summary()
        
        return self.model