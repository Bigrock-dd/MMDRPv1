# src/models/loss.py

import tensorflow as tf

def PCC_RMSE(y_true, y_pred, alpha=0.1):
    """
    结合PCC和RMSE的自定义损失函数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        alpha: PCC权重
    """
    # 确保输入是浮点型
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 计算PCC
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    
    y_true_var = y_true - y_true_mean
    y_pred_var = y_pred - y_pred_mean
    
    numerator = tf.reduce_sum(y_true_var * y_pred_var)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_var)) * tf.reduce_sum(tf.square(y_pred_var))) + 1e-8
    
    pcc = numerator / denominator
    
    # 计算RMSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse + 1e-8)
    
    # 组合损失
    return (1.0 - pcc) * alpha + rmse * (1.0 - alpha)

def RMSE(y_true, y_pred):
    """均方根误差"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) + 1e-8)

def PCC(y_true, y_pred):
    """皮尔逊相关系数"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    
    y_true_var = y_true - y_true_mean
    y_pred_var = y_pred - y_pred_mean
    
    numerator = tf.reduce_sum(y_true_var * y_pred_var)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_var)) * tf.reduce_sum(tf.square(y_pred_var))) + 1e-8
    
    return numerator / denominator