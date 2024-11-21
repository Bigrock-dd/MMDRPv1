# src/models/metrics.py

import numpy as np
from scipy import stats

def rmse(y_true, y_pred):
    """计算RMSE"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    dev = np.square(y_true - y_pred)
    return np.sqrt(np.sum(dev) / y_true.shape[0])

def pcc(y_true, y_pred):
    """计算PCC (Pearson Correlation Coefficient)"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # 使用numpy计算PCC
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    denominator = np.sqrt(np.sum(np.square(y_true - mean_true)) * 
                         np.sum(np.square(y_pred - mean_pred)))
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def pcc_rmse(y_true, y_pred, alpha=0.1):
    """计算PCC_RMSE组合指标"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # 计算RMSE
    dev = np.square(y_true - y_pred)
    r = np.sqrt(np.sum(dev) / y_true.shape[0])
    
    # 计算PCC
    p = pcc(y_true, y_pred)
    
    # 组合指标
    return (1-p)*alpha + r * (1 - alpha)

# 用于Tensorflow/Keras的损失函数版本
def PCC_RMSE_Loss(alpha=0.1):
    """创建PCC_RMSE损失函数"""
    import tensorflow as tf
    
    def loss_fn(y_true, y_pred):
        # 确保输入是浮点型
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 计算PCC
        y_true_mean = tf.reduce_mean(y_true)
        y_pred_mean = tf.reduce_mean(y_pred)
        
        y_true_var = y_true - y_true_mean
        y_pred_var = y_pred - y_pred_mean
        
        numerator = tf.reduce_sum(y_true_var * y_pred_var)
        denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_var)) * 
                            tf.reduce_sum(tf.square(y_pred_var)) + 1e-8)
        
        pcc = numerator / denominator
        
        # 计算RMSE
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        rmse = tf.sqrt(mse + 1e-8)
        
        # 返回组合损失
        return (1.0 - pcc) * alpha + rmse * (1.0 - alpha)
        
    return loss_fn

def evaluate_predictions(y_true, y_pred):
    """评估预测结果，返回多个指标"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'PCC': pcc(y_true, y_pred),
        'PCC_RMSE': pcc_rmse(y_true, y_pred),
        'MSE': np.mean(np.square(y_true - y_pred)),
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'R2': 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    }
    
    return metrics

# 测试代码
if __name__ == "__main__":
    # 生成测试数据
    y_true = np.array([[1], [2], [3], [4], [5]])
    y_pred = np.array([[1.1], [2.1], [2.9], [4.2], [4.8]])
    
    # 测试各个指标
    metrics = evaluate_predictions(y_true, y_pred)
    print("\nTest metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")