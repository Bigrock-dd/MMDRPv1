# src/models/trainer.py

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from .multimodal import MultiModalNet
from .metrics import rmse, pcc, pcc_rmse

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, shell_input_shape, gnn_input_shape, 
                 lr=0.001, dropout=0.2, alpha=0.1, use_gpu=True):
        """
        初始化训练器
        
        Args:
            shell_input_shape: 壳层特征输入形状
            gnn_input_shape: GNN特征输入形状
            lr: 学习率
            dropout: Dropout率
            alpha: PCC_RMSE损失函数中的权重参数
            use_gpu: 是否使用GPU
        """
        # GPU设置
        if not use_gpu:
            print("Disabling GPU...")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            tf.config.set_visible_devices([], 'GPU')
        else:
            try:
                # 获取可用的GPU列表
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Found {len(gpus)} GPU(s):")
                    for gpu in gpus:
                        print(f"  {gpu.name}")
                else:
                    print("No GPU found, falling back to CPU")
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            except Exception as e:
                print(f"Error setting up GPU: {e}")
                print("Falling back to CPU")
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # 确认当前设备
        devices = tf.config.list_physical_devices()
        print("Available devices:")
        for device in devices:
            print(f"  {device.name}")
            
        # 初始化模型
        self.model = MultiModalNet(
            shell_input_shape=shell_input_shape,
            gnn_input_shape=gnn_input_shape,
            lr=lr,
            dropout=dropout
        )
        self.alpha = alpha
        self.history = None
        
    def _check_data(self, X_shell, X_gnn, y=None):
        """检查并预处理输入数据"""
        # 转换数据类型
        X_shell = np.asarray(X_shell, dtype=np.float32)
        X_gnn = np.asarray(X_gnn, dtype=np.float32)
        
        # 检查并处理形状
        if len(X_shell.shape) != 3:
            X_shell = X_shell.reshape(-1, self.model.shell_input_shape[0], 
                                    self.model.shell_input_shape[1])
        if len(X_gnn.shape) != 2:
            X_gnn = X_gnn.reshape(-1, self.model.gnn_input_shape[0])
            
        if y is not None:
            y = np.asarray(y, dtype=np.float32)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            return X_shell, X_gnn, y
            
        return X_shell, X_gnn
        
    def train(self, X_shell_train, X_gnn_train, y_train,
              X_shell_val=None, X_gnn_val=None, y_val=None,
              epochs=100, batch_size=128, callbacks=None):
        """训练模型"""
        try:
            # 数据预处理
            print("Preprocessing training data...")
            X_shell_train, X_gnn_train, y_train = self._check_data(
                X_shell_train, X_gnn_train, y_train)
            print(f"Training data shapes: Shell={X_shell_train.shape}, "
                  f"GNN={X_gnn_train.shape}, y={y_train.shape}")
            
            # 处理验证数据
            validation_data = None
            if all(v is not None for v in [X_shell_val, X_gnn_val, y_val]):
                print("Preprocessing validation data...")
                X_shell_val, X_gnn_val, y_val = self._check_data(
                    X_shell_val, X_gnn_val, y_val)
                validation_data = ([X_shell_val, X_gnn_val], y_val)
                print(f"Validation data shapes: Shell={X_shell_val.shape}, "
                      f"GNN={X_gnn_val.shape}, y={y_val.shape}")
            
            # 构建模型
            if self.model.model is None:
                print("Building model...")
                self.model.build()
            
            # 设置回调函数
            if callbacks is None:
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        mode='min'
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        'best_model.h5',
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=False,
                        mode='min'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6,
                        mode='min'
                    )
                ]
            
            # 训练模型
            print("Starting training...")
            self.history = self.model.model.fit(
                [X_shell_train, X_gnn_train],
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            print("Training completed")
            return self.history
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise
            
    def evaluate(self, X_shell_test, X_gnn_test, y_test):
        """评估模型性能"""
        if self.model.model is None:
            raise ValueError("Model has not been trained yet!")
            
        try:
            # 数据预处理
            X_shell_test, X_gnn_test, y_test = self._check_data(
                X_shell_test, X_gnn_test, y_test)
                
            # 获取预测结果
            print("Making predictions...")
            y_pred = self.model.model.predict([X_shell_test, X_gnn_test])
            
            # 计算各种评估指标
            print("Calculating metrics...")
            metrics = {
                'RMSE': rmse(y_test, y_pred),
                'PCC': pcc(y_test, y_pred),
                'PCC_RMSE': pcc_rmse(y_test, y_pred, self.alpha),
                'MSE': np.mean(np.square(y_test - y_pred)),
                'MAE': np.mean(np.abs(y_test - y_pred)),
                'R2': 1 - np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test)))
            }
            
            print("\nEvaluation Results:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
                
            return metrics
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise
        
    def save_predictions(self, X_shell, X_gnn, y_true, output_file):
        """保存预测结果"""
        # 数据预处理
        X_shell, X_gnn, y_true = self._check_data(X_shell, X_gnn, y_true)
        
        # 预测
        y_pred = self.model.model.predict([X_shell, X_gnn])
        
        # 计算评估指标
        metrics = {
            'RMSE': rmse(y_true, y_pred),
            'PCC': pcc(y_true, y_pred),
            'PCC_RMSE': pcc_rmse(y_true, y_pred, self.alpha),
            'MSE': np.mean(np.square(y_true - y_pred)),
            'MAE': np.mean(np.abs(y_true - y_pred)),
            'R2': 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
        }
        
        # 保存结果
        results_df = pd.DataFrame({
            'y_true': y_true.ravel(),
            'y_pred': y_pred.ravel()
        })
        
        # 添加评估指标到DataFrame
        for metric_name, value in metrics.items():
            results_df[f'metric_{metric_name}'] = value
        
        # 创建输出目录
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 保存结果
        results_df.to_csv(output_file, index=False)
        print(f"\nPredictions and metrics saved to {output_file}")
        print("\nMetrics Summary:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    def save_model(self, filepath):
        """保存模型"""
        if self.model.model is None:
            raise ValueError("No model to save!")
        self.model.model.save(filepath)
        
    def load_model(self, filepath):
        """加载模型"""
        self.model.model = tf.keras.models.load_model(filepath)