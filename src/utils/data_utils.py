import os
import numpy as np
import pandas as pd

def load_and_process_data(filenames, n_features, reshape_dims, remove_h=False):
    """
    加载和处理数据
    
    Args:
        filenames: 数据文件列表
        n_features: 特征数量
        reshape_dims: 重塑维度
        remove_h: 是否移除氢原子
        
    Returns:
        X_shell: 壳层特征
        X_gnn: GNN特征
        y: 标签
    """
    X_shell, X_gnn, y = None, None, []
    
    for i, fn in enumerate(filenames):
        if not os.path.exists(fn):
            continue
            
        df = pd.read_csv(fn, index_col=0, header=0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        if i == 0:
            X_shell = df.iloc[:, :n_features].values
            X_gnn = df.iloc[:, n_features:n_features + 64].values
        else:
            X_shell = np.concatenate((X_shell, df.iloc[:, :n_features].values), axis=0)
            X_gnn = np.concatenate((X_gnn, df.iloc[:, n_features:n_features + 64].values), axis=0)
            
        if 'pKa' in df.columns:
            y.extend(df['pKa'].values)
            
    X_shell = X_shell.reshape(-1, reshape_dims[0], reshape_dims[1])
    X_gnn = X_gnn.reshape(-1, 64)
    
    return X_shell, X_gnn, np.array(y).reshape((-1, 1))