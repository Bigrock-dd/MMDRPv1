# scripts/train_model.py

import argparse
import os
import numpy as np
import pandas as pd
from src.models.trainer import ModelTrainer
from src.utils.data_utils import load_and_process_data

def main():
    parser = argparse.ArgumentParser(
        description="Train the protein-ligand binding model")
        
    parser.add_argument("-fn_train", type=str, nargs="+",
                       help="Training data files")
    parser.add_argument("-fn_validate", type=str, nargs="+",
                       help="Validation data files")
    parser.add_argument("-fn_test", type=str, nargs="+",
                       help="Test data files")
    parser.add_argument("-reshape", type=int, nargs="+",
                       default=[192, 1],
                       help="Reshape dimensions")
    parser.add_argument("-n_features", type=int, default=192,
                       help="Number of features")
    parser.add_argument("-epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("-batch", type=int, default=128,
                       help="Batch size")
    parser.add_argument("-use_gpu", type=bool, default=True,
                       help="Whether to use GPU")
    parser.add_argument("-model_path", type=str,
                       default="model.h5",
                       help="Path to save model")
    parser.add_argument("-predictions_path", type=str,
                       default="predictions.csv",
                       help="Path to save predictions")
                       
    args = parser.parse_args()
    
    try:
        print("Loading data...")
        
        # 加载数据
        X_shell_train, X_gnn_train, y_train = load_and_process_data(
            args.fn_train, args.n_features, args.reshape)
        print(f"Training data loaded. Shape: {X_shell_train.shape}, {X_gnn_train.shape}")
        
        X_shell_val, X_gnn_val, y_val = load_and_process_data(
            args.fn_validate, args.n_features, args.reshape)
        print(f"Validation data loaded. Shape: {X_shell_val.shape}, {X_gnn_val.shape}")
        
        X_shell_test, X_gnn_test, y_test = load_and_process_data(
            args.fn_test, args.n_features, args.reshape)
        print(f"Test data loaded. Shape: {X_shell_test.shape}, {X_gnn_test.shape}")
        
        print("Creating trainer...")
        trainer = ModelTrainer(
            shell_input_shape=(args.reshape[0], args.reshape[1]),
            gnn_input_shape=(64,),
            use_gpu=args.use_gpu
        )
        
        print("Starting training...")
        trainer.train(
            X_shell_train, X_gnn_train, y_train,
            X_shell_val, X_gnn_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch
        )
        
        print("Evaluating model...")
        metrics = trainer.evaluate(X_shell_test, X_gnn_test, y_test)
        
        # 创建保存目录
        model_dir = os.path.dirname(args.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            
        # 保存模型和预测结果
        print(f"\nSaving model to {args.model_path}")
        trainer.save_model(args.model_path)
        
        print(f"Saving predictions to {args.predictions_path}")
        trainer.save_predictions(
            X_shell_test, X_gnn_test, y_test,
            args.predictions_path
        )
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()