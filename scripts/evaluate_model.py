import argparse
from src.models.trainer import ModelTrainer
from src.utils.data_utils import load_and_process_data

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the protein-ligand binding model")
    
    parser.add_argument("-model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("-test_data", type=str, nargs="+",
                       required=True,
                       help="Test data files")
    parser.add_argument("-reshape", type=int, nargs="+",
                       default=[192, 1],
                       help="Reshape dimensions")
    parser.add_argument("-n_features", type=int, default=192,
                       help="Number of features")
    parser.add_argument("-output", type=str, default="predictions.csv",
                       help="Output file for predictions")
                       
    args = parser.parse_args()
    

    X_shell_test, X_gnn_test, y_test = load_and_process_data(
        args.test_data, args.n_features, args.reshape)
    

    trainer = ModelTrainer(
        shell_input_shape=(args.reshape[0], args.reshape[1]),
        gnn_input_shape=(64,)
    )
    trainer.model.load(args.model_path)
    

    metrics = trainer.evaluate(X_shell_test, X_gnn_test, y_test)
    print("Test metrics:", metrics)
    

    trainer.save_predictions(
        X_shell_test, X_gnn_test, y_test,
        args.output
    )

if __name__ == "__main__":
    main()