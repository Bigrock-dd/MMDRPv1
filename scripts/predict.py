import argparse
import pandas as pd
import tensorflow as tf
from src.models.trainer import ModelTrainer
from src.utils.data_utils import load_and_process_data
from src.models.multimodal import custom_loss
import numpy as np

def load_model(model_path):
    """Load the trained model from the specified path."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"custom_loss": custom_loss}  
    )

def preprocess_input(input_file):
    """Load and preprocess input data."""
    print(f"Loading input data from {input_file}...")
    data = pd.read_csv(input_file)


    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.shape[1] < 256:  
        raise ValueError("Insufficient numeric columns in input data for model inputs.")
    

    shell_features = numeric_data.iloc[:, :192].values.astype(np.float32)  
    gnn_features = numeric_data.iloc[:, 192:256].values.astype(np.float32) 


    shell_features = shell_features.reshape(-1, 192, 1)
    
    print(f"Shell features shape: {shell_features.shape}")
    print(f"GNN features shape: {gnn_features.shape}")
    
    return [shell_features, gnn_features]

def predict(model, input_data):
    """Generate predictions using the trained model."""
    print("Running predictions...")
    
    if not isinstance(input_data, list) or len(input_data) != 2:
        raise ValueError("Input data must be a list with two elements: [shell_features, gnn_features].")
    
    predictions = model.predict(input_data)
    return predictions

def save_predictions(predictions, output_file):
    """Save predictions to a CSV file."""
    print(f"Saving predictions to {output_file}...")
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_file, index=False)

def main(args):
    # Load the trained model
    model = load_model(args.model)

    # Preprocess input data
    input_data = preprocess_input(args.input)

    # Make predictions
    predictions = predict(model, input_data)

    # Save predictions
    save_predictions(predictions, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict binding affinities with MM-DRPNet")
    parser.add_argument("-m", "--model", required=True, help="Path to the trained model file")
    parser.add_argument("-i", "--input", required=True, help="Path to the input data file (CSV)")
    parser.add_argument("-o", "--output", required=True, help="Path to save the predictions (CSV)")

    args = parser.parse_args()
    main(args)