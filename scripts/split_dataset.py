# scripts/split_dataset.py

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

def split_dataset(df, val_size=0.1, test_size=0.1, seed=42):
    """
    Args:
        df
        val_size
        test_size
        seed
    
    Returns:
        train_df, val_df, test_df
    """
    total_samples = len(df)
    if total_samples < 10: 
        raise ValueError(f"Too few samples ({total_samples}). Need at least 10 samples for splitting.")
        

    total_split = val_size + test_size
    if total_split >= 1.0:
        raise ValueError(f"Val_size ({val_size}) + test_size ({test_size}) should be less than 1.0")
        

    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        random_state=seed
    )
    

    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=seed
    )
    
    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train, validation and test sets")
    
    parser.add_argument("-input", type=str, required=True,
                       help="Input features file")
    parser.add_argument("-train", type=str, required=True,
                       help="Output training set file")
    parser.add_argument("-val", type=str, required=True,
                       help="Output validation set file")
    parser.add_argument("-test", type=str, required=True,
                       help="Output test set file")
    parser.add_argument("-val_size", type=float, default=0.1,
                       help="Validation set size (default: 0.1)")
    parser.add_argument("-test_size", type=float, default=0.1,
                       help="Test set size (default: 0.1)")
    parser.add_argument("-seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
        

    for output_file in [args.train, args.val, args.test]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                print(f"Error creating directory '{output_dir}': {e}")
                sys.exit(1)
    
    try:
        print(f"Reading data from {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Total samples: {len(df)}")
        

        if df.empty:
            raise ValueError("The input file is empty.")
            
        print("Splitting dataset...")
        train_df, val_df, test_df = split_dataset(
            df,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed
        )
        

        print("Saving datasets...")
        train_df.to_csv(args.train, index=False)
        val_df.to_csv(args.val, index=False)
        test_df.to_csv(args.test, index=False)
        

        print("\nDataset split complete:")
        print(f"Total samples: {len(df)}")
        print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        print("\nFiles saved:")
        print(f"Training set: {args.train}")
        print(f"Validation set: {args.val}")
        print(f"Test set: {args.test}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()