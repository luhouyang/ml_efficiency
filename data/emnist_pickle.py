from pathlib import Path
import pandas as pd
import pickle

# https://medium.com/towards-data-science/why-i-stopped-dumping-dataframes-to-a-csv-and-why-you-should-too-c0954c410f8f

def preprocess_emnist(root, selection='emnist-balanced'):
    """Preprocess both train and test EMNIST dataset CSV files and save them as pickle files."""
    
    # Check if the directory exists
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory {root} does not exist.")

    # Check mapping file
    mapping_path = root_path / f"{selection}-mapping.txt"
    if not mapping_path.exists():
        raise FileNotFoundError(f"File {mapping_path} not found.")

    # Load mapping file
    print(f"Loading mapping file: {mapping_path}...")
    mapping = pd.read_csv(mapping_path, sep=' ')

    for split in ['train', 'test']:
        csv_path = root_path / f"{selection}-{split}.csv"
        pickle_path = root_path / f"{selection}-{split}.pkl"

        if not csv_path.exists():
            print(f"Skipping {csv_path}, file not found.")
            continue

        # Load dataset
        print(f"Loading {csv_path}...")
        dataset = pd.read_csv(csv_path)

        # Extract labels and data
        labels = dataset.iloc[:, 0].values
        data = dataset.iloc[:, 1:].values  # Skip first column (labels)

        # Save as pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump((data, labels, mapping), f)

        print(f"Saved preprocessed data to {pickle_path}")

if __name__ == "__main__":
    root = r'C:\Users\User\Desktop\Python\ml_efficiency\archive'
    preprocess_emnist(root)
