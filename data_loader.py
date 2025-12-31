data_loader_content = """
import pandas as pd
import numpy as np
import os

class GlucoseDataLoader:
    def __init__(self, raw_path='data/raw/original_nir_data.csv', 
                 processed_path='data/processed/engineered_calibration_data.csv'):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def load_processed_data(self):
        if not os.path.exists(self.processed_path):
            raise FileNotFoundError(f"Processed data not found at {self.processed_path}")

        df = pd.read_csv(self.processed_path)
        return df

    def get_train_test_split(self, test_size=0.2, cnn_format=False):
        from sklearn.model_selection import train_test_split
        df = self.load_processed_data()

        X = df[['Voltage']]
        y = df['Glucose']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if cnn_format:
            # Reshape for 1D-CNN (Samples, Steps, Features)
            X_train = X_train.values.reshape(X_train.shape[0], 1, 1)
            X_test = X_test.values.reshape(X_test.shape[0], 1, 1)

        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    loader = GlucoseDataLoader()
    print("Data Loader Initialized Successfully.")
"""

# Save to your repo folder
with open("data_loader.py", "w") as f:
    f.write(data_loader_content)

print("[SUCCESS]'data_loader.py' has been created.")