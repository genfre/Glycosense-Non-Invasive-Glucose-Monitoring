import pandas as pd
import numpy as np
import xgboost as xgb
import re

def clean_value(val):
    """Extracts a single average float from strings like '{'TEMP': array([31.8, ...])}'"""
    if isinstance(val, str) and 'array' in val:
        # Extract all numbers using regex
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        if numbers:
            # Convert found strings to floats and return the mean
            return np.mean([float(n) for n in numbers if '.' in n or n.isdigit()])
    return val

# 1. Load data
df = pd.read_csv('data/processed/master_training_data.csv')

# 2. Clean 'voltage' and 'temperature' columns specifically
for col in ['voltage', 'temperature']:
    # Apply the cleaner to handle the nested array strings
    df[col] = df[col].apply(clean_value)
    # Force to numeric, turning remaining errors into NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Drop rows that couldn't be cleaned (safety for the model)
df = df.dropna(subset=['voltage', 'glucose'])

# 4. Feature Engineering (Beer-Lambert)
df['absorbance'] = -np.log10((df['voltage'] + 1e-6) / 3.61)

X = df[['absorbance', 'temperature', 'skin_type']].fillna(0)
y = df['glucose']

# 5. Train XGBoost
model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
model.fit(X, y)

# 6. Save
model.save_model("glycosense_model.json")
print(f"--- SUCCESS: Cleaned {len(df)} samples and saved XGBoost model ---")