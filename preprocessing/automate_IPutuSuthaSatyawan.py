"""
Automate Preprocessing - IPutuSuthaSatyawan
Fungsi untuk preprocessing otomatis dataset retail
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def automate_preprocessing(input_path, output_path):
    """
    Melakukan preprocessing otomatis
    
    Parameters:
    input_path: path ke dataset raw
    output_path: path untuk menyimpan hasil preprocessing
    """
    
    print(f"[INFO] Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    
    # 1. Handle missing values
    print("[INFO] Handling missing values...")
    df = df.fillna({
        'Promotion': 'No Promotion',
        'Discount_Applied': 'No'
    })
    
    # 2. Konversi Date dan ekstrak fitur
    print("[INFO] Feature extraction from Date...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # 3. Encode categorical variables
    print("[INFO] Encoding categorical variables...")
    categorical_cols = ['Payment_Method', 'Store_Type', 
                         'Customer_Category', 'Season', 
                         'Discount_Applied', 'Promotion']
    
    # Hanya encode kolom yang ada di dataset
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"[INFO] Encoded {col}: {len(le.classes_)} categories")
    
    # 4. Drop unnecessary columns
    print("[INFO] Dropping unnecessary columns...")
    cols_to_drop = ['Transaction_ID', 'Customer_Name', 'Date']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(cols_to_drop, axis=1)
    
    # 5. Save hasil preprocessing
    print(f"[INFO] Saving preprocessed data to {output_path}")
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Preprocessing selesai!")
    print(f"[INFO] Shape akhir: {df.shape}")
    print(f"[INFO] Final columns: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    # Untuk running di GitHub Actions
    input_path = "../dataset_raw/Retail_Transactions_Dataset.csv"
    output_path = "Retail_Transactions_preprocessing.csv"
    
    # Jika file tidak ditemukan, coba path alternatif
    if not os.path.exists(input_path):
        input_path = "dataset_raw/Retail_Transactions_Dataset.csv"
    
    automate_preprocessing(input_path, output_path)