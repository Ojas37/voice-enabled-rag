import os
import pandas as pd

def extract_subset():
    large_path = "data/hinval.parquet"
    mini_path = "data/hinval_real_mini.parquet"
    
    print(f"Reading real dataset from {large_path}...", flush=True)
    try:
        # Read the large validation file
        df = pd.read_parquet(large_path)
        print(f"Loaded full validation dataset. Shape: {df.shape}", flush=True)
        
        # Take the first 200 rows
        df_mini = df.head(200)
        print(f"Extracted first 200 rows. Shape: {df_mini.shape}", flush=True)
        
        # Save to a mini parquet file
        df_mini.to_parquet(mini_path, index=False)
        print(f"Successfully saved real mini dataset to {mini_path}!", flush=True)
        
        # Verify the columns and a sample query
        print("\nVerifying mini dataset features:", flush=True)
        print("Columns:", list(df_mini.columns), flush=True)
        print("Sample Query:", df_mini.iloc[0]['query'], flush=True)
        print("Sample English Query:", df_mini.iloc[0]['Eng_Query'], flush=True)
        print("Sample Answer:", df_mini.iloc[0]['Answer'], flush=True)
        
    except Exception as e:
        print(f"Error during subset extraction: {e}", flush=True)

if __name__ == "__main__":
    extract_subset()
