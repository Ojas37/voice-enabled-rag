import os
import sys
import pandas as pd

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def extract_subset(large_path, mini_path, name, limit=1500):
    print(f"--- Extracting subset for {name} ---", flush=True)
    if not os.path.exists(large_path):
        print(f"Error: {large_path} does not exist.", flush=True)
        return False
        
    try:
        print(f"Reading {large_path}...", flush=True)
        df = pd.read_parquet(large_path)
        print(f"Loaded full dataset. Shape: {df.shape}", flush=True)
        
        # Take the head rows
        df_mini = df.head(limit)
        print(f"Extracted first {limit} rows. Shape: {df_mini.shape}", flush=True)
        
        # Save to local mini parquet file
        os.makedirs(os.path.dirname(mini_path), exist_ok=True)
        df_mini.to_parquet(mini_path, index=False)
        print(f"Successfully saved mini dataset to {mini_path}!", flush=True)
        
        # Verify a sample
        print("Columns:", list(df_mini.columns), flush=True)
        if len(df_mini) > 0:
            row = df_mini.iloc[0]
            print(f"Sample Query ID: {row['query_id']}", flush=True)
            print(f"Sample Eng Query: '{row['Eng_Query']}'", flush=True)
            print(f"Sample Query ({row['target_lang']}): '{row['query']}'", flush=True)
            print(f"Sample Eng Answer: '{row['Eng_Answer']}'", flush=True)
            print(f"Sample Answer: '{row['Answer']}'", flush=True)
            print(f"Passages count: {len(row['passages']['English_passages'])}", flush=True)
            
        return True
    except Exception as e:
        print(f"Error during subset extraction for {name}: {e}", flush=True)
        return False

def main():
    hin_large = "data/hinval.parquet"
    hin_mini = "data/hinval_real_mini.parquet"
    mar_large = "data/marval.parquet"
    mar_mini = "data/marval_real_mini.parquet"
    
    hin_success = extract_subset(hin_large, hin_mini, "Hindi", limit=1500)
    mar_success = extract_subset(mar_large, mar_mini, "Marathi", limit=1500)
    
    # Cleanup large files if extraction was successful
    if hin_success:
        print(f"Cleaning up {hin_large}...", flush=True)
        try:
            os.remove(hin_large)
            print(f"Deleted {hin_large}.", flush=True)
        except Exception as e:
            print(f"Failed to delete {hin_large}: {e}", flush=True)
            
    if mar_success:
        print(f"Cleaning up {mar_large}...", flush=True)
        try:
            os.remove(mar_large)
            print(f"Deleted {mar_large}.", flush=True)
        except Exception as e:
            print(f"Failed to delete {mar_large}: {e}", flush=True)

if __name__ == "__main__":
    main()
