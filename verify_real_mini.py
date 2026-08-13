import sys
import pandas as pd

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def verify_file(path, name):
    print(f"\n================ Verifying {name} Mini Dataset ================")
    try:
        df = pd.read_parquet(path)
        print(f"File: {path}")
        print(f"Successfully loaded. Shape: {df.shape}")
        print("Columns:", list(df.columns))
        
        # Verify row count is exactly 5000
        if df.shape[0] == 5000:
            print(f" SUCCESS: Row count is exactly 5,000!")
        else:
            print(f" WARNING: Row count is {df.shape[0]}, expected 5,000.")
            
        # Spot check first example
        row = df.iloc[0]
        print("\nSpot Check (First Example):")
        print(f"  Query ID: {row['query_id']}")
        print(f"  Query Type: {row['query_type']}")
        print(f"  Source Lang: {row['source_lang']} | Target Lang: {row['target_lang']}")
        print(f"  English Query: '{row['Eng_Query']}'")
        print(f"  Target Query:  '{row['query']}'")
        print(f"  English Answer: '{row['Eng_Answer']}'")
        print(f"  Target Answer:  '{row['Answer']}'")
        
        passages = row['passages']
        eng_passages = passages.get('English_passages', [])
        trans_passages = passages.get('Translated_passages', [])
        is_selected = list(passages.get('is_selected', []))
        
        print(f"  Passages Count: {len(eng_passages)}")
        if len(eng_passages) > 0:
            print(f"  First English passage: '{eng_passages[0][:150]}...'")
            print(f"  First Target passage:  '{trans_passages[0][:150]}...'")
            print(f"  is_selected map: {is_selected}")
            selected_indices = [i for i, val in enumerate(is_selected) if val == 1]
            print(f"  Ground truth passage indices: {selected_indices}")
            
    except Exception as e:
        print(f"Error verifying {name}: {e}")

def main():
    verify_file("data/hinval_real_mini.parquet", "Hindi")
    verify_file("data/marval_real_mini.parquet", "Marathi")

if __name__ == "__main__":
    main()
