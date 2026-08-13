import sys
import pandas as pd

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def verify():
    path = "data/hinval_real_mini.parquet"
    print(f"Reading mini real dataset from {path}...")
    try:
        df = pd.read_parquet(path)
        print("Dataset loaded successfully!")
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        
        # Display first 3 examples
        for idx in range(3):
            row = df.iloc[idx]
            print(f"\n--- Example {idx+1} ---")
            print(f"Query ID: {row['query_id']}")
            print(f"Query Type: {row['query_type']}")
            print(f"Source Lang: {row['source_lang']} | Target Lang: {row['target_lang']}")
            print(f"English Query: '{row['Eng_Query']}'")
            print(f"Hindi Query:   '{row['query']}'")
            print(f"English Answer: '{row['Eng_Answer']}'")
            print(f"Hindi Answer:   '{row['Answer']}'")
            
            passages = row['passages']
            # passages is a dictionary containing list of English and Translated passages, and is_selected
            eng_passages = passages.get('English_passages', [])
            trans_passages = passages.get('Translated_passages', [])
            is_selected = passages.get('is_selected', [])
            
            print(f"Passages count: {len(eng_passages)}")
            if len(eng_passages) > 0:
                print(f"  First English passage: '{eng_passages[0]}'")
                print(f"  First Hindi passage:   '{trans_passages[0]}'")
                print(f"  is_selected: {list(is_selected)}")
                # find indices where is_selected is 1
                ground_truth_indices = [i for i, val in enumerate(is_selected) if val == 1]
                print(f"  Ground truth passage index: {ground_truth_indices}")
                
    except Exception as e:
        print("Error during verification:", e)

if __name__ == "__main__":
    verify()
