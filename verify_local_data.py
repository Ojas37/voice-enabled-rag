import sys
import pandas as pd

# Set sys.stdout and sys.stderr to write in utf-8 to support Devanagari script on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def verify():
    print("Verifying local mini Hindi dataset...")
    df_hi = pd.read_parquet("data/hinval_mini.parquet")
    print("Shape:", df_hi.shape)
    print("Columns:", list(df_hi.columns))
    for idx, row in df_hi.iterrows():
        print(f"\nRecord {idx+1}:")
        print(f"  Query ID: {row['query_id']}")
        print(f"  English Query: '{row['Eng_Query']}'")
        print(f"  Hindi Query:   '{row['query']}'")
        print(f"  English Answer: '{row['Eng_Answer']}'")
        print(f"  Hindi Answer:   '{row['Answer']}'")
        passages = row['passages']
        # passages is saved as a dict of lists in pandas parquet
        print(f"  Passages count: {len(passages['English_passages'])}")
        print(f"  First English passage: '{passages['English_passages'][0]}'")
        print(f"  First Hindi passage:   '{passages['Translated_passages'][0]}'")
        print(f"  is_selected map: {list(passages['is_selected'])}")

    print("\n" + "="*50 + "\n")

    print("Verifying local mini Marathi dataset...")
    df_mr = pd.read_parquet("data/marval_mini.parquet")
    print("Shape:", df_mr.shape)
    print("Columns:", list(df_mr.columns))
    for idx, row in df_mr.iterrows():
        print(f"\nRecord {idx+1}:")
        print(f"  Query ID: {row['query_id']}")
        print(f"  English Query: '{row['Eng_Query']}'")
        print(f"  Marathi Query: '{row['query']}'")
        print(f"  English Answer: '{row['Eng_Answer']}'")
        print(f"  Marathi Answer: '{row['Answer']}'")
        passages = row['passages']
        print(f"  Passages count: {len(passages['English_passages'])}")
        print(f"  First English passage: '{passages['English_passages'][0]}'")
        print(f"  First Marathi passage: '{passages['Translated_passages'][0]}'")
        print(f"  is_selected map: {list(passages['is_selected'])}")

if __name__ == "__main__":
    verify()
