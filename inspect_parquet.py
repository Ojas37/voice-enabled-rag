import sys
from datasets import load_dataset

def inspect():
    print("Loading hintrain.parquet...", flush=True)
    try:
        # Load Hindi training split
        dataset = load_dataset("ai4bharat/MSMARCO-XI", data_files={"train": "train/hintrain.parquet"}, split="train")
        print("Dataset loaded successfully!", flush=True)
        print(f"Number of examples: {len(dataset)}", flush=True)
        print("Schema Features:", dataset.features, flush=True)
        
        # Get first example
        example = next(iter(dataset))
        print("\nFirst example keys:", list(example.keys()), flush=True)
        print("\nFirst example details:", flush=True)
        for k, v in example.items():
            if k == 'passages':
                print(f"  passages: {type(v)} of length {len(v)}", flush=True)
                if len(v) > 0:
                    print(f"    first passage keys: {list(v[0].keys()) if isinstance(v[0], dict) else type(v[0])}", flush=True)
                    print(f"    first passage: {v[0]}", flush=True)
            elif k == 'answers':
                print(f"  answers: {v}", flush=True)
            elif k == 'query':
                print(f"  query: {v}", flush=True)
            else:
                val_str = str(v)
                print(f"  {k}: {val_str[:300]}...", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    inspect()
