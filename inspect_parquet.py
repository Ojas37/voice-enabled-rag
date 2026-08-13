import sys
from datasets import load_dataset

def inspect():
    print("Streaming hinval.parquet...", flush=True)
    try:
        # Load Hindi validation split in streaming mode
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI", 
            data_files={"validation": "validation/hinval.parquet"}, 
            split="validation",
            streaming=True
        )
        print("Dataset streamed successfully!", flush=True)
        print("Features/Schema:", dataset.features, flush=True)
        
        # Get first example
        example = next(iter(dataset))
        print("\nFirst example keys:", list(example.keys()), flush=True)
        print("\nFirst example details:", flush=True)
        for k, v in example.items():
            if k == 'passages':
                print(f"  passages: {type(v)} of length {len(v['English_passages'])}", flush=True)
                if len(v['English_passages']) > 0:
                    print(f"    first English passage: {v['English_passages'][0]}", flush=True)
                    print(f"    first Translated passage: {v['Translated_passages'][0]}", flush=True)
                    print(f"    first passage is_selected: {v['is_selected'][0]}", flush=True)
            elif k == 'answers' or k == 'Answer':
                print(f"  {k}: {v}", flush=True)
            elif k == 'query' or k == 'Eng_Query' or k == 'Eng_Answer':
                print(f"  {k}: {v}", flush=True)
            else:
                val_str = str(v)
                print(f"  {k}: {val_str[:300]}...", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    inspect()
