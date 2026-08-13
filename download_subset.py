import os
from datasets import load_dataset
import pandas as pd

def download_subset(filename, repo_path, dest_name, limit=200):
    print(f"Streaming {filename} from HF (getting first {limit} records)...", flush=True)
    try:
        # Load the parquet file from huggingface in streaming mode
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            data_files={dest_name: repo_path},
            split=dest_name,
            streaming=True
        )
        
        records = []
        iterator = iter(dataset)
        for i in range(limit):
            try:
                record = next(iterator)
                records.append(record)
                if (i + 1) % 50 == 0 or (i + 1) == limit:
                    print(f"Loaded record {i+1}/{limit}...", flush=True)
            except StopIteration:
                print("Reached end of dataset before limit.", flush=True)
                break
        
        # Save to local mini parquet file
        os.makedirs("data", exist_ok=True)
        dest_path = os.path.join("data", f"{dest_name}_mini.parquet")
        df = pd.DataFrame(records)
        df.to_parquet(dest_path, index=False)
        print(f"Successfully saved {len(records)} records to {dest_path}.\n", flush=True)
    except Exception as e:
        print(f"Error downloading subset for {filename}: {e}\n", flush=True)

def main():
    # Download 200 rows for both Hindi and Marathi validation files
    download_subset("Hindi Validation", "validation/hinval.parquet", "hinval", limit=200)
    download_subset("Marathi Validation", "validation/marval.parquet", "marval", limit=200)

if __name__ == "__main__":
    main()
