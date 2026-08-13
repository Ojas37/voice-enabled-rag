from datasets import load_dataset, get_dataset_config_names

def inspect():
    print("Getting config names...")
    try:
        configs = get_dataset_config_names("ai4bharat/MSMARCO-XI")
        print("Available configs:", configs)
    except Exception as e:
        print("Error getting config names:", e)

    print("\nStreaming default config...")
    try:
        dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
        example = next(iter(dataset))
        print("Dataset Keys:", list(example.keys()))
        print("Sample Data:")
        for k, v in example.items():
            val_str = str(v)
            print(f"  {k}: {val_str[:300]}...")
            if isinstance(v, dict):
                print(f"  Keys inside {k}: {list(v.keys())}")
    except Exception as e:
        print("Error streaming default config:", e)

if __name__ == "__main__":
    inspect()
