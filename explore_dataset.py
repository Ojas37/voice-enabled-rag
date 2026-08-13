import os
from datasets import load_dataset

def explore():
    print("Streaming MSMARCO-XI for Hindi ('hi')...")
    try:
        dataset_hi = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train", streaming=True)
        # Get first example
        example = next(iter(dataset_hi))
        print("\nHindi Dataset Example Keys:", list(example.keys()))
        print("\nHindi Dataset Example details:")
        for k, v in example.items():
            if k == 'passages':
                print(f"  passages: {len(v)} passages (showing first passage structure):")
                print(f"    {v[0]}")
            elif k == 'answers':
                print(f"  answers: {v}")
            elif k == 'query':
                print(f"  query: {v}")
            else:
                val_str = str(v)
                print(f"  {k}: {val_str[:200]}...")
    except Exception as e:
        print(f"Error loading Hindi dataset: {e}")

    print("\n" + "="*50 + "\n")

    print("Streaming MSMARCO-XI for Marathi ('mr')...")
    try:
        dataset_mr = load_dataset("ai4bharat/MSMARCO-XI", "mr", split="train", streaming=True)
        example_mr = next(iter(dataset_mr))
        print("\nMarathi Dataset Example Keys:", list(example_mr.keys()))
        print("\nMarathi Dataset Example details:")
        for k, v in example_mr.items():
            if k == 'passages':
                print(f"  passages: {len(v)} passages (showing first passage structure):")
                print(f"    {v[0]}")
            elif k == 'answers':
                print(f"  answers: {v}")
            elif k == 'query':
                print(f"  query: {v}")
            else:
                val_str = str(v)
                print(f"  {k}: {val_str[:200]}...")
    except Exception as e:
        print(f"Error loading Marathi dataset: {e}")

if __name__ == "__main__":
    explore()
