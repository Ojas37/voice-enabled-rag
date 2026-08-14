import os
import requests
from tqdm import tqdm

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...", flush=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as file, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=os.path.basename(output_path)
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
    print(f"Finished downloading {output_path}.\n", flush=True)

def main():
    files_to_download = {
        "hinval.parquet": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet",
        "marval.parquet": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/marval.parquet"
    }
    
    for filename, url in files_to_download.items():
        dest = os.path.join("data", filename)
        if not os.path.exists(dest):
            download_file(url, dest)
        else:
            print(f"{filename} already exists at {dest}. Skipping.", flush=True)

if __name__ == "__main__":
    main()
