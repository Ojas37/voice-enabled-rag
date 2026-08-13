from huggingface_hub import list_repo_files

def list_files():
    try:
        files = list_repo_files(repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset")
        print("Files in ai4bharat/MSMARCO-XI:")
        for f in sorted(files):
            print(f"  {f}")
    except Exception as e:
        print("Error listing repo files:", e)

if __name__ == "__main__":
    list_files()
